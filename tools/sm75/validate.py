"""Validate a separately built engine; never connects to or stops an existing service."""
import argparse
import base64
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / 'artifacts'
import quality as harness

def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--cuda-bin', type=Path)
    parser.add_argument('--vision', action='store_true')
    parser.add_argument('--vision-temperature', type=float, default=0,
                        help='Use deterministic sampling for exact screenshot transcription')
    parser.add_argument('--vision-recheck-from', help='Reuse passed non-vision checks from the same executable and launch command')
    parser.add_argument('--long', action='store_true')
    parser.add_argument('--ring', action='store_true')
    parser.add_argument('--no-graph', action='store_true')
    parser.add_argument('--device-profile-path', type=Path, default=ROOT/'sm75/device-profiles.json')
    parser.add_argument('--prefill-chunk', type=int, default=384)
    parser.add_argument('--draft-tokens', type=int, default=4)
    args = parser.parse_args()
    if args.prefill_chunk <= 0 or args.prefill_chunk % 128:
        parser.error('--prefill-chunk must be a positive multiple of 128')
    if not 1 <= args.draft_tokens <= 15:
        parser.error('--draft-tokens must be in [1,15] for MTP')
    if args.vision_recheck_from and (not args.vision or args.long or args.ring):
        parser.error('--vision-recheck-from requires --vision without --long or --ring')
    folder = STAGE / 'validation' / args.name
    folder.mkdir(parents=True, exist_ok=False)
    for port in (3683, 3684):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    env = {k.upper():v for k,v in os.environ.items() if not k.upper().startswith('NINFER_')}
    if args.cuda_bin:
        env['PATH'] = str(args.cuda_bin.resolve(strict=True)) + os.pathsep + env['PATH']
    env['LOCALAPPDATA'] = str(folder / 'runtime-cache')
    env['NINFER_PREFILL_ALIGN'] = '0'
    profile_source = args.device_profile_path.resolve(strict=True)
    active_profile = folder/'device-profiles.json'
    profile_source_sha256 = harness.sha(profile_source)
    shutil.copy2(profile_source, active_profile)
    env['NINFER_DEVICE_PROFILES'] = str(active_profile)
    if args.ring:
        env.update(NINFER_KV_WINDOW='16384', NINFER_KV_RETRIEVE='8192',
                   NINFER_HOST_PAGEABLE='1', NINFER_KV_REUSE_HOSTBACKED='1',
                   NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL='64')
    command = [str(args.engine.resolve(strict=True)), str(args.model.resolve(strict=True)),
               '--host','127.0.0.1','--port','3683','--stats-port','3684',
               '--model-id','qwen3.8-27b','--max-context','102400',
               '--kv-capacity','17920' if args.ring else '102400','--kv-dtype','rk8v4',
               '--host-kv-mib','8192' if args.ring else '0','--prefill-chunk',str(args.prefill_chunk),
               '--device-state-slots','0','--gdn-state-fp16','--spec','mtp','--draft-tokens',str(args.draft_tokens),
               '--ngram-draft-tokens','0','--default-max-tokens','94208',
               '--default-reasoning-effort','xhigh','--max-concurrency','1','--max-shared-prefixes','0',
               '--temperature','1','--top-p','.95','--top-k','20','--device-profile','auto',
               '--cuda-graph-allowance-mib','128','--request-log-jsonl',str(folder/'requests.jsonl')]
    if args.vision:
        command += ['--vision','--vision-residency','overlay','--vision-max-merged','1024']
    if args.no_graph:
        command += ['--no-cuda-graph']
    write(folder/'command.json', command)
    write(folder/'environment.json', {k:v for k,v in env.items() if k.startswith('NINFER_')})
    with Path(command[0]).open('rb') as executable:
        executable_sha256 = hashlib.file_digest(executable, 'sha256').hexdigest()
    results = {'checks':{}, 'complete':False, 'executable_sha256':executable_sha256,
               'parameters':{'prefill_chunk':args.prefill_chunk,'draft_tokens':args.draft_tokens},
               'device_profile_source':str(profile_source),
               'device_profile_sha256':harness.sha(active_profile),
               'device_profile_source_sha256':profile_source_sha256}
    if results['device_profile_sha256'] != profile_source_sha256:
        raise RuntimeError('Device profile changed while copying the test snapshot')
    if args.vision_recheck_from:
        prior_folder = STAGE / 'validation' / args.vision_recheck_from
        prior = json.loads((prior_folder/'summary.json').read_text(encoding='utf-8'))
        prior_command = json.loads((prior_folder/'command.json').read_text(encoding='utf-8'))
        def comparable_command(value):
            value = value.copy()
            value[value.index('--request-log-jsonl')+1] = '<request-log>'
            return value
        def comparable_environment(value):
            value = value.copy()
            if 'NINFER_DEVICE_PROFILES' in value:
                value['NINFER_DEVICE_PROFILES'] = '<device-profile>'
            return value
        nonvision = {k:v for k,v in prior['checks'].items() if not k.startswith('vision_')}
        if (not prior.get('complete') or not nonvision or not all(nonvision.values())
                or prior.get('executable_sha256') != executable_sha256
                or prior.get('device_profile_sha256') != results['device_profile_sha256']
                or comparable_command(prior_command) != comparable_command(command)
                or comparable_environment(json.loads((prior_folder/'environment.json').read_text(encoding='utf-8')))
                    != comparable_environment({k:v for k,v in env.items() if k.startswith('NINFER_')})):
            raise ValueError('Cannot reuse validation from a different or incomplete configuration')
        results['checks'].update(nonvision)
        results['inherited_validation'] = args.vision_recheck_from
        results['prior_vision_checks'] = {k:v for k,v in prior['checks'].items() if k.startswith('vision_')}
        for key in ('long_first', 'long_followup'):
            if key in prior: results[key] = prior[key]
        results['inherited_sampled_peak_gpu_mib'] = prior.get('sampled_peak_gpu_mib')
    process = None
    memory_samples=[]
    stop_monitor=threading.Event()
    def monitor():
        while not stop_monitor.is_set():
            try:
                sample=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],
                                      capture_output=True,text=True,timeout=4,creationflags=subprocess.CREATE_NO_WINDOW)
                used,total=map(int,sample.stdout.strip().splitlines()[0].split(','))
                memory_samples.append({'time':time.time(),'used_mib':used,'total_mib':total})
            except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):
                pass
            stop_monitor.wait(2)
    monitor_thread=threading.Thread(target=monitor,daemon=True)
    monitor_thread.start()
    try:
        results['checks']['executable_matches_before'] = harness.sha(Path(command[0])) == executable_sha256
        results['checks']['device_profile_matches_before'] = (
            harness.sha(active_profile) == profile_source_sha256 == harness.sha(profile_source))
        if not all(results['checks'].values()):
            raise RuntimeError('Validation inputs changed before launch')
        with (folder/'stdout.log').open('w') as stdout, (folder/'stderr.log').open('w') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
        write(folder/'pid.json', {'pid':process.pid})
        deadline = time.monotonic()+900
        while time.monotonic()<deadline:
            if process.poll() is not None:
                raise RuntimeError(f'engine startup exited {process.returncode}')
            try:
                with urllib.request.urlopen(harness.BASE+'/health', timeout=2) as response:
                    health=json.load(response)
                if health.get('status') in ('ok','ready') or health.get('ok'):
                    break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError('engine ready timeout')
        print('READY', process.pid, flush=True)
        write(folder/'models.json', harness.request('/v1/models'))
        if not args.vision_recheck_from:
            r=harness.generate(folder,'arithmetic',[{'role':'user','content':'What is 17 + 25? Reply with the number only.'}],64,reasoning_effort='none',temperature=0)
            results['checks']['arithmetic']=r['content'].strip()=='42'
            toolbody=dict(model='qwen3.8-27b',messages=[{'role':'user','content':'Call report_result with value 42.'}],
                          tools=[{'type':'function','function':{'name':'report_result','description':'Report the result',
                                 'parameters':{'type':'object','properties':{'value':{'type':'integer'}},'required':['value']}}}],
                          tool_choice={'type':'function','function':{'name':'report_result'}},max_tokens=256,reasoning_effort='none')
            tool=harness.request('/v1/chat/completions',toolbody)
            write(folder/'tool.json',tool)
            calls=tool['choices'][0]['message'].get('tool_calls',[])
            results['checks']['tool_call']=bool(calls and calls[0]['function']['name']=='report_result' and json.loads(calls[0]['function']['arguments'])=={'value':42})
            # Cancel after a committed prefill boundary, before the prompt finishes. The identical
            # history is then submitted twice to exercise reuse of any published continuation.
            prompt='Cancellation regression document.\n' + ('The amber service stores immutable records and checks every transaction.\n'*1500) + '\nReply with the single word READY.'
            body=dict(model='qwen3.8-27b',messages=[{'role':'user','content':prompt}],
                      stream=True,return_progress=True,max_tokens=64,reasoning_effort='none',temperature=0)
            request=urllib.request.Request(harness.BASE+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
            progress=[]
            with urllib.request.urlopen(request,timeout=600) as response:
                for line in response:
                    if not line.startswith(b'data:'):continue
                    payload=line[5:].strip()
                    if payload==b'[DONE]':break
                    event=json.loads(payload)
                    if event.get('error'):raise RuntimeError(event['error'])
                    value=event.get('prompt_progress')
                    if value:
                        progress.append(value)
                        if 4096<=value['processed']<value['total']:
                            break
            write(folder/'cancel-progress.json',progress)
            results['checks']['cancel_mid_prefill']=bool(progress and 4096<=progress[-1]['processed']<progress[-1]['total'])
            time.sleep(2)
            for n in (1,2):
                retry=harness.generate(folder,f'cancel-retry-{n}',body['messages'],64,reasoning_effort='none',temperature=0)
                results['checks'][f'cancel_retry_{n}']=retry['content'].strip()=='READY'
            records=[json.loads(line) for line in (folder/'requests.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            cancelled=[record for record in records
                       if record.get('result',{}).get('finish_reason')=='cancelled'
                       or (record.get('event')=='request_error'
                           and record.get('error',{}).get('message')=='client disconnected')]
            write(folder/'cancel-server-records.json',cancelled)
            results['checks']['server_confirmed_cancel']=bool(cancelled)
        if args.vision:
            fixtures=ROOT/'sm75/fixtures/vision'
            for case in json.loads((fixtures/'fixtures.json').read_text(encoding='utf-8'))[:3]:
                image='data:image/png;base64,'+base64.b64encode((fixtures/case['file']).read_bytes()).decode()
                messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':image}},{'type':'text','text':case['prompt']}]}]
                r=harness.generate(folder,'vision-'+case['name'],messages,4096,seed=201,
                                   temperature=args.vision_temperature)
                content=r['content'].strip()
                if content.startswith('```'):content=content.split('\n',1)[1].rsplit('```',1)[0]
                try:actual=json.loads(content)
                except ValueError:actual={}
                results['checks']['vision_'+case['name']]=all(actual.get(k)==v for k,v in case['expected'].items())
        if args.long or args.ring:
            length=32000 if args.ring else 95000
            fixture=json.loads((ROOT/f'sm75/fixtures/fixture-{length}.json').read_text(encoding='utf-8'))
            messages=[{'role':'user','content':fixture['prompt']}]
            r=harness.generate(folder,'long-first',messages,4096,seed=173)
            results['long_first']=r
            results['checks']['long_cold_prompt']=harness.cached_tokens(r)==0
            results['checks']['long_later_records']=all(f'{k}={v}' in r['content'].splitlines() for k,v in fixture['records'][3:])
            messages += [{'role':'assistant','content':r['content']}, {'role':'user','content':'Now extract the verified hotfix IDs for aurora, birch and cedar from the original OFFICIAL_RELEASE_MANIFEST records. Output only package=hotfix, one per line.'}]
            followup=harness.generate(folder,'long-followup',messages,4096,seed=174)
            results['long_followup']=followup
            results['checks']['long_earlier_records']=all(f'{k}={v}' in followup['content'].splitlines() for k,v in fixture['records'][:3])
            if args.ring:
                server_log=(folder/'stderr.log').read_text(encoding='utf-8',errors='replace')
                retrieval=[tuple(map(int, match)) for match in re.findall(
                    r'ring retrieve: which=text preferred=(\d+) restored=(\d+)', server_log)]
                write(folder/'ring-retrieval.json',retrieval)
                results['checks']['ring_active']=('[ring] budgets:' in server_log
                    and r['usage']['prompt_tokens']>int(command[command.index('--kv-capacity')+1]))
                results['checks']['ring_retrieval']=any(preferred>0 and restored>0
                                                        for preferred,restored in retrieval)
        results['complete']=True
    except Exception as error:
        results['error']=repr(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        results['process_exit']=None if process is None else process.poll()
        stop_monitor.set()
        monitor_thread.join(timeout=6)
        results['executable_sha256_after']=harness.sha(Path(command[0]))
        results['device_profile_sha256_after']=harness.sha(active_profile)
        results['device_profile_source_sha256_after']=harness.sha(profile_source)
        results['checks']['executable_unchanged']=(results['executable_sha256_after']==executable_sha256)
        results['checks']['device_profile_unchanged']=(results['device_profile_sha256_after']==profile_source_sha256)
        results['checks']['device_profile_source_unchanged']=(results['device_profile_source_sha256_after']==profile_source_sha256)
        if not all(results['checks'][key] for key in (
                'executable_unchanged','device_profile_unchanged','device_profile_source_unchanged')):
            results['complete']=False
        write(folder/'gpu-memory.json',memory_samples)
        results['sampled_peak_gpu_mib']=max([s['used_mib'] for s in memory_samples]+[results.get('inherited_sampled_peak_gpu_mib') or 0])
        write(folder/'summary.json',results)
        print(json.dumps(results['checks']),flush=True)
    if not all(results['checks'].values()):raise AssertionError('validation checks failed')

if __name__=='__main__':main()
