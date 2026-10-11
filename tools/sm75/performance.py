"""Sequential, restart-isolated comparison of the two local distributions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import threading
import time
import urllib.request

import quality as harness

ROOT = Path(__file__).resolve().parents[2]
OLD_FIXTURES = ROOT / 'sm75/fixtures'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('--new-executable', type=Path, required=True)
    parser.add_argument('--old-executable', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--new-profile', type=Path, required=True)
    parser.add_argument('--old-profile', type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / 'artifacts' / args.output
    output.mkdir(parents=True, exist_ok=False)
    binaries = {'old':args.old_executable.resolve(strict=True), 'new':args.new_executable.resolve(strict=True)}
    profile_sources = {'old':args.old_profile.resolve(strict=True), 'new':args.new_profile.resolve(strict=True)}
    profiles = {}
    for version, path in profile_sources.items():
        profiles[version] = output / (version + '-device-profiles.json')
        shutil.copy2(path, profiles[version])
    base_command = json.loads((ROOT / 'sm75/runtime-command.json').read_text())
    base_command[1] = str(args.model.resolve(strict=True))
    base_command += ['--seed', '42']
    requests = {}
    for key, name in [('long-first', 'long-first'), ('long-followup', 'long-followup'),
                      ('long-decode', 'long-decode')]:
        requests[key] = json.loads((OLD_FIXTURES / (name + '.request.json')).read_text(encoding='utf-8'))
    requests['short-decode'] = dict(requests['long-decode'])
    requests['short-decode']['messages'] = [requests['long-decode']['messages'][-1]]
    fixture = json.loads((ROOT / 'sm75/fixtures/fixture-95000.json').read_text(encoding='utf-8'))
    order = ['old', 'new', 'new', 'old', 'old', 'new']
    manifest = {'order': order, 'binaries': {v: {'path': str(p), 'sha256': sha(p)} for v,p in binaries.items()},
                'profile_sha256': {v:sha(p) for v,p in profiles.items()},
                'common_command': base_command[1:], 'environment': {'NINFER_PREFILL_ALIGN':'0'},
                'note': 'Same auto profile policy, each distribution uses its own calibrated routes and shipped CUDA libraries. Startup excluded; same warmup; fresh process per round.',
                'complete': False, 'rounds': []}
    write(output / 'manifest.json', manifest)
    for key, body in requests.items():
        write(output / (key + '.request.json'), body)
    def summarize():
        table = {}
        for version in binaries:
            table[version] = {}
            for key in requests:
                values = [r['results'][key] for r in manifest['rounds'] if r['version']==version and key in r['results']]
                if values:
                    table[version][key] = {metric: {'median': statistics.median([v[metric] for v in values]),
                        'min': min(v[metric] for v in values), 'max': max(v[metric] for v in values)}
                        for metric in ('ttft_s','wall_s','decode_tok_s','prompt_tok_s','completion_tokens','cached_tokens')}
        write(output / 'aggregate.json', table)
        write(output / 'manifest.json', manifest)
    for index, version in enumerate(order, 1):
        for port in (3683,3684):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',port))
        folder = output / f'{index:02d}-{version}'
        folder.mkdir()
        active_profile = output / 'active-device-profiles.json'
        shutil.copy2(profiles[version], active_profile)
        env = {k.upper():v for k,v in os.environ.items() if not k.upper().startswith('NINFER_')}
        env.update(NINFER_PREFILL_ALIGN='0', NINFER_DEVICE_PROFILES=str(active_profile),
                   LOCALAPPDATA=str(output / 'runtime-cache'))
        command = list(base_command)
        command[0] = str(binaries[version])
        command[command.index('--request-log-jsonl')+1] = str(folder / 'requests.jsonl')
        write(folder / 'command.json', command)
        write(folder / 'environment.json', {k:v for k,v in env.items() if k.startswith('NINFER_')})
        run = {'version':version,'index':index,'results':{},'complete':False}
        manifest['rounds'].append(run)
        process = None
        stop = threading.Event()
        samples = []
        def monitor():
            while not stop.is_set():
                result = subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader,nounits'], capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
                samples.append({'time':time.time(),'values':result.stdout.strip()})
                stop.wait(2)
        thread = threading.Thread(target=monitor,daemon=True)
        thread.start()
        try:
            print(f'ROUND {index}/6 {version} START',flush=True)
            with (folder/'stdout.log').open('w') as stdout, (folder/'stderr.log').open('w') as stderr:
                process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            write(folder/'pid.json',{'pid':process.pid})
            deadline = time.monotonic()+900
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'engine exited {process.returncode}')
                try:
                    with urllib.request.urlopen(harness.BASE+'/health',timeout=2) as response:
                        health=json.load(response)
                    if health.get('status') in ('ok','ready') or health.get('ok'):
                        break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('ready timeout')
            print(f'ROUND {index}/6 {version} READY',flush=True)
            harness.generate(folder,'warmup',[{'role':'user','content':'What is 17 + 25? Reply with the number only.'}],64,reasoning_effort='none',temperature=0)
            for key in ('short-decode','long-first','long-followup','long-decode'):
                print(f'ROUND {index}/6 {version} {key} START',flush=True)
                body = requests[key]
                overrides = {k:v for k,v in body.items() if k not in ('messages','max_tokens','seed')}
                result = harness.generate(folder,key,body['messages'],body['max_tokens'],seed=body['seed'],**overrides)
                assert json.loads((folder/(key+'.request.json')).read_text(encoding='utf-8')) == body
                usage=result['usage']; timings=result['timings']
                measure = {'ttft_s':result['ttft_s'],'wall_s':result['wall_s'],
                           'decode_tok_s':timings['predicted_per_second'],
                           'prompt_tok_s':timings['prompt_per_second'],
                           'completion_tokens':usage['completion_tokens'],
                           'reasoning_tokens':usage.get('completion_tokens_details',{}).get('reasoning_tokens'),
                           'prompt_tokens':usage['prompt_tokens'],
                           'cached_tokens':usage.get('prompt_tokens_details',{}).get('cached_tokens',0),
                           'draft_n':timings.get('draft_n'), 'draft_n_accepted':timings.get('draft_n_accepted'),
                           'finish':result['finish']}
                if key in ('long-first','long-followup'):
                    expected=fixture['records'][3:] if key=='long-first' else fixture['records'][:3]
                    measure['records_correct']=sum(f'{k}={v}' in result['content'].splitlines() for k,v in expected)
                if key.endswith('decode'):
                    measure['fixed_length_pass']=usage['completion_tokens']==2048
                run['results'][key]=measure
                summarize()
            run['complete']=True
        except BaseException as error:
            run['error']=repr(error)
            raise
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired: process.kill();process.wait(timeout=10)
            stop.set();thread.join(timeout=6)
            write(folder/'gpu-samples.json',samples)
            run['profile_sha256_after']=sha(active_profile)
            summarize()
        print(f'ROUND {index}/6 {version} COMPLETE',flush=True)
        time.sleep(5)
    manifest['complete']=True
    summarize()
    print('COMPARISON COMPLETE',flush=True)


if __name__=='__main__':
    main()
