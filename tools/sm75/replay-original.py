"""Bounded, uninstrumented replay of the original third-party SM75 distribution.

No scoring, compilation, binary patching, package adoption or automatic retries.
Transport text is observable; model tensor finiteness is explicitly unobserved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO.parents[1]
OUT = ROOT / 'benchmarks/original-sm75-20261014'
ARCHIVE = Path('I:/release-2080ti.7z')
HIST = ROOT / 'archive/20261014-before-rollback/benchmarks/decode-next-20261013/performance/04-old'
Q5 = ROOT / 'releases/fusion-43c39c6-sm75-q5load'
TASKS = ['warmup', 'short-decode', 'long-first', 'long-followup', 'long-decode']
FLAGS = {'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0)}
URL = 'http://127.0.0.1:3683'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def emit(**data):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def prepare():
    require(not OUT.exists(), 'Experiment exists; do not reset its budget')
    OUT.mkdir(parents=True)
    members = subprocess.check_output(['tar', '-tf', str(ARCHIVE)], text=True, **FLAGS).splitlines()
    selected = [m for m in members if m.endswith('.exe') or m.endswith('.dll')]
    require(len(selected) == 10 and all(m.startswith('release-2080ti/') and '..' not in m for m in selected), 'Unexpected original archive members')
    # One streaming decompression: compare every shipped runtime byte to the
    # retained root distribution, without extracting another multi-GB copy.
    identities = {}
    proc = subprocess.Popen(['tar', '-xOf', str(ARCHIVE), *selected], stdout=subprocess.PIPE, stderr=subprocess.PIPE, **FLAGS)
    try:
        for member in selected:
            p = ROOT / Path(member).name
            remaining = p.stat().st_size
            h = hashlib.sha256()
            while remaining:
                block = proc.stdout.read(min(4 * 1024 * 1024, remaining))
                require(block, 'Truncated archive member: ' + member)
                remaining -= len(block)
                h.update(block)
            expected = h.hexdigest()
            require(sha(p) == expected, 'Root distribution differs from archive: ' + member)
            identities[str(p)] = {'archive_member': member, 'bytes': p.stat().st_size, 'sha256': expected}
            emit(verified_original_member=member)
        require(not proc.stdout.read(1), 'Unexpected archive trailing bytes')
        error = proc.stderr.read().decode(errors='replace')
        require(proc.wait(timeout=30) == 0, error)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=15)
    frozen = read(REPO / 'sm75/current-release.json')
    require(all(sha(Q5 / n) == h for n, h in frozen['files'].items()), 'Default q5 release changed')
    model = read(ROOT / 'benchmarks/rollback-20261014/plan.json')['model']
    st = Path(model['path']).stat()
    require((st.st_size, st.st_mtime_ns) == (model['size'], model['mtime_ns']), 'Model stat changed')
    (OUT / 'requests').mkdir()
    inputs = {}
    for name in TASKS:
        source = HIST / (name + '.request.json')
        target = OUT / 'requests' / source.name
        shutil.copy2(source, target)
        require(sha(source) == sha(target), 'Request copy differs')
        inputs[str(target)] = sha(target)
    profile = OUT / 'device-profiles.json'
    shutil.copy2(Q5 / 'device-profiles.json', profile)
    preserved = [Path(__file__), profile, REPO / 'sm75/runtime-command.json', REPO / 'sm75/current-release.json']
    preserved += list(ROOT.glob('start-*.bat'))
    inputs.update({str(p): sha(p) for p in preserved})
    plan = dict(created_unix=time.time(), source_thread='01a11cba-f2bf-7362-b42b-c810272046f1',
                source='Original third-party 杭古同学 release-2080ti.7z, not our 842ea42 build',
                archive=str(ARCHIVE), archive_bytes=ARCHIVE.stat().st_size,
                archive_sha256=sha(ARCHIVE), runtime_files=identities,
                executable=str(ROOT / 'ninfer-serve.exe'), model=model,
                profile=str(profile), inputs=inputs, default_files=frozen['files'],
                runtime_command=read(REPO / 'sm75/runtime-command.json'),
                order=['original-1', 'original-2'], tasks=TASKS,
                limits={'cold_starts': 2, 'requests': 10, 'ready_seconds': 120,
                        'request_seconds': 420, 'model_experiment_seconds': 1200},
                request_policy='Historical request bytes, seeds and per-request output budgets unchanged; server default output budget remains 94208. Fixed warmup included in the 10-request cap.',
                stop_policy='Stop entire experiment on startup/runtime/transport/timeout/configuration failure. No automatic retries, no third run, no extension after a negative result.',
                observations={'illegal_token': 'Original server error and raw SSE',
                              'same_seed_fork': 'Compare decoded content and reasoning separately in the two same-binary sessions; require request and actual configuration agreement',
                              'nan': 'No tensor capture in original EXE: unobserved unless original diagnostics explicitly expose it. Absence of token error does not prove finite tensors.',
                              'quality': 'No ability grading', 'performance': 'No speed claim'},
                limitations=['Original CUDA 13 DLLs differ from q5 CUDA 12 DLLs; whole-package attribution only.',
                             'Third-party adaptation source/patches unavailable; a positive result is not proof of an unmodified upstream defect.',
                             'Current unchanged q5 route profile is explicitly supplied; this is not replay of the author BAT defaults.',
                             'Visible text equality is weaker than actual token-ID or internal tensor equality.'],
                counts={'cold_starts': 0, 'requests': 0}, started_unix=None, closed=False)
    write(OUT / 'plan.json', plan)
    emit(prepared=True, original_exe_sha256=identities[str(ROOT / 'ninfer-serve.exe')]['sha256'], limits=plan['limits'])


def idle():
    script = "@(Get-Process | Where-Object { $_.ProcessName -like 'ninfer*' -or $_.ProcessName -in @('replay','compute-sanitizer','ncu') }).Count"
    p = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script], capture_output=True, text=True, check=True, timeout=20, **FLAGS)
    require(int(p.stdout.strip()) == 0, 'Existing inference process: leave it untouched')
    for port in (3683, 3684):
        with socket.socket() as s:
            s.bind(('127.0.0.1', port))


def events(folder):
    p = folder / 'requests.jsonl'
    if not p.exists():
        return []
    result = []
    for line in p.read_text(encoding='utf-8').splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A live writer can leave an unfinished final line.
    return result


def validate_runtime(folder):
    starts = [e for e in events(folder) if e.get('event') == 'server_start']
    require(len(starts) == 1, 'Missing or duplicate server identity')
    start = starts[0]
    expected = dict(kv_capacity=102400, max_context=102400, kv_cache='rotated-k8g64-v4g32',
                    prefill_chunk=384, gdn_state_fp16=True, max_concurrency=1,
                    speculative_backend='mtp', speculative_draft_window=4,
                    ngram_draft_window=0, ngram_archive_bytes=0, vision=False, cuda_graph=True)
    engine = start['engine']
    require(all(engine.get(k) == v for k, v in expected.items()), 'Actual engine configuration differs')
    cache = engine['context_cache']
    require(cache['host_kv_capacity_bytes'] == 0 and cache['device_state_slots'] == 0 and cache['max_shared_prefixes'] == 0, 'Cache configuration differs')
    server = start['server']
    require(server['default_output_tokens'] == 94208 and server['default_reasoning_effort'] == 'xhigh', 'Server generation defaults differ')
    sampling = start['sampling_defaults']['server_overrides']
    require(sampling['temperature'] == 1 and sampling['top_k'] == 20 and abs(sampling['top_p'] - .95) < 1e-6, 'Sampling defaults differ')
    write(folder / 'runtime-check.json', {'passed': True, 'server_start': start})


def send(request_path, folder, name):
    result = dict(content='', reasoning='', finish=None, stream_done=False, usage={}, timings={})
    began = time.monotonic()
    try:
        request = urllib.request.Request(URL + '/v1/chat/completions', request_path.read_bytes(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=420) as response, (folder / (name + '.sse')).open('xb') as stream:
            result['http_status'] = response.status
            for line in response:
                stream.write(line)
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    result['stream_done'] = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    result['server_error'] = event['error']
                    raise RuntimeError('Server returned streaming error')
                result['usage'] = event.get('usage') or result['usage']
                result['timings'] = event.get('timings') or result['timings']
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    result['content'] += delta.get('content') or ''
                    result['reasoning'] += delta.get('reasoning_content') or ''
                    result['finish'] = choice.get('finish_reason') or result['finish']
        require(result['stream_done'] and result['finish'] in ('stop', 'length'), 'Incomplete response')
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - began
        for channel in ('content', 'reasoning'):
            result[channel + '_sha256'] = hashlib.sha256(result[channel].encode('utf-8')).hexdigest()
        write(folder / (name + '.result.json'), result)
    return result


def run_one(plan, label):
    idle()
    require(all(sha(p) == h for p, h in plan['inputs'].items()), 'Frozen input changed')
    require(all(sha(p) == info['sha256'] for p, info in plan['runtime_files'].items()), 'Original runtime changed')
    st = Path(plan['model']['path']).stat()
    require((st.st_size, st.st_mtime_ns) == (plan['model']['size'], plan['model']['mtime_ns']), 'Model changed')
    if plan['started_unix'] is None:
        plan['started_unix'] = time.time()
    deadline = plan['started_unix'] + plan['limits']['model_experiment_seconds']
    require(time.time() + 120 < deadline, 'Insufficient remaining time')
    folder = OUT / label
    folder.mkdir()
    profile = folder / 'device-profiles.json'
    shutil.copy2(plan['profile'], profile)
    env = {k.upper(): v for k, v in os.environ.items() if not k.upper().startswith('NINFER_')}
    for name in ('CUDA_LAUNCH_BLOCKING', 'CUDA_FORCE_PTX_JIT', 'CUDA_FORCE_JIT', 'CUDA_INJECTION64_PATH'):
        require(env.get(name) in (None, '', '0'), 'Unexpected CUDA override: ' + name)
    env.update(NINFER_PREFILL_ALIGN='0', NINFER_DEVICE_PROFILES=str(profile), LOCALAPPDATA=str(folder / 'runtime-cache'))
    env['PATH'] = str(ROOT) + os.pathsep + env.get('PATH', '')
    replace = {'<engine>': plan['executable'], '<model>': plan['model']['path'], '<request-log>': str(folder / 'requests.jsonl')}
    command = [replace.get(x, x) for x in plan['runtime_command']]
    manifest = dict(label=label, command=command, started_unix=time.time(), requests=[], complete=False)
    write(folder / 'manifest.json', manifest)
    write(folder / 'environment.json', {k: v for k, v in env.items() if k.startswith('NINFER_') or k == 'LOCALAPPDATA'})
    plan['counts']['cold_starts'] += 1
    write(OUT / 'plan.json', plan)
    proc = timer = None
    expired = threading.Event()
    def timeout_stop():
        expired.set()
        if proc is not None and proc.poll() is None:
            proc.terminate()
    try:
        with (folder / 'stdout.log').open('xb') as out, (folder / 'stderr.log').open('xb') as err:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, **FLAGS)
        manifest['pid'] = proc.pid
        write(folder / 'manifest.json', manifest)
        timer = threading.Timer(min(120, deadline - time.time()), timeout_stop)
        timer.daemon = True
        timer.start()
        while not expired.is_set():
            require(proc.poll() is None, 'Server exited before ready')
            try:
                with urllib.request.urlopen(URL + '/health', timeout=2) as response:
                    health = json.load(response)
                if health.get('status') in ('ok', 'ready') or health.get('ok'):
                    break
            except OSError:
                pass
            time.sleep(.5)
        timer.cancel()
        require(not expired.is_set(), 'Startup timeout')
        validate_runtime(folder)
        # Loaded-library paths confirm this is the author's CUDA 13 runtime.
        script = f"(Get-Process -Id {proc.pid}).Modules | Select-Object ModuleName,FileName | ConvertTo-Json -Compress"
        modules = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script], capture_output=True, text=True, timeout=15, **FLAGS)
        write(folder / 'loaded-modules.json', {'returncode': modules.returncode, 'stdout': modules.stdout, 'stderr': modules.stderr})
        emit(session=label, ready=True, pid=proc.pid)
        for name in TASKS:
            require(plan['counts']['requests'] < 10 and time.time() + 15 < deadline, 'Request/time cap reached')
            plan['counts']['requests'] += 1
            write(OUT / 'plan.json', plan)
            request_path = OUT / 'requests' / (name + '.request.json')
            row = {'name': name, 'started_unix': time.time(), 'request_sha256': sha(request_path)}
            manifest['requests'].append(row)
            write(folder / 'manifest.json', manifest)
            timer = threading.Timer(min(420, deadline - time.time()), timeout_stop)
            timer.daemon = True
            timer.start()
            emit(session=label, request=name, started=True)
            result = send(request_path, folder, name)
            timer.cancel()
            require(not expired.is_set(), 'Request deadline reached')
            row.update(finished_unix=time.time(), finish=result['finish'], usage=result['usage'],
                       content_sha256=result['content_sha256'], reasoning_sha256=result['reasoning_sha256'])
            write(folder / 'manifest.json', manifest)
            emit(session=label, request=name, complete=True, elapsed_seconds=result['elapsed_seconds'], usage=result['usage'])
        manifest['complete'] = True
    except BaseException as error:
        manifest['error'] = repr(error)
        emit(session=label, stopped=manifest['error'])
    finally:
        if timer:
            timer.cancel()
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            manifest['process_exit'] = proc.returncode
            manifest['owned_process_stopped'] = proc.poll() is not None
        manifest['deadline_expired'] = expired.is_set()
        manifest['profile_unchanged'] = sha(profile) == sha(plan['profile'])
        manifest['finished_unix'] = time.time()
        write(folder / 'manifest.json', manifest)
    return manifest['complete']


def summarize(plan):
    comparisons, errors, completed = {}, [], 0
    actual_requests = {}
    for label in plan['order']:
        folder = OUT / label
        actual_requests[label] = [e['request'] for e in events(folder) if e.get('event') == 'request_start']
        errors += [{'session': label, 'event': e} for e in events(folder) if e.get('event') == 'request_error']
        for name in TASKS:
            p = folder / (name + '.result.json')
            if p.exists():
                d = read(p)
                completed += bool(d.get('stream_done') and d.get('finish') in ('stop', 'length') and not d.get('error'))
    for index, name in enumerate(TASKS):
        paths = [OUT / label / (name + '.result.json') for label in plan['order']]
        if not all(p.exists() for p in paths):
            continue
        a, b = map(read, paths)
        row = {'both_complete': all(x.get('stream_done') and not x.get('error') for x in (a, b)), 'channels': {}}
        reqs = [actual_requests[label] for label in plan['order']]
        row['actual_requests_equal'] = all(len(v) > index for v in reqs) and reqs[0][index] == reqs[1][index]
        for channel in ('content', 'reasoning'):
            x, y = a[channel], b[channel]
            prefix = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
            row['channels'][channel] = {'equal': x == y, 'lengths': [len(x), len(y)], 'common_prefix_chars': prefix,
                'first_difference_1based': None if x == y else prefix + 1,
                'excerpts': [] if x == y else [v[max(0, prefix-40):prefix+120] for v in (x, y)]}
        comparisons[name] = row
    summary = dict(counts=plan['counts'], completed_requests=completed,
                   elapsed_model_seconds=time.time() - plan['started_unix'] if plan['started_unix'] else 0,
                   comparisons=comparisons, runtime_errors=errors,
                   illegal_token_observed=any('outside the public token domain' in str(e) for e in errors),
                   nan_tensor_status='UNOBSERVED: original binary has no tensor capture',
                   default_package_unchanged=all(sha(Q5 / n) == h for n, h in plan['default_files'].items()),
                   original_runtime_unchanged=all(sha(p) == v['sha256'] for p, v in plan['runtime_files'].items()),
                   frozen_inputs_unchanged=all(sha(p) == h for p, h in plan['inputs'].items()),
                   ability_grading=False, performance_claim=False, limitations=plan['limitations'])
    write(OUT / 'summary.json', summary)
    emit(summary=str(OUT / 'summary.json'), illegal_token_observed=summary['illegal_token_observed'], completed_requests=completed)


def execute():
    plan = read(OUT / 'plan.json')
    require(not plan['closed'] and plan['counts'] == {'cold_starts': 0, 'requests': 0}, 'Experiment already executed; no retries')
    try:
        for label in plan['order']:
            if not run_one(plan, label):
                break
    finally:
        plan.update(closed=True, closed_unix=time.time(), further_runs_authorized=False)
        write(OUT / 'plan.json', plan)
        summarize(plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'execute'])
    args = parser.parse_args()
    {'prepare': prepare, 'execute': execute}[args.action]()
