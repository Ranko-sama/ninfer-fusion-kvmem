"""Frozen binary ablation and reversible retirement of superseded local experiments.

Run with the workspace Python 3.11. No build, calibration, package adoption or
product-source edit is performed. Each run owns exactly one local server.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO.parents[1]
HERE = ROOT / 'benchmarks/rollback-20261014'
ARCHIVE = ROOT / 'archive/20261014-before-rollback'
SOURCE = ROOT / 'upgrade-staging/rollback-sm75-842ea42'
Q5 = ROOT / 'releases/fusion-43c39c6-sm75-q5load'
BASE = ROOT / 'releases/fusion-43c39c6-sm75'
TASKS = ('transaction-patch', 'incremental-build')
FLAGS = {'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0)}
URL = 'http://127.0.0.1:3683'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def bound(p, parent):
    resolved = p.resolve()
    resolved.relative_to(parent.resolve())
    require(resolved != parent.resolve(), 'Refusing root operation')
    return resolved


def prepare():
    require(not (HERE / 'plan.json').exists(), 'Plan already exists; do not reset counts')
    HERE.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=False)
    snap = ARCHIVE / 'workspace-before'
    snap.mkdir()
    (snap / 'working-tree.patch').write_bytes(git('diff', '--binary', 'HEAD'))
    (snap / 'status.txt').write_bytes(git('status', '--short'))
    paths = git('diff', '--name-only', 'HEAD').decode().splitlines()
    paths += ['sm75/current-release.json']
    for name in paths:
        original = REPO / name
        target = snap / 'repository' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    shutil.copy2(ROOT / '项目维护入口.txt', snap / '项目维护入口.txt')
    original = ROOT / 'benchmarks/rk8v4-decode-20261013'
    fixtures = HERE / 'fixtures'
    fixtures.mkdir()
    for name in ('long-code-grade.py', 'long-code-spec.py', 'long-code-model-hash.json'):
        shutil.copy2(original / name, fixtures / name)
    shutil.copytree(original / 'long-code-fixtures', fixtures / 'long-code-fixtures', ignore=shutil.ignore_patterns('__pycache__'))
    (fixtures / 'requests').mkdir()
    for task in TASKS:
        shutil.copy2(original / 'long-code-baseline-default-budget' / (task + '.request.json'), fixtures / 'requests' / (task + '.json'))
    # Bind copied fixtures to the original manifest without changing that historical manifest.
    historical = read(original / 'long-code-fixtures/manifest.json')['sha256']
    copied = {}
    for p in fixtures.rglob('*'):
        if p.is_file():
            rel = p.relative_to(fixtures)
            source = original / rel
            if str(source) in historical:
                require(sha(p) == historical[str(source)], 'Historical fixture changed: ' + str(rel))
            copied[str(p)] = sha(p)
    # Extract the immutable source tree; do not reset the dirty development checkout.
    require(not SOURCE.exists(), 'Candidate source already exists')
    SOURCE.mkdir()
    blob = git('archive', '842ea42', 'src-tree/fusion-engine-src')
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        for member in archive.getmembers():
            bound(SOURCE / member.name, SOURCE)
            require(not member.issym() and not member.islnk(), 'Unexpected source archive link')
        archive.extractall(SOURCE, filter='data')
    executables = {}
    dlls = {}
    for name, folder in (('q5', Q5), ('compat', BASE)):
        manifest = read(folder / 'manifest.json')
        actual = sha(folder / 'ninfer-serve.exe')
        require(actual == manifest['executable_sha256'], 'Release EXE changed')
        executables[name] = {'path': str(folder / 'ninfer-serve.exe'), 'sha256': actual}
        dlls[name] = {p.name: sha(p) for p in folder.glob('*.dll')}
    require(dlls['q5'] == dlls['compat'], 'DLL confound between arms')
    frozen = read(REPO / 'sm75/current-release.json')
    require(all(sha(Q5 / n) == v for n, v in frozen['files'].items()), 'Default release changed')
    model = read(fixtures / 'long-code-model-hash.json')
    st = Path(model['path']).stat()
    require((st.st_size, st.st_mtime_ns) == (model['size'], model['mtime_ns']), 'Model identity changed')
    command = read(REPO / 'sm75/runtime-command.json')
    bound_files = dict(copied)
    for p in [Path(__file__), REPO / 'sm75/runtime-command.json', REPO / 'sm75/current-release.json',
              ROOT / 'start-fusion-sm75-coder390-100k-lan.bat', ROOT / 'start-fusion-sm75-coder390-100k-vision-overlay-lan.bat', Q5 / 'device-profiles.json']:
        bound_files[str(p)] = sha(p)
    for name in TASKS:
        body = read(fixtures / 'requests' / (name + '.json'))
        require((body['max_tokens'], body['temperature'], body['top_p'], body['top_k'], body['reasoning_effort']) == (94208, 1, .95, 20, 'xhigh'), 'Request defaults changed')
    plan = dict(created_unix=time.time(), authority='Explicit 2026-10-14 rollback, cleanup and ablation request; previous paused goal stays archived.',
                variable='Published engine EXE: q5 versus compatibility-only SM75; all later optimization groups removed together.',
                limitation='Grouped ablation, not single-kernel causal attribution; no build reproducibility claim. No instrumentation or C0.',
                profile_note='The old package has a different bundled profile. Both arms explicitly use the unchanged current q5 profile.',
                order=['q5-1', 'compat-1', 'compat-2', 'q5-2'], tasks=list(TASKS),
                limits={'sessions': 4, 'requests': 8, 'ready_seconds': 120, 'request_seconds': 1200, 'experiment_seconds': 3600},
                repeat_gate='Run second pair only if compat-1 passes both original tasks; a failed rollback candidate is not retried.',
                admission='141/141 and 108/108 on both cold sessions, no runtime failure, within-arm content and reasoning exact. Additional full functional/numerical qualification still required for publication.',
                binaries=executables, dlls=dlls['q5'], profile=str(Q5 / 'device-profiles.json'), model=model,
                runtime_command=command, files=bound_files, default_files=frozen['files'],
                development_before={str(REPO / p): sha(REPO / p) for p in paths if p != 'sm75/README.md'},
                source_candidate=str(SOURCE), source_commit=git('rev-parse', '842ea42').decode().strip(),
                source_archive_sha256=hashlib.sha256(blob).hexdigest(), counts={'sessions': 0, 'requests': 0}, experiment_started_unix=None)
    write(HERE / 'plan.json', plan)
    write(SOURCE / 'candidate.json', {k: plan[k] for k in ('source_commit', 'source_archive_sha256', 'binaries', 'limitation')})
    # CPU oracle transport check: original reference code must still score fully after relocation.
    refs = {}
    for task in TASKS:
        f = fixtures / 'long-code-fixtures' / task
        result = subprocess.run([sys.executable, str(fixtures / 'long-code-grade.py'), '--worker', str(f / 'reference.py'), task, str(f / 'cases.json')], capture_output=True, text=True, timeout=30, **FLAGS)
        require(result.returncode == 0, result.stderr)
        grade = json.loads(result.stdout)
        require(grade['passed'] == grade['total'], 'Relocated reference grader failed')
        refs[task] = {'passed': grade['passed'], 'total': grade['total']}
    write(HERE / 'reference-check.json', refs)
    print(json.dumps({'prepared': True, 'reference': refs, 'source': str(SOURCE)}, ensure_ascii=False), flush=True)


def idle():
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', "@(Get-Process | Where-Object { $_.ProcessName -like 'ninfer*' -or $_.ProcessName -in @('replay','compute-sanitizer','ncu') }).Count"], capture_output=True, text=True, check=True, timeout=20, **FLAGS)
    require(int(result.stdout.strip()) == 0, 'Existing GPU process; do not terminate user processes')
    for port in (3683, 3684):
        with socket.socket() as s:
            s.bind(('127.0.0.1', port))


def send(body_path, folder, task):
    raw = body_path.read_bytes()
    result = dict(content='', reasoning='', finish=None, stream_done=False, usage={}, timings={})
    begin = time.monotonic()
    try:
        request = urllib.request.Request(URL + '/v1/chat/completions', raw, {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response, (folder / (task + '.sse')).open('xb') as stream:
            for line in response:
                stream.write(line)
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    result['stream_done'] = True
                    break
                event = json.loads(data)
                require(not event.get('error'), str(event.get('error')))
                result['usage'] = event.get('usage') or result['usage']
                result['timings'] = event.get('timings') or result['timings']
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    result['content'] += delta.get('content') or ''
                    result['reasoning'] += delta.get('reasoning_content') or ''
                    result['finish'] = choice.get('finish_reason') or result['finish']
        require(result['stream_done'] and result['finish'] == 'stop', 'Incomplete generation')
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - begin
        write(folder / (task + '.result.json'), result)
    return result


def run(label):
    plan = read(HERE / 'plan.json')
    counts = plan['counts']
    require(counts['sessions'] < 4 and label == plan['order'][counts['sessions']], 'Out of frozen run order or exhausted limit')
    if counts['sessions'] >= 2:
        first = read(HERE / 'compat-1/manifest.json')
        require(first['all_tasks_pass'], 'Rollback failed initial gate; no repeat pair')
    require(all(sha(p) == value for p, value in plan['files'].items()), 'Frozen input changed')
    require(all(sha(v['path']) == v['sha256'] for v in plan['binaries'].values()), 'Binary changed')
    m = plan['model']; st = Path(m['path']).stat()
    require((st.st_size, st.st_mtime_ns) == (m['size'], m['mtime_ns']), 'Model changed')
    idle()
    for name in ('CUDA_LAUNCH_BLOCKING', 'CUDA_FORCE_PTX_JIT', 'CUDA_FORCE_JIT', 'CUDA_INJECTION64_PATH'):
        require(os.environ.get(name) in (None, '', '0'), 'Native override present: ' + name)
    if plan['experiment_started_unix'] is None:
        plan['experiment_started_unix'] = time.time()
    deadline = plan['experiment_started_unix'] + plan['limits']['experiment_seconds']
    require(time.time() + 180 < deadline, 'Experiment time exhausted')
    folder = HERE / label
    folder.mkdir()
    profile = folder / 'device-profiles.json'
    shutil.copy2(plan['profile'], profile)
    arm = label.split('-')[0]
    binary = plan['binaries'][arm]
    env = {k.upper(): v for k, v in os.environ.items() if not k.upper().startswith('NINFER_')}
    env.update(NINFER_PREFILL_ALIGN='0', NINFER_DEVICE_PROFILES=str(profile), LOCALAPPDATA=str(folder / 'runtime-cache'))
    env['PATH'] = str(Path(binary['path']).parent) + os.pathsep + env.get('PATH', '')
    substitutions = {'<engine>': binary['path'], '<model>': m['path'], '<request-log>': str(folder / 'requests.jsonl')}
    command = [substitutions.get(x, x) for x in plan['runtime_command']]
    manifest = dict(label=label, binary=binary, command=command, started_unix=time.time(), tasks={}, all_tasks_pass=False, owned_process_stopped=False)
    write(folder / 'manifest.json', manifest)
    write(folder / 'environment.json', {k: v for k, v in env.items() if k.startswith('NINFER_') or k == 'LOCALAPPDATA'})
    counts['sessions'] += 1
    write(HERE / 'plan.json', plan)
    proc = None
    watchdog = None
    try:
        with (folder / 'stdout.log').open('xb') as out, (folder / 'stderr.log').open('xb') as err:
            proc = subprocess.Popen(command, cwd=Path(binary['path']).parent, env=env, stdout=out, stderr=err, **FLAGS)
        manifest['pid'] = proc.pid
        write(folder / 'manifest.json', manifest)
        ready_end = min(time.time() + plan['limits']['ready_seconds'], deadline)
        while time.time() < ready_end:
            require(proc.poll() is None, 'Server exited before health')
            try:
                with urllib.request.urlopen(URL + '/health', timeout=2) as response:
                    health = json.load(response)
                if health.get('status') in ('ok', 'ready') or health.get('ok'):
                    break
            except OSError:
                pass
            time.sleep(.5)
        else:
            raise TimeoutError('Server startup deadline')
        for task in TASKS:
            require(counts['requests'] < 8 and time.time() + 90 < deadline, 'Request or time budget exhausted')
            counts['requests'] += 1
            write(HERE / 'plan.json', plan)
            request_path = HERE / 'fixtures/requests' / (task + '.json')
            shutil.copy2(request_path, folder / (task + '.request.json'))
            watchdog = threading.Timer(min(plan['limits']['request_seconds'], deadline - time.time()), lambda: proc.terminate() if proc.poll() is None else None)
            watchdog.daemon = True
            watchdog.start()
            print(json.dumps({'session': label, 'request': task, 'started': True}), flush=True)
            result = send(request_path, folder, task)
            watchdog.cancel()
            content = result['content'].strip()
            match = re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```', content, re.S)
            source = folder / (task + '.py')
            source.write_text(match.group(1) if match else content, encoding='utf-8')
            fixture = HERE / 'fixtures'
            grade_run = subprocess.run([sys.executable, str(fixture / 'long-code-grade.py'), '--worker', str(source), task, str(fixture / 'long-code-fixtures' / task / 'cases.json')], capture_output=True, text=True, timeout=30, **FLAGS)
            require(grade_run.returncode == 0, grade_run.stderr)
            grade = json.loads(grade_run.stdout)
            write(folder / (task + '.grade.json'), grade)
            row = dict(passed=grade['passed'], total=grade['total'], finish=result['finish'],
                       content_sha256=hashlib.sha256(result['content'].encode()).hexdigest(),
                       reasoning_sha256=hashlib.sha256(result['reasoning'].encode()).hexdigest(),
                       elapsed_seconds=result['elapsed_seconds'], usage=result['usage'])
            manifest['tasks'][task] = row
            write(folder / 'manifest.json', manifest)
            print(json.dumps({'session': label, 'request': task, **row}), flush=True)
        manifest['all_tasks_pass'] = len(manifest['tasks']) == 2 and all(x['passed'] == x['total'] for x in manifest['tasks'].values())
    except BaseException as error:
        manifest['error'] = repr(error)
    finally:
        if watchdog:
            watchdog.cancel()
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            manifest['owned_process_stopped'] = proc.poll() is not None
        manifest['profile_unchanged'] = sha(profile) == sha(plan['profile'])
        manifest['inputs_unchanged'] = all(sha(p) == h for p, h in plan['files'].items())
        manifest['finished_unix'] = time.time()
        write(folder / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def inventory(path):
    result = {'files': 0, 'bytes': 0, 'reparse_points': []}
    pending = [path]
    while pending:
        current = pending.pop()
        st = current.lstat()
        if getattr(st, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            result['reparse_points'].append({'relative': str(current.relative_to(path)), 'target': os.readlink(current)})
        elif current.is_dir():
            pending.extend(current.iterdir())
        else:
            result['files'] += 1
            result['bytes'] += st.st_size
    return result


def cleanup():
    require((HERE / 'plan.json').exists(), 'Snapshot and fixture preparation must precede cleanup')
    idle()
    ledger = ARCHIVE / 'moves.json'
    require(not ledger.exists(), 'Cleanup already recorded; do not repeat')
    retired = ARCHIVE / 'benchmarks'
    retired.mkdir()
    moves = []
    for old in sorted((ROOT / 'benchmarks').iterdir()):
        if old == HERE:
            continue
        old = bound(old, ROOT / 'benchmarks')
        target = bound(retired / old.name, ARCHIVE)
        require(not target.exists(), 'Archive target already exists')
        before = inventory(old)
        identity = old.stat().st_ino
        moves.append({'original': str(old), 'archived': str(target), 'inventory': before, 'moved': False})
        write(ledger, moves)
        # Same-volume rename retains contents and nested junctions; never follows their targets.
        old.rename(target)
        require(target.stat().st_ino == identity and inventory(target) == before, 'Archive identity mismatch')
        moves[-1]['moved'] = True
        write(ledger, moves)
    print(json.dumps({'archived_top_level_items': len(moves), 'files': sum(x['inventory']['files'] for x in moves), 'default_packages_touched': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'cleanup'])
    parser.add_argument('--label', choices=['q5-1', 'compat-1', 'compat-2', 'q5-2'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'cleanup':
        cleanup()
    else:
        require(args.label is not None, '--label is required')
        run(args.label)
