"""Restart-isolated chunk/MTP sweep using the fixed public 95K workload.

Only chunk and draft count vary. --self-test validates fixtures and accounting
without loading a model, querying the GPU, or writing benchmark measurements.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import threading
import time
import urllib.request

import quality as harness

ROOT = Path(__file__).resolve().parents[2]
KEYS = ('short-decode', 'long-first', 'long-followup', 'long-decode')
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
CACHE_EXPECTED = {'long-followup': 94995, 'long-decode': 95073}
WARMUP = dict(model='qwen3.8-27b', messages=[{'role': 'user', 'content':
    'What is 17 + 25? Reply with the number only.'}], max_tokens=64, seed=42,
    temperature=0, top_p=.95, top_k=20, reasoning_effort='none', stream=True,
    stream_options={'include_usage': True})


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def fingerprint(path):
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'sha256': harness.sha(path)}


def unchanged_stat(path, original):
    stat = path.stat()
    return stat.st_size == original['size'] and stat.st_mtime_ns == original['mtime_ns']


def parse_configs(raw, raw_order):
    values = json.loads(raw)
    if not isinstance(values, list) or not values:
        raise ValueError('--configs must be a nonempty JSON list')
    configs = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {'name', 'chunk', 'draft'}:
            raise ValueError('Each config must have exactly name, chunk and draft')
        name = value['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
            raise ValueError('Config name must be a short safe directory name')
        if name in configs:
            raise ValueError('Duplicate config name: ' + name)
        if type(value['chunk']) is not int or not 1 <= value['chunk'] <= 102400:
            raise ValueError('chunk must be an integer in [1, 102400]')
        if type(value['draft']) is not int or not 1 <= value['draft'] <= 15:
            raise ValueError('draft must be an integer in [1, 15] (this sweep uses MTP)')
        configs[name] = value
    names = list(configs)
    order = ([part.strip() for part in raw_order.split(',')] if raw_order else
             names + list(reversed(names)) + names)
    if not order or set(order) != set(names):
        raise ValueError('--order must use only known names and include every config')
    return configs, order


def fixtures(decode_tokens):
    if not 1 <= decode_tokens <= 4096:
        raise ValueError('--decode-tokens must be in [1, 4096] for the fixed 95K history')
    bodies = {key: read(ROOT / 'sm75/fixtures' / (key + '.request.json'))
              for key in KEYS if key != 'short-decode'}
    bodies['short-decode'] = copy.deepcopy(bodies['long-decode'])
    bodies['short-decode']['messages'] = [copy.deepcopy(bodies['long-decode']['messages'][-1])]
    for key in ('short-decode', 'long-decode'):
        bodies[key]['max_tokens'] = decode_tokens
    for body in bodies.values():
        if not (body['temperature'] == 1 and body['top_p'] == .95 and body['top_k'] == 20
                and body['reasoning_effort'] == 'xhigh' and type(body['seed']) is int
                and body['stream'] is True and body['stream_options'] == {'include_usage': True}):
            raise ValueError('Public fixture sampling contract changed')
    fixture = read(ROOT / 'sm75/fixtures/fixture-95000.json')
    if fixture['tokens'] != 95000 or len(fixture['records']) != 6:
        raise ValueError('95K public record fixture changed')
    return {key: bodies[key] for key in KEYS}, fixture['records']


def replace_arg(command, flag, value):
    if command.count(flag) != 1:
        raise ValueError('Expected one command argument: ' + flag)
    command[command.index(flag) + 1] = str(value)


def command_for(base, engine, model, config, log):
    command = list(base)
    command[0:2] = [str(engine), str(model)]
    replace_arg(command, '--prefill-chunk', config['chunk'])
    replace_arg(command, '--draft-tokens', config['draft'])
    replace_arg(command, '--request-log-jsonl', log)
    if '--seed' in command:
        replace_arg(command, '--seed', 42)
    else:
        command += ['--seed', '42']
    return command


def nonconfig_command(command):
    normalized = list(command)
    for flag in ('--prefill-chunk', '--draft-tokens', '--request-log-jsonl'):
        replace_arg(normalized, flag, '<varies>')
    return normalized


def refuse_existing_engines(own_pid=None):
    if os.name == 'nt':
        script = ("$ErrorActionPreference = 'Stop'; "
                  "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
                  "$processes = @(Get-Process -ErrorAction Stop | Select-Object ProcessName, Id); "
                  "ConvertTo-Json -InputObject $processes -Compress")
        result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive',
                                 '-Command', script], check=True, capture_output=True,
                                text=True, encoding='utf-8', creationflags=NO_WINDOW)
        processes = json.loads(result.stdout)
        if not isinstance(processes, list) or not processes or any(
                not isinstance(process, dict)
                or not isinstance(process.get('ProcessName'), str)
                or not process['ProcessName']
                or type(process.get('Id')) is not int or process['Id'] < 0
                for process in processes):
            raise RuntimeError('Invalid Windows process enumeration; refusing to run')
        rows = [(process['ProcessName'], process['Id']) for process in processes]
    else:
        result = subprocess.run(['ps', '-eo', 'pid=,comm='], check=True,
                                capture_output=True, text=True)
        rows = [(parts[1], int(parts[0])) for line in result.stdout.splitlines()
                if len(parts := line.strip().split(maxsplit=1)) == 2]
    conflicts = [{'name': name, 'pid': pid} for name, pid in rows
                 if Path(name).name.lower().startswith('ninfer') and pid != own_pid]
    if conflicts:
        raise RuntimeError('Existing NInfer process may contend for GPU; refusing to run: '
                           + json.dumps(conflicts))


def check_ports():
    for port in (3683, 3684):
        with socket.socket() as sock:
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind(('127.0.0.1', port))


def wait_ready(process):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Engine exited with status {process.returncode}; inspect stderr.log')
        try:
            with urllib.request.urlopen(harness.BASE + '/health', timeout=2) as response:
                health = json.load(response)
            if health.get('status') in ('ok', 'ready') or health.get('ok'):
                return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError('Engine readiness timeout')


def send(folder, key, body):
    overrides = {key: value for key, value in body.items()
                 if key not in ('messages', 'max_tokens', 'seed')}
    result = harness.generate(folder, key, body['messages'], body['max_tokens'],
                              seed=body['seed'], **overrides)
    if read(folder / (key + '.request.json')) != body:
        raise RuntimeError('Serialized request differs from the canonical fixture: ' + key)
    return result


def log_events(path, expected_done, wait=True):
    deadline = time.monotonic() + (10 if wait else 0)
    while True:
        events = []
        if path.exists():
            # The last line can be in flight while the server appends it.
            for line in path.read_text(encoding='utf-8').splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    if not wait:
                        raise
        done = [event for event in events if event.get('event') == 'request_done']
        if len(done) >= expected_done:
            return events, done
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Expected {expected_done} request_done events, found {len(done)}')
        time.sleep(.05)


def round_accounting(result, event):
    completion = result['usage']['completion_tokens']
    timing = event['engine_timing']['decode']
    spec = event['speculative']
    rounds = timing['rounds']
    speculative_rounds = spec['rounds']
    accepted = spec['accepted_tokens']
    positions = spec['accepted_per_position']
    # These fixtures have one initial prefill output token and no forced-control
    # units; use actually published output, not all target-licensed draft tokens.
    decode_tokens = max(completion - 1, 0)
    decode_seconds = event['timings_seconds']['decode']
    host = timing['host_exposed_seconds']
    wait = timing['device_wait_exposed_seconds']
    return {'decode_rounds': rounds, 'speculative_rounds': speculative_rounds,
            'fallback_steps': spec['fallback_steps'], 'accepted_tokens': accepted,
            'drafted_tokens': spec['drafted_tokens'], 'accepted_per_position': positions,
            'survival_per_position': [n / speculative_rounds if speculative_rounds else None
                                      for n in positions],
            'effective_decode_tokens': decode_tokens,
            'effective_tokens_per_round': decode_tokens / rounds if rounds else None,
            'licensed_tokens_per_speculative_round': 1 + accepted / speculative_rounds
                if speculative_rounds else None,
            'round_ms': 1000 * decode_seconds / rounds if rounds else None,
            'host_exposed_ms_per_round': 1000 * host / rounds if rounds else None,
            'host_exposed_percent': 100 * host / (host + wait) if host + wait else None,
            'acceptance': accepted / spec['drafted_tokens'] if spec['drafted_tokens'] else None,
            'engine_timing': event['engine_timing'], 'speculative': spec}


def measure(key, result, event, records, decode_tokens):
    usage, timings = result['usage'], result['timings']
    output = usage['completion_tokens']
    cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
    metrics = {'ttft_s': result['ttft_s'], 'wall_s': result['wall_s'],
               'decode_tok_s': timings['predicted_per_second'],
               'prompt_tok_s': timings['prompt_per_second'], 'completion_tokens': output,
               'reasoning_tokens': usage.get('completion_tokens_details', {}).get('reasoning_tokens'),
               'prompt_tokens': usage['prompt_tokens'], 'cached_tokens': cached,
               'finish': result['finish'], 'content_sha256': text_hash(result['content']),
               'reasoning_sha256': text_hash(result['reasoning']),
               'request_id': event['request']['request_id']}
    metrics.update(round_accounting(result, event))
    audit = {'request_log_output_agrees': event['result']['completion_tokens'] == output,
             'request_log_prompt_agrees': event['result']['prompt_tokens'] == usage['prompt_tokens'],
             'request_log_cache_agrees': event['result']['prefix_cache_hit_tokens'] == cached,
             'no_forced_control_units': event['engine_timing']['units']['control'] == 0,
             'accepted_position_sum_agrees': sum(metrics['accepted_per_position']) == metrics['accepted_tokens'],
             'round_count_agrees': metrics['decode_rounds'] == metrics['speculative_rounds'] + metrics['fallback_steps']}
    if key.endswith('decode'):
        audit['fixed_output_tokens'] = output == decode_tokens and result['finish'] == 'length'
    if key == 'long-first':
        audit['cold_95k_prompt'] = cached == 0 and usage['prompt_tokens'] == 95000
    if key in CACHE_EXPECTED:
        audit['expected_cache_hit'] = cached == CACHE_EXPECTED[key]
    if key in ('long-first', 'long-followup'):
        wanted = records[3:] if key == 'long-first' else records[:3]
        expected_lines = {f'{name}={value}' for name, value in wanted}
        actual_lines = {line.strip() for line in result['content'].splitlines() if line.strip()}
        metrics['records_correct'] = len(expected_lines & actual_lines)
        audit['exact_record_answer'] = actual_lines == expected_lines and result['finish'] == 'stop'
    metrics['audit'] = audit
    return metrics


def monitor_gpu(stop, samples):
    names = ('memory_mib', 'utilization_percent', 'temperature_c', 'sm_clock_mhz',
             'memory_clock_mhz', 'power_w')
    query = 'memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw'
    while not stop.is_set():
        sample = {'time_unix': time.time()}
        try:
            result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=' + query,
                                     '--format=csv,noheader,nounits'], capture_output=True,
                                    text=True, timeout=8, creationflags=NO_WINDOW, check=True)
            fields = next(csv.reader(io.StringIO(result.stdout)))
            sample['raw'] = result.stdout.strip()
            for name, value in zip(names, fields):
                try:
                    sample[name] = float(value.strip())
                except ValueError:
                    sample[name] = None
        except Exception as error:
            sample['error'] = repr(error)
        samples.append(sample)
        stop.wait(2)


def gpu_summary(samples):
    out = {'sample_count': len(samples), 'errors': sum('error' in row for row in samples)}
    for name in ('memory_mib', 'temperature_c', 'sm_clock_mhz', 'memory_clock_mhz', 'power_w'):
        values = [row[name] for row in samples if isinstance(row.get(name), (int, float))]
        if values:
            out[name] = {'min': min(values), 'median': statistics.median(values), 'max': max(values)}
    active = [row['sm_clock_mhz'] for row in samples
              if (row.get('utilization_percent') or 0) >= 90 and row.get('sm_clock_mhz') is not None]
    out['busy_sm_clock_median_mhz'] = statistics.median(active) if active else None
    return out


def save_summary(output, manifest):
    metrics = ('ttft_s', 'wall_s', 'decode_tok_s', 'prompt_tok_s', 'completion_tokens',
               'cached_tokens', 'effective_tokens_per_round', 'round_ms', 'acceptance',
               'host_exposed_percent')
    aggregate = {}
    for name in manifest['configs']:
        aggregate[name] = {}
        for key in KEYS:
            rows = [run['results'][key] for run in manifest['runs']
                    if run['name'] == name and run.get('valid') and key in run['results']]
            if not rows:
                continue
            values = {'valid_runs': len(rows)}
            for metric in metrics:
                numbers = [row[metric] for row in rows if isinstance(row.get(metric), (int, float))
                           and math.isfinite(row[metric])]
                if numbers:
                    values[metric] = {'median': statistics.median(numbers), 'min': min(numbers),
                                      'max': max(numbers)}
            aggregate[name][key] = values
    comparison = {}
    for key in KEYS:
        rows = [{'run': run['index'], 'name': run['name'],
                 'content_sha256': run['results'][key]['content_sha256'],
                 'reasoning_sha256': run['results'][key]['reasoning_sha256']}
                for run in manifest['runs'] if key in run['results']]
        comparison[key] = {'runs': rows,
            'all_final_identical': len({row['content_sha256'] for row in rows}) <= 1 if rows else None,
            'all_reasoning_identical': len({row['reasoning_sha256'] for row in rows}) <= 1 if rows else None}
    manifest['output_comparison'] = comparison
    write(output / 'manifest.json', manifest)
    write(output / 'aggregate.json', aggregate)
    lines = ['SM75 configuration comparison', f"Complete: {manifest['complete']}",
             'Order: ' + ', '.join(manifest['order']),
             'Only prefill chunk and MTP draft count may differ; fresh process per run.',
             'Median [min, max]; only runs passing all audits enter aggregates.',
             'Effective tokens/round = (actual completion tokens - one initial prefill token) / decode rounds.',
             'Round ms = logged decode seconds * 1000 / decode rounds. Licensed speculative yield may',
             'exceed published output at terminal boundaries. Output differences are reported, not quality failures.',
             'Fixed token caps measure throughput, not completion of a coding task.', '']
    for name, tasks in aggregate.items():
        lines.append(name + ': ' + json.dumps(manifest['configs'][name]))
        for key, values in tasks.items():
            def show(metric):
                item = values.get(metric)
                return (f"{item['median']:.3f} [{item['min']:.3f}, {item['max']:.3f}]"
                        if item else 'unavailable')
            lines.append(f"  {key}: n={values['valid_runs']} TTFT={show('ttft_s')} s; "
                         f"decode={show('decode_tok_s')} tok/s; "
                         f"effective={show('effective_tokens_per_round')} tok/round; "
                         f"round={show('round_ms')} ms")
    lines += ['', 'Audits and actual counts:']
    for run in manifest['runs']:
        lines.append(f"  {run['index']:02d} {run['name']}: complete={run['complete']}, "
                     f"valid={run.get('valid', False)}, audit={json.dumps(run.get('audit', {}))}")
        if run.get('error'):
            lines.append('    error=' + run['error'])
        for key, result in run['results'].items():
            lines.append(f"    {key}: completion={result['completion_tokens']}, "
                         f"decode rounds={result['decode_rounds']}, speculative rounds={result['speculative_rounds']}, "
                         f"accepted positions={result['accepted_per_position']}, audit={json.dumps(result['audit'])}")
    lines += ['', 'Output identity across all completed responses:']
    for key, value in comparison.items():
        lines.append(f"  {key}: final={value['all_final_identical']}, reasoning={value['all_reasoning_identical']}")
    (output / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def self_test(configs, order, bodies, records, base):
    assert set(order) == set(configs) and len(records) == 6
    normalized = [nonconfig_command(command_for(base, Path('engine'), Path('model'), config,
                                               Path(config['name']) / 'requests.jsonl'))
                  for config in configs.values()]
    assert all(row == normalized[0] for row in normalized)
    assert bodies['short-decode']['messages'] == [bodies['long-decode']['messages'][-1]]
    assert bodies['short-decode']['max_tokens'] == bodies['long-decode']['max_tokens']
    # Synthetic arithmetic oracle only; never serialized as benchmark output.
    result = {'usage': {'completion_tokens': 11}}
    event = {'engine_timing': {'decode': {'rounds': 4, 'host_exposed_seconds': .002,
              'device_wait_exposed_seconds': .198}}, 'timings_seconds': {'decode': .2},
              'speculative': {'rounds': 4, 'accepted_tokens': 6, 'drafted_tokens': 12,
              'fallback_steps': 0, 'accepted_per_position': [3, 2, 1]}}
    accounting = round_accounting(result, event)
    assert accounting['effective_tokens_per_round'] == 2.5
    assert accounting['licensed_tokens_per_speculative_round'] == 2.5
    assert accounting['round_ms'] == 50 and accounting['host_exposed_percent'] == 1
    assert accounting['survival_per_position'] == [.75, .5, .25]
    rejected = ('[]', '[{"name":"../escape","chunk":384,"draft":4}]',
                '[{"name":"a","chunk":true,"draft":4}]',
                '[{"name":"a","chunk":384,"draft":0}]')
    for raw in rejected:
        try:
            parse_configs(raw, None)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid config accepted')
    try:
        parse_configs(json.dumps(list(configs.values())), 'unknown')
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid order accepted')
    print('Self-test passed: public fixtures, configs/order, command isolation and accounting. '
          'No GPU work or benchmark measurements produced.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('engine', 'model', 'profile', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--configs', help='JSON list of {name, chunk, draft}')
    parser.add_argument('--order', help='Comma-separated names; default forward, reverse, forward')
    parser.add_argument('--decode-tokens', type=int, default=1024)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    default_test = '[{"name":"chunk384-k4","chunk":384,"draft":4},' \
                   '{"name":"chunk512-k4","chunk":512,"draft":4}]'
    if not args.configs and not args.self_test:
        parser.error('--configs is required')
    try:
        configs, order = parse_configs(args.configs or default_test, args.order)
        bodies, records = fixtures(args.decode_tokens)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    base = read(ROOT / 'sm75/runtime-command.json')
    if args.self_test:
        self_test(configs, order, bodies, records, base)
        return
    for name in ('engine', 'model', 'profile', 'output'):
        if getattr(args, name) is None:
            parser.error('--' + name + ' is required')
    engine, model, profile = (getattr(args, name).resolve(strict=True)
                              for name in ('engine', 'model', 'profile'))
    refuse_existing_engines()
    check_ports()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print('Hashing engine, model and profile before the sweep...', flush=True)
    inputs = {name: fingerprint(path) for name, path in
              (('engine', engine), ('model', model), ('profile', profile))}
    frozen = output / 'frozen-device-profiles.json'
    shutil.copy2(profile, frozen)
    if harness.sha(frozen) != inputs['profile']['sha256']:
        raise RuntimeError('Profile changed while freezing it')
    canonical_command = nonconfig_command(command_for(base, engine, model,
                                         next(iter(configs.values())), Path('<log>')))
    manifest = {'complete': False, 'configs': configs, 'order': order, 'inputs': inputs,
                'decode_tokens': args.decode_tokens, 'nonconfig_command': canonical_command,
                'canonical_requests_sha256': {key: digest(body) for key, body in bodies.items()},
                'warmup_sha256': digest(WARMUP), 'runs': [],
                'accounting_note': 'Effective decode output subtracts one initial prefill token; '
                    'forced-control units must be zero. Target-licensed tokens can exceed published terminal output.',
                'scope': 'Fixed public workload; throughput and retrieval checks, not a general quality benchmark.'}
    for key, body in bodies.items():
        write(output / (key + '.request.json'), body)
    write(output / 'warmup.request.json', WARMUP)
    save_summary(output, manifest)
    for index, name in enumerate(order, 1):
        refuse_existing_engines()
        check_ports()
        if not all(unchanged_stat(path, inputs[label]) for label, path in
                   (('engine', engine), ('model', model), ('profile', profile))):
            raise RuntimeError('Input file metadata changed during sweep')
        folder = output / f'{index:02d}-{name}'
        folder.mkdir()
        active = folder / 'device-profiles.json'
        shutil.copy2(frozen, active)
        command = command_for(base, engine, model, configs[name], folder / 'requests.jsonl')
        env = {key.upper(): value for key, value in os.environ.items()
               if not key.upper().startswith('NINFER_')}
        env.update(NINFER_PREFILL_ALIGN='0', NINFER_DEVICE_PROFILES=str(active),
                   LOCALAPPDATA=str(folder / 'runtime-cache'))
        write(folder / 'command.json', command)
        write(folder / 'environment.json', {key: value for key, value in env.items()
                                            if key.startswith('NINFER_') or key == 'LOCALAPPDATA'})
        run = {'name': name, 'index': index, 'complete': False, 'valid': False,
               'results': {}, 'profile_sha256_before': harness.sha(active), 'audit': {}}
        manifest['runs'].append(run)
        process = None
        stop, samples = threading.Event(), []
        monitor = threading.Thread(target=monitor_gpu, args=(stop, samples), daemon=True)
        monitor.start()
        try:
            print(f'RUN {index}/{len(order)} {name} START', flush=True)
            with (folder / 'stdout.log').open('w', encoding='utf-8') as stdout, \
                 (folder / 'stderr.log').open('w', encoding='utf-8') as stderr:
                process = subprocess.Popen(command, cwd=engine.parent, env=env, stdout=stdout,
                                           stderr=stderr, creationflags=NO_WINDOW)
            write(folder / 'pid.json', {'pid': process.pid})
            wait_ready(process)
            refuse_existing_engines(process.pid)
            send(folder, 'warmup', WARMUP)
            for position, key in enumerate(KEYS, 2):
                refuse_existing_engines(process.pid)
                print(f'RUN {index}/{len(order)} {name} {key}', flush=True)
                result = send(folder, key, bodies[key])
                events, done = log_events(folder / 'requests.jsonl', position)
                if len(done) != position:
                    raise RuntimeError('Unexpected extra request in isolated test server')
                event = done[-1]
                if event['request']['requested_output_tokens'] != bodies[key]['max_tokens'] or \
                   event['request']['sampling']['seed'] != bodies[key]['seed']:
                    raise RuntimeError('Request log seed or token budget differs from fixture')
                run['results'][key] = measure(key, result, event, records, args.decode_tokens)
                write(folder / (key + '.engine-event.json'), event)
                if harness.sha(active) != inputs['profile']['sha256']:
                    raise RuntimeError('Device profile changed during measurement; pre-calibrate separately')
                save_summary(output, manifest)
            starts = [event for event in events if event.get('event') == 'server_start']
            if len(starts) != 1:
                raise RuntimeError('Expected exactly one server_start event')
            run['server_start'] = starts[0]
            run['audit'] = {
                'identical_requests': all(read(folder / (key + '.request.json')) == bodies[key] for key in KEYS),
                'identical_warmup': read(folder / 'warmup.request.json') == WARMUP,
                'identical_nonconfig_arguments': nonconfig_command(command) == canonical_command,
                'requested_config_active': starts[0]['engine']['prefill_chunk'] == configs[name]['chunk']
                    and starts[0]['engine']['speculative_draft_window'] == configs[name]['draft'],
                'exact_request_count': len(done) == 5,
                'unique_request_ids': len({event['request']['request_id'] for event in done}) == 5,
                'all_task_checks': all(all(result['audit'].values()) for result in run['results'].values())}
            run['complete'] = True
        except BaseException as error:
            run['error'] = repr(error)
            raise
        finally:
            # Never terminate another service: only the Popen object created above.
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            stop.set()
            monitor.join(timeout=10)
            write(folder / 'gpu-samples.json', samples)
            run['gpu'] = gpu_summary(samples)
            run['profile_sha256_after'] = harness.sha(active)
            run['audit']['frozen_profile_unchanged'] = (
                run['profile_sha256_before'] == run['profile_sha256_after'] == inputs['profile']['sha256'])
            run['audit']['gpu_samples_present'] = 'memory_mib' in run['gpu'] and 'sm_clock_mhz' in run['gpu']
            run['valid'] = run['complete'] and all(run['audit'].values())
            save_summary(output, manifest)
        if not run['valid']:
            raise RuntimeError('Run failed audit; see manifest.json: ' + name)
        print(f'RUN {index}/{len(order)} {name} PASS', flush=True)
    print('Verifying input hashes after the sweep...', flush=True)
    manifest['final_input_sha256'] = {label: harness.sha(path) for label, path in
                                     (('engine', engine), ('model', model), ('profile', profile))}
    manifest['audit'] = {'all_runs_valid': all(run['valid'] for run in manifest['runs']),
        'all_planned_runs_completed': len(manifest['runs']) == len(order),
        'input_hashes_unchanged': all(manifest['final_input_sha256'][label] == value['sha256']
                                    for label, value in inputs.items()),
        'frozen_source_unchanged': harness.sha(frozen) == inputs['profile']['sha256']}
    manifest['complete'] = all(manifest['audit'].values())
    save_summary(output, manifest)
    if not manifest['complete']:
        raise RuntimeError('Final audit failed; inspect manifest.json')
    print('COMPARISON COMPLETE:', output / 'report.txt', flush=True)


if __name__ == '__main__':
    main()
