"""Bounded Swift model check using the frozen q5 release and replay transport.

Reuses replay-original.py without changing that closed experiment or its inputs.
No compilation, instrumentation, ability scoring, retry, or package adoption.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parents[1]
HELPER = Path(__file__).with_name('replay-original.py')
HELPER_SHA = '6b4f9c0bd28ac76d9716b4cc697e9642f2442d02ada23120404a773bde09baeb'
OUT = WORKSPACE / 'benchmarks/swift-model-20261014'
MODEL = Path('C:/QWEN/Huihui-Qwen3.8-27B-abliterated-Swift-1.5-GSQ-RCO-IQ3_S-mtp.ninfer')
SOURCE = WORKSPACE / 'benchmarks/original-sm75-20261014/requests'

spec = importlib.util.spec_from_file_location('frozen_replay', HELPER)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
replay.require(replay.sha(HELPER) == HELPER_SHA, 'Frozen replay helper changed')
replay.OUT = OUT
# The q5 executable must resolve runtime dependencies from its own distribution.
replay.ROOT = replay.Q5


def inspect_model():
    sys.path.insert(0, str(REPO / 'src-tree/fusion-engine-src'))
    from tools.artifact.reader import Artifact
    from tools.artifact.schema import TensorObject
    with Artifact(MODEL) as artifact:
        return dict(name=artifact.directory.metadata.get('name'),
                    artifact_id=artifact.artifact_id.hex(),
                    components=artifact.directory.components,
                    formats=dict(Counter(o.format for o in artifact.objects if isinstance(o, TensorObject))),
                    scope='Directory/configuration only; no tensor-finiteness qualification')


def prepare():
    replay.require(not OUT.exists(), 'Experiment already exists; do not reset its budget')
    replay.idle()
    frozen = replay.read(REPO / 'sm75/current-release.json')
    replay.require(all(replay.sha(replay.Q5 / n) == h for n, h in frozen['files'].items()), 'q5 release changed')
    metadata = inspect_model()
    replay.require('mtp' in metadata['components'], 'Swift MTP component missing')
    st = MODEL.stat()
    replay.emit(hashing_model=str(MODEL), bytes=st.st_size)
    model = dict(path=str(MODEL), size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=replay.sha(MODEL))
    replay.require((MODEL.stat().st_size, MODEL.stat().st_mtime_ns) == (st.st_size, st.st_mtime_ns), 'Model changed while hashing')
    OUT.mkdir(parents=True)
    replay.write(OUT / 'artifact-summary.json', metadata)
    (OUT / 'requests').mkdir()
    inputs = {}
    for task in replay.TASKS:
        src = SOURCE / (task + '.request.json')
        dst = OUT / 'requests' / src.name
        shutil.copy2(src, dst)
        replay.require(replay.sha(src) == replay.sha(dst), 'Historical request copy differs')
        inputs[str(dst)] = replay.sha(dst)
    profile = OUT / 'device-profiles.json'
    shutil.copy2(replay.Q5 / profile.name, profile)
    preserved = [Path(__file__), HELPER, profile, OUT / 'artifact-summary.json',
                 REPO / 'sm75/runtime-command.json', REPO / 'sm75/current-release.json']
    preserved.extend(WORKSPACE.glob('start-*.bat'))
    inputs.update({str(p): replay.sha(p) for p in preserved})
    runtime_files = {str(replay.Q5 / n): {'sha256': h, 'bytes': (replay.Q5 / n).stat().st_size}
                     for n, h in frozen['files'].items() if n.endswith(('.exe', '.dll'))}
    plan = dict(created_unix=time.time(),
                source='Existing local Swift IQ3_S model with the unchanged q5 release',
                executable=str(replay.Q5 / 'ninfer-serve.exe'), model=model,
                model_metadata='artifact-summary.json', profile=str(profile),
                runtime_files=runtime_files, default_files=frozen['files'], inputs=inputs,
                runtime_command=replay.read(REPO / 'sm75/runtime-command.json'),
                original_default_model=replay.read(WORKSPACE / 'benchmarks/original-sm75-20261014/plan.json')['model'],
                tasks=replay.TASKS, order=['swift-1', 'swift-2'],
                limits={'cold_starts': 2, 'requests': 10, 'ready_seconds': 120,
                        'request_seconds': 420, 'model_experiment_seconds': 1200},
                request_policy='Byte-identical historical requests, seeds and per-request budgets; server defaults unchanged. Compare each request only with its repeat on Swift.',
                stop_policy='Stop the entire experiment on startup/runtime/transport/timeout/configuration failure. No retry, third run, compilation, instrumentation or default adoption.',
                observations={'same_seed_fork': 'Content and reasoning compared separately with actual-request agreement',
                              'illegal_token': 'Original release request errors, stderr and raw SSE',
                              'nan': 'Unobserved internally: q5 release has no tensor capture',
                              'quality': 'Not graded', 'performance': 'No claim'},
                limitations=['Swift changes model weights, quantization formats, MTP weights and applicable kernel routes together; not a weight-only intervention.',
                             'No simultaneous Coder390 control in this bounded round. Historical original-package results used a different EXE and DLLs.',
                             'Same-seed comparison is within Swift; different-model output equality is not required.',
                             'A negative result in two repeats does not prove correctness or exclude an intermittent fault.',
                             'NaN tensor finiteness and actual token-ID equality are not captured.'],
                counts={'cold_starts': 0, 'requests': 0}, started_unix=None, closed=False)
    replay.write(OUT / 'plan.json', plan)
    replay.emit(prepared=True, model_sha256=model['sha256'], limits=plan['limits'])


original_summary = replay.summarize


def summarize(plan):
    original_summary(plan)
    summary = replay.read(OUT / 'summary.json')
    summary['test_runtime_unchanged'] = summary.pop('original_runtime_unchanged')
    summary['nan_tensor_status'] = 'UNOBSERVED: uninstrumented q5 release has no tensor capture'
    for key, model in (('test_model_stat_unchanged', plan['model']),
                       ('default_model_stat_unchanged', plan['original_default_model'])):
        st = Path(model['path']).stat()
        summary[key] = (st.st_size, st.st_mtime_ns) == (model['size'], model['mtime_ns'])
    replay.write(OUT / 'summary.json', summary)


replay.summarize = summarize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'execute'])
    args = parser.parse_args()
    {'prepare': prepare, 'execute': replay.execute}[args.action]()
