"""Six objective quality tasks and 447 hidden code cases, portable old/new runner."""
import argparse, ast, copy, hashlib, itertools, json, os, random, re, shutil, socket, subprocess, sys, time, urllib.request
from fractions import Fraction
from pathlib import Path
BASE = 'http://127.0.0.1:3683'
CREATE_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def make_tasks():
    integer = next(n for n in range(10001, 1000000) if n % 7 == 3 and n % 11 == 5 and n % 13 == 9 and n % 9 == 4)
    # P(mixed | A)=8/15, P(mixed | B)=2/5, P(A | mixed)=4/7.
    probability = (Fraction(8, 15) * Fraction(3, 4)) / (Fraction(8, 15) + Fraction(2, 5))
    edges = [('S','A',8),('S','B',4),('A','C',5),('A','D',12),('B','A',2),('B','C',11),('B','D',9),('C','D',1),('C','T',10),('D','T',6)]
    candidates = []
    def paths(node, costs):
        if node == 'T':
            candidates.append(min(sum(costs)-c+c//2 for c in costs))
        for a,b,c in edges:
            if a == node:
                paths(b, costs+[c])
    paths('S', [])
    suffix = ' Return only a JSON object with key "answer"; no explanation.'
    return [
        {'id':'probability', 'kind':'answer', 'expected':str(probability), 'prompt':
         'Two boxes are chosen with equal probability. Box A contains 4 red and 2 blue balls; box B contains 1 red and 4 blue balls. Draw three balls without replacement from the chosen box. You are told that the first two balls contain exactly one red and one blue, but not their order. What is the probability the third ball is red? Give a reduced fraction as a string.'+suffix},
        {'id':'integer-constraints', 'kind':'answer', 'expected':integer, 'prompt':
         'Find the smallest integer n strictly greater than 10000 such that n mod 7 = 3, n mod 11 = 5, n mod 13 = 9, and n mod 9 = 4.'+suffix},
        {'id':'discount-path', 'kind':'answer', 'expected':min(candidates), 'prompt':
         'In a directed weighted graph, find the minimum cost from S to T. You must use a coupon on exactly one traversed edge: its cost w becomes floor(w/2). Other edges keep their costs. Edges (from,to,cost): '+json.dumps(edges)+'. Give the minimum cost as an integer.'+suffix},
        {'id':'topological-sort', 'kind':'code', 'function':'topo', 'prompt':
         'Write Python 3 function topo(n, edges). Nodes are integers 0..n-1. Each directed edge (u,v) requires u before v. Return the lexicographically smallest topological order including isolated nodes, or None if there is a cycle. Duplicate edges must not change the result. n may be 0. Do not mutate inputs. Target O((n+m) log n). Return only Python source, no tests or explanation. You may import heapq and collections.'},
        {'id':'shortest-subarray', 'kind':'code', 'function':'shortest_subarray', 'prompt':
         'Write Python 3 function shortest_subarray(nums, k) returning the length of the shortest NONEMPTY contiguous subarray whose sum is at least integer k, or -1 if none. nums may contain positive, negative, and zero integers; k may also be zero or negative; nums may be empty. Do not mutate nums. Require O(n) time for n up to 200000. Return only Python source, no tests or explanation. You may import collections.'},
        {'id':'merge-patch', 'kind':'code', 'function':'merge_patch', 'prompt':
         'Write Python 3 function merge_patch(target, patch) for JSON values. If patch is not a dict (including list or None), return a deep copy of patch. If patch is a dict, start from a deep copy of target if target is a dict, otherwise from an empty dict. For each patch key, None removes the key if present; otherwise recursively merge that value into the old value. Neither input may be modified. The result must share no mutable dict/list objects with either input, including untouched target branches. All string keys (including "__proto__", "constructor", "prototype") are ordinary keys and must be preserved. Return only Python source, no tests or explanation. You may import copy.'},
    ]

def make_cases():
    rng = random.Random(830712)
    cases = {k:[] for k in ('topological-sort','shortest-subarray','merge-patch')}
    for n, edges in [(0,[]),(3,[(0,1),(0,1)]),(2,[(0,1),(1,0)]),(1,[(0,0)])] + [
            (n, [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randrange(15))])
            for n in [rng.randrange(1,8) for _ in range(80)]]:
        expected = next((list(p) for p in itertools.permutations(range(n)) if all(p.index(u)<p.index(v) for u,v in edges)), None)
        cases['topological-sort'].append({'args':[n,edges], 'expected':expected})
    for nums,k in [([],0),([0],0),([-1],-2),([-5],-2),([2,-1,2],3),([84,-37,32,40,95],167)] + [
            ([rng.randrange(-10,11) for _ in range(rng.randrange(30))], rng.randrange(-15,40)) for _ in range(200)]:
        choices = [j-i for i in range(len(nums)) for j in range(i+1,len(nums)+1) if sum(nums[i:j])>=k]
        cases['shortest-subarray'].append({'args':[nums,k], 'expected':min(choices, default=-1)})
    cases['shortest-subarray'].append({'args':[[-1]*99999+[100000],100000], 'expected':1})
    cases['shortest-subarray'].append({'args':[[1]*100000,100000], 'expected':100000})
    def ref(target, patch):
        if not isinstance(patch, dict): return copy.deepcopy(patch)
        original = target if isinstance(target,dict) else {}
        keys = original.keys() | patch.keys()
        return {k:ref(original.get(k),patch[k]) if k in patch else copy.deepcopy(original[k])
                for k in keys if k not in patch or patch[k] is not None}
    def value(depth=0):
        if depth>=3: return rng.choice([None,False,True,0,3,'abc'])
        t = rng.randrange(4)
        if t==0: return {k:value(depth+1) for k in rng.sample(['a','b','c','__proto__','constructor','prototype'],rng.randrange(5))}
        if t==1: return [value(depth+1) for _ in range(rng.randrange(4))]
        return value(3)
    pairs = [({'keep':[{'a':1}],'drop':2},{'drop':None}), ({'a':[1]},{}), (None,{'a':{'b':None}}), ({'a':1},[{'x':2}]), ({'__proto__':{'x':1}}, {'constructor':4})]
    pairs += [(value(),value()) for _ in range(150)]
    for target,patch in pairs:
        cases['merge-patch'].append({'args':[target,patch], 'expected':ref(target,patch)})
    return cases

def strip_fence(text):
    text = text.strip()
    match = re.fullmatch(r'```(?:python|json)?\s*\n(.*?)\n```', text, re.S)
    return match.group(1) if match else text

def worker(source, task_id, cases_path):
    # Restricted namespaces plus syntax checks; subprocess has an external time limit.
    tree = ast.parse(source.read_text(encoding='utf-8'))
    allowed_modules = {'collections','heapq','copy'}
    banned = {'eval','exec','open','compile','globals','locals','vars','getattr','setattr','delattr','input','breakpoint','__import__'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(x.name not in allowed_modules for x in node.names): raise ValueError('unsupported import')
        if isinstance(node, ast.ImportFrom) and (node.module not in allowed_modules or node.level): raise ValueError('unsupported import')
        if isinstance(node, ast.Name) and (node.id in banned or node.id.startswith('__')): raise ValueError('unsupported name')
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'): raise ValueError('unsupported private attribute')
    import builtins
    safe_names = 'range len min max sum enumerate zip sorted reversed list dict set tuple int float str bool isinstance abs all any next iter None True False ValueError TypeError Exception'.split()
    safe = {name:getattr(builtins,name) for name in safe_names if hasattr(builtins,name)}
    def limited_import(name, *args, **kwargs):
        if name not in allowed_modules: raise ImportError(name)
        return __import__(name, *args, **kwargs)
    safe['__import__'] = limited_import
    namespace = {'__builtins__':safe, '__name__':'candidate'}
    exec(compile(tree, str(source), 'exec'),namespace)
    names = {'topological-sort':'topo','shortest-subarray':'shortest_subarray','merge-patch':'merge_patch'}
    fn = namespace[names[task_id]]
    def mutable_ids(v):
        ids = {id(v)} if isinstance(v,(dict,list)) else set()
        for child in (v.values() if isinstance(v,dict) else v if isinstance(v,(list,tuple)) else []): ids |= mutable_ids(child)
        return ids
    results = []
    for index,case in enumerate(json.loads(cases_path.read_text(encoding='utf-8'))[task_id]):
        args = copy.deepcopy(case['args']); before = copy.deepcopy(args)
        try:
            result = fn(*args)
            correct = result == case['expected']
            unchanged = args == before
            independent = task_id != 'merge-patch' or not (mutable_ids(result) & mutable_ids(args))
            results.append({'index':index,'pass':correct and unchanged and independent,'correct':correct,'unchanged':unchanged,'independent':independent})
        except Exception as e: results.append({'index':index,'pass':False,'error':repr(e)})
    print(json.dumps({'passed':sum(x['pass'] for x in results),'total':len(results),'failures':[x for x in results if not x['pass']]}))

def generate(folder, key, messages, limit=2048, seed=42, **overrides):
    body=dict(model='qwen3.8-27b',messages=messages,temperature=1,top_p=.95,top_k=20,
              reasoning_effort='xhigh',max_tokens=limit,seed=seed,stream=True,stream_options={'include_usage':True})
    body.update(overrides); write(folder/(key+'.request.json'),body)
    started=time.perf_counter(); first=None; content=''; reasoning=''; usage={}; timings={}; finish=None
    req=urllib.request.Request(BASE+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=2400) as response:
        for raw in response:
            line=raw.decode('utf-8').strip()
            if not line.startswith('data:'): continue
            line=line[5:].strip()
            if line=='[DONE]': break
            event=json.loads(line)
            if event.get('error'): raise RuntimeError(event['error'])
            usage=event.get('usage') or usage; timings=event.get('timings') or timings
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{})
                if delta.get('content') or delta.get('reasoning_content'):
                    if first is None: first=time.perf_counter()-started
                content+=delta.get('content') or ''; reasoning+=delta.get('reasoning_content') or ''
                finish=choice.get('finish_reason') or finish
    result=dict(key=key,wall_s=time.perf_counter()-started,ttft_s=first,usage=usage,timings=timings,finish=finish,content=content,reasoning=reasoning)
    write(folder/(key+'.result.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('content','reasoning')},ensure_ascii=False),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=3)
    parser.add_argument('--self-test', action='store_true', help='Check fixture generation without a GPU')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--baseline-exe', type=Path)
    parser.add_argument('--candidate-exe', type=Path)
    parser.add_argument('--baseline-profile', type=Path)
    parser.add_argument('--candidate-profile', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--context', type=int, default=102400)
    args = parser.parse_args()
    if args.worker:
        worker(Path(args.worker[0]), args.worker[1], Path(args.worker[2])); return
    tasks, cases = make_tasks(), make_cases()
    if args.self_test:
        assert [t['expected'] for t in tasks[:3]] == ['3/7', 16285, 14]
        assert {k:len(v) for k,v in cases.items()} == {'topological-sort':84,'shortest-subarray':208,'merge-patch':155}
        assert cases['merge-patch'][2]['expected'] == {'a':{}}
        print('Fixture checks passed: 6 tasks, 447 code cases.'); return
    for name in ('model','baseline_exe','candidate_exe','baseline_profile','candidate_profile','output'):
        if getattr(args,name) is None: parser.error('--'+name.replace('_','-')+' is required')
    model = args.model.resolve(strict=True)
    binaries = {'old':args.baseline_exe.resolve(strict=True),'new':args.candidate_exe.resolve(strict=True)}
    sources = {'old':args.baseline_profile.resolve(strict=True),'new':args.candidate_profile.resolve(strict=True)}
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    write(output/'tasks.json',tasks); write(output/'code-cases.json',cases)
    profiles = {}
    for v,p in sources.items():
        profiles[v] = output/(v+'-device-profiles.json'); shutil.copy2(p,profiles[v])
    manifest = {'complete':False,'scope':'6 tasks, one greedy completion per task/version; not general intelligence equivalence.',
                'model':str(model),'context':args.context,
                'parameters':{'temperature':0,'top_p':.95,'top_k':20,'seed':42,'reasoning_effort':'medium','max_tokens':6144},
                'tasks_sha256':sha(output/'tasks.json'),'cases_sha256':sha(output/'code-cases.json'),
                'binaries':{v:{'path':str(p),'sha256':sha(p)} for v,p in binaries.items()},
                'profile_sha256':{v:sha(p) for v,p in profiles.items()},'runs':[]}
    write(output/'manifest.json',manifest)
    for version in ('old','new'):
        folder=output/version; folder.mkdir()
        for port in (3683,3684):
            with socket.socket() as s: s.bind(('127.0.0.1',port))
        active=output/'active-device-profiles.json'; shutil.copy2(profiles[version],active)
        env={k.upper():v for k,v in os.environ.items() if not k.upper().startswith('NINFER_')}
        env.update(NINFER_PREFILL_ALIGN='0',NINFER_DEVICE_PROFILES=str(active),LOCALAPPDATA=str(output/'runtime-cache'))
        command=[str(binaries[version]),str(model),'--host','127.0.0.1','--port','3683','--stats-port','3684',
                 '--model-id','qwen3.8-27b','--max-context',str(args.context),'--kv-capacity',str(args.context),
                 '--kv-dtype','rk8v4','--host-kv-mib','0','--prefill-chunk','384','--device-state-slots','0',
                 '--gdn-state-fp16','--spec','mtp','--draft-tokens','4','--ngram-draft-tokens','0',
                 '--default-max-tokens','94208','--default-reasoning-effort','xhigh','--max-concurrency','1',
                 '--max-shared-prefixes','0','--temperature','1','--top-p','.95','--top-k','20',
                 '--device-profile','auto','--cuda-graph-allowance-mib','128','--seed','42',
                 '--request-log-jsonl',str(folder/'requests.jsonl')]
        write(folder/'command.json',command)
        run={'version':version,'complete':False,'results':[]}; manifest['runs'].append(run)
        process=None
        try:
            with (folder/'stdout.log').open('w') as stdout,(folder/'stderr.log').open('w') as stderr:
                process=subprocess.Popen(command,cwd=binaries[version].parent,env=env,stdout=stdout,stderr=stderr,creationflags=CREATE_NO_WINDOW)
            deadline=time.monotonic()+600
            while time.monotonic()<deadline:
                if process.poll() is not None: raise RuntimeError(f'engine exited {process.returncode}; see {folder}')
                try:
                    with urllib.request.urlopen(BASE+'/health',timeout=2) as response: health=json.load(response)
                    if health.get('status') in ('ok','ready') or health.get('ok'): break
                except OSError: pass
                time.sleep(1)
            else: raise TimeoutError('ready timeout')
            generate(folder,'warmup',[{'role':'user','content':'What is 17 + 25? Reply with the number only.'}],64,temperature=0,reasoning_effort='none')
            for task in tasks:
                parameters=dict(manifest['parameters']); limit=parameters.pop('max_tokens')
                result=generate(folder,task['id'],[{'role':'user','content':task['prompt']}],limit,**parameters)
                grade={'id':task['id'],'finish':result['finish'],'tokens':result['usage'].get('completion_tokens'),'wall_s':result['wall_s']}
                content=strip_fence(result['content'])
                if result['finish']!='stop': grade.update(passed=False,error='Incomplete answer / token cap')
                elif task['kind']=='answer':
                    try:
                        answer=json.loads(content)['answer']; grade.update(answer=answer,expected=task['expected'],passed=answer==task['expected'])
                    except Exception as e: grade.update(passed=False,error=repr(e))
                else:
                    source=folder/(task['id']+'.py'); source.write_text(content,encoding='utf-8')
                    try:
                        check=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(source),task['id'],str(output/'code-cases.json')],capture_output=True,text=True,encoding='utf-8',timeout=20,creationflags=CREATE_NO_WINDOW)
                        if check.returncode: grade.update(passed=False,grader_error=check.stderr)
                        else:
                            tests=json.loads(check.stdout); grade.update(passed=tests['passed']==tests['total'],tests=tests)
                    except subprocess.TimeoutExpired: grade.update(passed=False,error='Code exceeded 20 second test limit')
                run['results'].append(grade); write(output/'manifest.json',manifest)
                print(version,task['id'],json.dumps(grade),flush=True)
            run['profile_sha256_after']=sha(active)
            if sha(active)!=manifest['profile_sha256'][version]: raise RuntimeError('Device profile changed during test')
            run['complete']=True
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired: process.kill();process.wait(timeout=10)
            write(output/'manifest.json',manifest)
    same_requests=True; comparison={}
    for t in tasks:
        key=t['id']
        def read(v,suffix): return json.loads((output/v/(key+suffix)).read_text(encoding='utf-8'))
        same_requests &= read('old','.request.json')==read('new','.request.json')
        a,b=read('old','.result.json'),read('new','.result.json')
        comparison[key]={'final_content_identical':a['content']==b['content'],'reasoning_text_identical':a['reasoning']==b['reasoning']}
    manifest.update(complete=True,same_requests=same_requests,output_comparison=comparison)
    write(output/'manifest.json',manifest)
    print('Completed. Compare task scores and failures in',output/'manifest.json')


def request(path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    with urllib.request.urlopen(urllib.request.Request(BASE+path, body, {'Content-Type':'application/json'}), timeout=1800) as r:
        return json.load(r)


if __name__ == '__main__':
    main()
