"""Summarize saved same-parameter measurements, without rerunning inference."""
import json
from pathlib import Path
import statistics
import sys

sys.stdout.reconfigure(encoding='utf-8')
folder = Path(sys.argv[1]).resolve()
manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
aggregate = json.loads((folder/'aggregate.json').read_text(encoding='utf-8'))
assert manifest['complete'], 'Benchmark has not completed'
assert len(manifest['rounds']) == 6 and all(r['complete'] for r in manifest['rounds'])
commands = []
environments = []
audit = {'six_complete_rounds': True, 'identical_requests': True,
         'cold_long_prompts': True, 'fixed_2048_tokens': True,
         'all_record_checks_pass': True, 'expected_cache_hits': True,
         'route_profiles_unchanged': True}
for run in manifest['rounds']:
    run_folder = folder / f"{run['index']:02d}-{run['version']}"
    command = json.loads((run_folder/'command.json').read_text())
    command[0] = '<binary>'
    command[command.index('--request-log-jsonl')+1] = '<log>'
    commands.append(command)
    environments.append(json.loads((run_folder/'environment.json').read_text()))
    audit['route_profiles_unchanged'] &= run['profile_sha256_after']==manifest['profile_sha256'][run['version']]
    for key,result in run['results'].items():
        audit['identical_requests'] &= json.loads((folder/(key+'.request.json')).read_text(encoding='utf-8')) == json.loads((run_folder/(key+'.request.json')).read_text(encoding='utf-8'))
        if key=='long-first':
            audit['cold_long_prompts'] &= result['cached_tokens']==0 and result['prompt_tokens']==95000
        if key.endswith('decode'):
            audit['fixed_2048_tokens'] &= result['completion_tokens']==2048 and result['finish']=='length'
        if 'records_correct' in result:
            audit['all_record_checks_pass'] &= result['records_correct']==3
        if key in ('long-followup','long-decode'):
            audit['expected_cache_hits'] &= result['cached_tokens']=={'long-followup':94995,'long-decode':95073}[key]
audit['identical_runtime_arguments'] = all(c==commands[0] for c in commands)
audit['identical_ninfer_environment'] = all(e==environments[0] for e in environments)
(folder/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
assert all(audit.values()), audit
lines = ['RTX 2080 Ti 22GB 新旧 Fusion 同参数对比', '',
         '测试完整结束：' + str(manifest['complete']),
         '版本顺序：old → new → new → old → old → new；各 3 轮，每轮独立重启。',
         '共同配置：同一 Qwen3.8-27B-Coder390 Q3LynnStyle Q8MTP 模型；102400 上下文和 GPU KV；',
         'RK8V4；host KV 0；prefill chunk 384；NINFER_PREFILL_ALIGN=0；FP16 GDN；MTP4；',
         'ngram 0；并发 1；共享前缀 0；device state slots 0；CUDA graph allowance 128 MiB；',
         'temperature 1、top-p 0.95、top-k 20、xhigh；请求 seed 固定。',
         '同样 warmup 后依次运行短生成、95K 首问、固定历史追问、固定历史长生成。',
         '所有轮次同名请求 JSON 完全相同；首问要求 cached_tokens=0。',
         '启动和模型加载不计入请求时间；每版使用各自随包 CUDA 库及预校准 auto 算子选择。',
         '因此这是两个发行版的整体对比，不能分离出 CUDA、内核和上游改动各自的贡献。', '',
         '统计：中位数 [最小值, 最大值]；变化 = 新版 / 旧版 - 1。', '']
rows = [('短上下文固定 2048 tokens 生成速度 (tok/s)','short-decode','decode_tok_s'),
        ('95K 首 token (秒)','long-first','ttft_s'),
        ('95K 预填充速度 (tok/s)','long-first','prompt_tok_s'),
        ('95K 提取回答生成速度 (tok/s)','long-first','decode_tok_s'),
        ('缓存追问首 token (秒)','long-followup','ttft_s'),
        ('95K 历史固定 2048 tokens 生成速度 (tok/s)','long-decode','decode_tok_s')]
for title,case,metric in rows:
    if not all(case in aggregate[v] for v in ('old','new')):
        continue
    old=aggregate['old'][case][metric]; new=aggregate['new'][case][metric]
    show=lambda v:f"{v['median']:.2f} [{v['min']:.2f}, {v['max']:.2f}]"
    lines += [title, f"  旧版 {show(old)}；新版 {show(new)}；变化 {(new['median']/old['median']-1)*100:+.1f}%"]
lines += ['', '逐轮结果与有效性检查：']
for run in manifest['rounds']:
    lines.append(f"第 {run['index']} 轮 {run['version']}，complete={run['complete']}")
    for key,result in run['results'].items():
        canonical=json.loads((folder/(key+'.request.json')).read_text(encoding='utf-8'))
        actual=json.loads((folder/f"{run['index']:02d}-{run['version']}"/(key+'.request.json')).read_text(encoding='utf-8'))
        assert canonical==actual, 'Requests differ'
        if key=='long-first':
            assert result['cached_tokens']==0, 'Cold prompt reused cache'
        lines.append(f"  {key}: TTFT={result['ttft_s']:.3f}s; decode={result['decode_tok_s']:.3f} tok/s; prompt={result['prompt_tokens']}; cached={result['cached_tokens']}; output={result['completion_tokens']}; finish={result['finish']}")
        if 'records_correct' in result:
            lines.append(f"    记录正确 {result['records_correct']}/3")
        if 'fixed_length_pass' in result:
            lines.append(f"    2048-token 固定长度：{result['fixed_length_pass']}")
            if result.get('reasoning_tokens') is not None:
                lines.append(f"    其中思考 tokens：{result['reasoning_tokens']}")
        if result.get('draft_n'):
            lines.append(f"    MTP 接受率：{result['draft_n_accepted']/result['draft_n']*100:.2f}%")
    samples=json.loads((folder/f"{run['index']:02d}-{run['version']}"/'gpu-samples.json').read_text())
    samples=[[float(v.strip()) for v in s['values'].split(',')] for s in samples if len(s['values'].split(','))==5 and '[N/A]' not in s['values']]
    if samples:
        active=[s for s in samples if s[1]>=90]
        lines.append(f"  显存采样峰值 {max(s[0] for s in samples):.0f} MiB；最高温度 {max(s[2] for s in samples):.0f} C；高负载 SM 频率中位数 {statistics.median(s[3] for s in active):.0f} MHz")
    lines.append(f"  路由缓存测试中未变化：{run['profile_sha256_after']==manifest['profile_sha256'][run['version']]}")
lines += ['', '边界：3 轮仅用于本机实测，不代表其他任务或模型。随机种子相同不保证不同内核逐 token 输出相同。',
          '输出 token 数包含思考和正文；不同内核的生成内容与两者比例可能不同，达到长度上限不代表完成了代码任务。',
          '提取任务的输出长度可能不同，因此固定 2048-token 项更适合比较生成吞吐；MTP 接受率也会影响结果。',
          '固定历史追问使用相同的预存 assistant 内容，以保持输入完全一致。',
          'GPU 温度、频率和整卡显存每 2 秒采样，包含桌面占用，未锁频或变更电源设置。',
          '本次不测试视觉、KVMem 小池、模型代码质量；不从速度推导质量。', '',
          '二进制 SHA256：']
for version, info in manifest['binaries'].items():
    lines += [f"{version}: {info['path']}", info['sha256']]
(folder/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[:30]))
