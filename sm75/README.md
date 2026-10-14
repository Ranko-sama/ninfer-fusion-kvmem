# RTX 2080 Ti 适配与维护

## 当前状态（2026-10-14）

本分支同步当前本地采用的 `fusion-43c39c6-sm75-q5load` 优化源码、测试、验证工具和精简历史记录。
本次仅提交源码与记录，不构建或发布新二进制，不切换默认运行包，不重启模型实验。
后续 C0 候选未采纳，其插桩与内核改动不在默认产品源码中。根因未解决，新的运行包采纳仍受阻断。

默认模型仍为 Coder390 Q8MTP。运行环境为 RTX 2080 Ti sm75 / 68 SM / 22528 MiB、
Windows、CUDA 12.8.61、Python 3.11；100K 配置不是 11GB 原装卡的通用配方。
固定配置为 RK8V4、FP16 GDN、MTP4、chunk384、102400 上下文/KV、并发1、无 KVMem ring，
temperature1、top_p .95、top_k20、xhigh、94208 默认输出上限。
完整参数见 [runtime-command.json](runtime-command.json)，本地默认包身份见
[current-release.json](current-release.json)，[baseline.json](baseline.json) 保留较早的历史基线。

`current-release.json` 的 adoption/quality gate 是当时有限验证的采用记录，不代表后来发现的问题已解决。
q5load 是实现包名，不是将整个模型重新量化为 Q5。新编译产物必须获得自己的身份与验收，
不能将历史 EXE 的哈希或通过记录当作任意新构建的证明。

## 已采用优化与证据范围

在原 SM75 兼容和 GDN/attention 优化基础上，本次源码包含：

- SM75 INT8-key prompt 下一块 KV 预取；见 [历史决定](results/20261012-engine/decision.txt)。
- Q3_K fused T5、K5120 的对齐读取；见 [历史决定](results/20261012-decode/decision.txt)。
- Q6_K 非融合 T5、N5120、K6144/17408 的对齐读取；见 [历史决定](results/20261013-decode/decision.txt)。
- 同形状、16 字节对齐的 Q5_K packed-header 读取；见 [采用记录](results/20261013-q5decode/decision.txt)。

保留原浮点运算、归约顺序和不满足条件时的原路径。Q5 公共数值资格及源码绑定见
[numerical-qualification.json](results/20261013-q5decode/numerical-qualification.json) 和
[candidate-frozen.json](results/20261013-q5decode/candidate-frozen.json)。这些是既有证据，本次没有重跑 CUDA 测试。

Q5 热点内核耗时降低约 3.1%–5.9%，不能等同于整机收益。原有限样本中短/长生成每轮耗时
仅改善约 0.04% / 0.21%；短请求首 token 中位延迟增加约28ms，95K首问/追问吞吐下降约
8.34% / 2.34%。样本有限且输出轨迹不一致，不能认定这些 API 变化均由优化造成。
短/9K前缀双方均为444/447，三个既有失败相同，见 [质量对照](results/20261013-q5decode/quality-audit-final.txt)。
这不证明通用能力等价、逐 token 一致或所有场景最快。

## 已知问题必须按版本区分

| 版本或实验 | 同种子可见文本分叉 | 非法 token / NaN 的直接证据 |
| --- | --- | --- |
| 当前 q5load 原发布包 | 已观察到，Swift 短输入也出现 | 已核对记录中未确认非法 token；没有直接 NaN 捕获，不能据此排除 |
| 较早 q6load | 本文不借其他版本结果代替验证 | 非法 token 已记录；当次未捕获内部 NaN |
| 后续 C0 候选及主机诊断版 | 与旧输出不同不足以单独归因 | 非法 token 已出现，诊断版直接捕获 target hidden/logits 全 NaN；未采纳 |
| I 盘 RC3 适配版 | 本文不代其做重复性结论 | 非法 token 多次出现，其中一次直接记录 logits 全 NaN；不是当前 q5load |
| 第三方原始 SM75 包 | 两次冷启动在95K请求中出现 | 该次10请求未见非法 token；内部 NaN 未观测 |

已核对历史日志至少有7次独立非法 token 事件，其中2次直接捕获 NaN；两类次数不能相加，
也不能称为 q5load 发生了7次故障。最早已找到的非法 token 日志为2026-10-10 04:12，
来自旧根目录程序的 FP32 GDN / MTP3 配置。原始故障行、来源路径与哈希见
[故障证据索引](results/20261014-sync/evidence-index.json)。旧版本同样出错不证明所有版本有同一根因，
也不能将每次非法 token 都推定为 NaN。

C0 的 [验收失败记录](results/20261013-t5-c0-recheck/decision.txt) 与
[NaN 直接捕获](results/20261013-token-domain-diagnostic/decision.txt) 单独保存。
本分支没有通过 token 替换、clamp、重试或降低默认精度/上下文预算掩盖问题。

## 已关闭的回退与模型检查

- [回退消融](results/20261014-rollback/decision.txt)：同模型、请求、种子、profile与默认参数，
  原q5修复题141/141，早期兼容包124/141，因此拒绝替换。q5补全题1200秒截止，
  候选补全题主动停止，两者均未评分；超时或主动停止不能当作 NaN/非法 token。
- [第三方原包检查](results/20261014-original/decision.txt)：原EXE与CUDA13 DLL经保留归档核对，
  2次冷启动、10请求、约10分30秒。95K冷首问推理从第214字符不同，最终正文一致；
  未做能力评分，也没有证明纯净上游或跨请求状态是原因。
- [Swift模型检查](results/20261014-swift/decision.txt)：原q5load加载另一份Swift模型，
  2次冷启动、10/10请求完成，约9分29秒。99-token短请求及三类95K请求出现分叉；
  Coder390特有权重包不是该现象的必要条件。没有同步Coder390对照，不能称模型权重单因素实验。

上述实验均已关闭，无自动重试、继续遍历版本或模型的计划。保持 q5load 和默认参数不变。
源码回退候选 `842ea42` 仅在本机独立导出；测试使用历史 EXE，未证明导出目录可逐位重建该 EXE。

## 源码、工具与本地归档

`origin` 为 Ranko-sama/ninfer-fusion-kvmem，`upstream` 为 1314521gjy/ninfer-fusion-kvmem。
`main` 保持上游历史，`sm75` 保存本机适配。本机源码在 `upgrade-staging/fusion-sm75`，
现有构建树为 `upgrade-staging/build-sm75`；不要删除或重新配置已有构建树。

| 工具 | 用途 |
| --- | --- |
| [build.ps1](../tools/sm75/build.ps1) | 检查依赖、显式构建和单元测试 |
| [serve.ps1](../tools/sm75/serve.ps1) | 展示或启动显式指定的引擎，默认配置不变 |
| [validate.py](../tools/sm75/validate.py) | 文本、视觉、工具、取消恢复与长上下文回归 |
| [quality.py](../tools/sm75/quality.py) | 固定能力题、逐例评分及无需GPU的 `--self-test` |
| [compare-configs.py](../tools/sm75/compare-configs.py) | 显式配置对照及无需GPU的 `--self-test`；不是当前运行计划 |
| [performance.py](../tools/sm75/performance.py) / [report-performance.py](../tools/sm75/report-performance.py) | 正确性门槛通过后使用的性能工具 |
| [check-upstream.py](../tools/sm75/check-upstream.py) | 只读检查上游，不自动合并或发布 |

[rollback.py](../tools/sm75/rollback.py)、[replay-original.py](../tools/sm75/replay-original.py) 和
[replay-swift.py](../tools/sm75/replay-swift.py) 是已关闭实验的原脚本，不是可直接在任意克隆中重跑的通用入口。
它们依赖本机 `upgrade-staging/fusion-sm75` 目录层级、外置发布包、模型、历史输入与冻结清单；
保留原脚本便于审计，不要为重新运行而删除已有计划或重置计数。所有GPU实验串行，只终止自有进程。

Git内保存精简结论与关键故障摘录；模型、EXE/DLL、完整SSE、构建缓存和约9.34GiB历史归档保留本机。
历史记录中的绝对路径、相对证据目录及旧“未提交”表述均描述当时状态，不是当前仓库的下载路径或新运行指令。
已归档103个旧实验条目、18组报告，保留恢复映射和嵌套junction目标，没有删除失败证据。
当前源码同步不代表根因修复、技术债清偿或一次新的能力验收。
