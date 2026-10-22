# SM75 冻结组合源码版本

这三个版本归档同一台 RTX 2080 Ti 22 GiB / SM75 上已有的冻结实现及历史验证结果。此次发布只做源码来源与提交内容静态核对：**没有重新构建，没有数值/模型运行，没有追加真实请求。**

| 版本 | 固定标签 | 版本分支 |
|---|---|---|
| 发布前原版 | [sm75-original-20261015](https://github.com/Ranko-sama/ninfer-fusion-kvmem/tree/sm75-original-20261015) | 原 `sm75` 保留在 `349ad64` |
| 六项 | [sm75-six-20261022](https://github.com/Ranko-sama/ninfer-fusion-kvmem/tree/sm75-six-20261022) | `codex/sm75-six-20261022` |
| 十四项 | [sm75-fourteen-20261022](https://github.com/Ranko-sama/ninfer-fusion-kvmem/tree/sm75-fourteen-20261022) | `codex/sm75-fourteen-20261022` |
| 十七项 | [sm75-seventeen-20261022](https://github.com/Ranko-sama/ninfer-fusion-kvmem/tree/sm75-seventeen-20261022) | `codex/sm75-seventeen-20261022` |

**当前检出版本的结论见 [VALIDATION.md](VALIDATION.md)，源码来源见 [manifest.json](manifest.json)。** 提交按原版 → 六项 → 十四项 → 十七项排列；每个标签都能单独检出完整源码。原 `main`、`sm75` 和 `sm75-v0.1.0` 不移动。

“六项／十四项／十七项”沿用讨论中的累计候选计数；更早的 Q4_K half-loop 等公共基线改动也包含在完整源码内，不重复计数。这些数字不是相对原 GitHub 版本的全部差异数量。

## 指标与适用范围

- 主要真实负载：Coder390 Q2LynnStyle/Q3LynnStyle Q8MTP，31,137-token 冷源码问答，cache hit=0，seed42，temperature1、top_p0.95、top_k20、xhigh，单次输出预算4096（包含 reasoning）。
- 引擎条件：102400上下文与KV容量、rk8v4、prefill chunk384、MTP4、GDN FP16、单并发。日常默认94208输出上限与历史测试请求覆盖值4096是不同层级，没有在此次发布中修改。
- 性能表中的百分比均指**对应对照中的耗时减少**。不同批次、不同基线的收益不能相加或连乘；没有建立原 GitHub 版本到十七项的完整同期对照。
- 这些数据不等于缓存命中率提升、decode TPS收益或完整 Claude Code/OpenCode 工具任务收益。
- 有限数值检查与有限回答评价不是全域浮点等价或一般模型能力无回归证明。采用决定与旧验收裁决分别记录，负例和异常样本不删除。

## 使用与回退

这里提供冻结生产源码快照，不包含所有历史临时数值夹具或完整构建工作树。生产差异集中于五个文件，其他受版本控制的内容继承原GitHub基线；本地其他未提交改动不混入。没有重新验证干净检出的构建。

这里提供源码版本，不含模型权重、EXE、DLL或完整运行包；检出源码不会改变已运行的服务。当前机器的默认BAT、模型参数、device profile、已部署十四项程序及其旧程序回退均未变动。

需要构建时使用仓库现有 `tools/sm75/build.ps1`，明确 `CMAKE_CUDA_ARCHITECTURES=75` 并检查本机依赖。不要把通用CMake默认架构当成SM75，也不要将此次归档解释为在干净检出上重新构建验证过。

回退时在干净的独立检出中选择上表所需标签，再按既有流程构建/部署；不要在有未提交改动的工作区强制重置。原版回退标签指向发布前的 `349ad64`，不是初始 `sm75-v0.1.0` 标签。

## 证据阅读

`evidence/` 收录选择后的历史报告原文，包含当时的KEEP、撤回和UNRESOLVED结论。这些报告记录各自阶段，不会因后来的采用决定而重写。旧报告中的“未提交”“未保留”等描述指报告当时；本次版本状态以同目录 `VALIDATION.md` 为准。

完整原始回答、请求JSONL、GPU采样、数值导出、构建日志保留于本地对应证据目录；此仓库归档未上传全部原始材料，不应声称公开材料包含可完整重演的评测包。此前排查过的私有问题记录未读取、未纳入此次提交。

