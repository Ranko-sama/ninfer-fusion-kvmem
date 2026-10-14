# RTX 2080 Ti / Windows 本机适配

当前状态（2026-10-14）以 [sm75 维护指南](../../../sm75/README.md) 为准。
本文各节是带日期的历史构建与采用记录；本次源码同步不发布新二进制，也不表示分叉或 NaN 根因已解决。

历史记录（2026-10-11）：优化版通过本机 100K 全 GPU KV 文本/视觉验证；新旧各三轮同参数对照
基本追平旧 UP，结果见“优化后同参数复测”。旧版 KVMem 小池早期记录检查未通过，
发行启动器默认使用全 GPU KV。

## 构建基线

- Fusion 源码：`1314521gjy/ninfer-fusion-kvmem`，`43c39c6c6b284760af5d78cbdc3b465b49b8b46f`。
- 硬件：RTX 2080 Ti，22,528 MiB 显存，计算能力 7.5；Ryzen 5 5600。
- 编译器：CUDA 12.8.61、Visual Studio 2022 BuildTools / MSVC 19.44.35227。
- CMake 3.31.6，Ninja，Release，单架构 `CMAKE_CUDA_ARCHITECTURES=75`。
- 依赖：vcpkg 基线 `4bca8fd8654e5ba76f92661db7bfe954768ad8ef`，`x64-windows`，FFmpeg 8.1.2、libcurl 8.21.0。

适配参考了 `noame19/ninfer-tesla-t10` 的 sm_75 移植提交
`2f8c1603fe1758dca9e620f147044fc5c1561625`。本次基于 Fusion 源码继续实现 Windows
构建、数值边界和共享内存适配；并未使用该参考仓库的引擎替代 Fusion。

## 实现范围

- Turing 原生 FP16/INT8 MMA 组合为调用方要求的矩阵形状。
- BF16 操作数能精确表示为 FP16 时使用 Tensor Core；否则使用 FP32 运算保留其指数范围。
  GDN 私有 BF16/TF32 乘法对范围内操作数使用 FP16 Tensor Core，直接累积到 FP32。
  BF16 正常范围转换精确；极小数尾部允许舍入，每个操作数附加绝对误差不超过 2^-25。
  TF32 正常范围的 10 位尾数舍入合并进 FP16 转换；相对 TF32 参考的尾部误差不超过 2^-24。
  K 项点积的转换误差上界为 Σ[ε(|a|+|b|)+ε²]，ε 分别取上述上界，再加 FP32 累积误差；
  大数及非有限数仍使用全指数范围回退。此优化没有修改通用 BF16/TF32 基元。
  双阶段 GDN 按其原有混合精度契约实现残差乘法（10 位尾数、FP32 指数范围），
  使用上述有界转换路径，超界则显式 TF32 舍入后走 FP32 回退。
  全范围回退移出展开的热循环，避免将标量 shuffle 代码复制到每个矩阵乘法位置。
  通用 TF32 实现仍保留 FP32 运算；没有将任意 TF32 输入直接缩窄为 FP16。
  此处不承诺与上一适配版的全 FP32 私有中间运算逐位一致，最终输出和状态仍须通过
  原有 FP64 数学参考与原误差阈值，不能用速度测试替代数值验证。
- 不支持的异步共享内存拷贝改为同步拷贝，保留部分拷贝补零与同步语义。
- BF16 成对转换与采样归约使用 sm_75 支持的实现。
- INT8/RK8V4 Prompt 注意力使用 64 行 query、32 列 KV、16 warps，共享内存 55,552 字节；
  保留块大小改变后的 KV 页号和页内偏移修复。校准启用已有的 FP16 P×V 路由。
  该路由的 INT4 V 解码使用 packed half2 运算；对所有有限 FP16 scale 与 16 个码值的
  1,015,808 种组合逐位验证，结果与独立解码一致，不改变缓存量化格式。
  超过 64 KiB 共享内存的 prompt/decode 路由不能被配置或校准选中。
- FP8 prompt 及 NVFP4/FP8/K8V4 decode 使用 16 列 KV 块；decode 的概率缓冲保留
  64 列存储跨度，以满足原有共享内存 swizzle 地址规则。
- Q8 sliced-K 与 grouped sliced-K 路径在 sm_75 上按 64 KiB 限制减少暂存阶段，
  必要时减少 K 方向 warp 数；编译期断言检查最终共享内存用量。
- Q8 成对投影的 256-token 配置使用两个 128-token 块；BF16 的 96 KiB 配置减少
  K 方向 warp 数或暂存阶段，包含普通线性与残差融合的新旧路径。
- GDN gating 的 cooperative kernel 通过 CUDA occupancy API 查询本机实际驻留数，
  同时检查完整块与尾块，并据此分段提交，避免沿用 sm_86 的驻留数导致启动失败。
- 修复 MSVC 的 lambda 内核指针捕获和超长字符串字面量问题；修复本机 CMake 的中文
  `/showIncludes` 前缀误解码，并关闭本项目不使用的 C++ 模块扫描。
- Windows 单架构 sm_75 构建在链接前使用 CUDA `nvprune` 去除算子库中的重复 PTX，
  保留原生 sm_75 cubin。原链接产物约 2.12 GB，Windows 加载器返回错误 193；
  精简后服务约 885 MB，已能实际启动及加载模型。

不修改模型制品格式、OpenAI/Anthropic 协议和 Fusion 的 KVMem 算法。
RTX 3090 的性能数据与路由表不能作为本显卡的性能证明。

## 本机复现入口

工作区 `D:\AI\ninfer-fusion-kvmem` 中：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File upgrade-staging/build-sm75.ps1 -Action configure
powershell -NoProfile -ExecutionPolicy Bypass -File upgrade-staging/build-sm75.ps1 -Action build -Jobs 6
powershell -NoProfile -ExecutionPolicy Bypass -File upgrade-staging/build-sm75.ps1 -Action probe
powershell -NoProfile -ExecutionPolicy Bypass -File upgrade-staging/build-sm75.ps1 -Action test
powershell -NoProfile -ExecutionPolicy Bypass -File upgrade-staging/build-sm75.ps1 -Action regression
```

构建输出在独立的 `upgrade-staging/build-sm75`，不得混入其他 CUDA 架构或旧引擎的对象文件。
测试进程需要空闲 GPU。真实模型验证脚本为 `upgrade-staging/validate-sm75.py`，使用独立
端口 3683/3684、独立设备配置缓存，并在结束后关闭自己启动的服务。
`regression` 覆盖完整注意力、Q8 线性、GDN gating，以及 BF16、Q8 add/pair/SwiGLU
的新旧两套线性路由。模型验证记录可执行文件 SHA-256，打包时要求与测试记录一致。

## 首次适配验证记录（历史基线）

已通过：sm_75 基元 GPU 测试，FP16/BF16/INT8/TF32 矩阵计算对照 CPU 双精度参考；
包含超出 FP16 指数范围的 BF16/TF32 输入。4/8/16 字节部分拷贝的每种有效长度，以及
BF16 的舍入、零、无穷和 NaN 转换边界，均为零不匹配。

已通过首轮算子测试：采样、类型转换、GDN 与 FP16 GDN 状态的单/双阶段路径、
INT8 prompt 与 GQA 打包注意力、speculative round，以及 Q4/Q5/Q6 线性计算。
完整注意力、Q8 线性、Q8 add/pair/SwiGLU 新旧路径，以及 BF16 线性与残差融合新旧
路径，均已在修正后通过回归。GDN gating 的数值及 CUDA Graph 路径也已通过；容量
测试改为检查区间内每个宽度，以覆盖 68-SM 设备上位于原调优端点之间的最大值。

日志：`upgrade-staging/ops-tests-2.log`、`sm75-bundle-test.log`、`ops-regression.log`、
`ops-regression-bf16.log`、`ops-regression-bf16-2.log`。前几份保留了修复前失败记录；
最后一份记录 BF16 四项全通过，GDN 修正后的通过记录在倒数第二份。

模型保持为 `C:\QWEN\Qwen3.8-27B-Coder390-EfficientThink-Q3LynnStyle-Q8MTP.ninfer`。
文本与视觉配置均使用 102400 GPU KV、RK8V4、MTP4、FP16 GDN、chunk 384，关闭自动
prefill 对齐，逻辑上下文 102400，host KV 为 0。完成以下端到端检查：

- 算术、强制工具调用正确；原工具调用触发的 cooperative grid 启动错误已消除。
- 18K 提示词预填充中途断开，服务确认 HTTP 499；相同请求连续重试两次均正确返回。
  文本配置第二次复用 18017 tokens，约 0.19 秒完成。
- 95K 文档先提取后部三条记录，再追问前部三条记录，文本和视觉模式均全部匹配。
- 视觉三张图在 `temperature=0` 的精确转录测试中全部字段匹配。首次按默认温度 1
  测试时，密集代码图 12 个字段中有 1 个多了结尾斜杠；保留该失败记录，不将一次
  复测通过解释为任何截图都能逐字无误。

结果目录为 `upgrade-staging/validation/{text-fullkv,vision-fullkv,vision-exact}`。
`vision-exact` 只重测图片，核对了可执行文件哈希、启动参数和环境后复用
`vision-fullkv` 已通过的非视觉检查；原始失败记录也会保存在发行清单中。

| 本机 95K 用例 | 文本 | 视觉 overlay |
| --- | ---: | ---: |
| 首 token 时间 | 224.44 s | 224.66 s |
| 生成速度（本用例） | 58.59 tok/s | 58.19 tok/s |
| 追问首 token 时间 | 0.83 s | 0.84 s |
| 追问命中缓存 | 94995 tokens | 94995 tokens |
| 全卡采样显存峰值 | 21622 MiB | 21594 MiB |

显存每 2 秒采样，包含桌面占用，不能当作逐帧精确峰值。视觉三图首 token 约
4.69–5.02 秒（温度 0）。这些速度仅代表本测试负载。相同 95K 记录提取请求，旧版
首 token 为 166.67 秒、生成 53.22 tok/s；新版为 224.44 秒、58.59 tok/s。相同追问
旧版首 token 为 0.87 秒、生成 59.11 tok/s，新版为 0.83 秒、57.96 tok/s。
此前引用的旧版 34.59 tok/s 来自另一项固定输出 2048 tokens 的长思考测试，
不能与此处的 58.59 tok/s 直接比较；后续同参数结果见下节。以上均为单次结果，输出长度不同，
且文本 chunk 从旧版 512 改为 384，不能把全部差异归因于单一代码修改。

## 优化前同参数性能补测（2026-10-11）

在 RTX 2080 Ti 22GB 上，新旧发行版各测 3 轮，顺序为旧、新、新、旧、旧、新；
每轮重启清空 KV，执行相同预热和完全相同的请求。统一使用 chunk 384、
`NINFER_PREFILL_ALIGN=0`、102400 全 GPU KV、RK8V4、FP16 GDN、MTP4、并发 1，
采样参数及请求随机种子也相同。各版本保留自身 CUDA 库和校准后的 auto 算子配置，
因此这是发行版整体性能比较，不能单独归因到某个内核或上游修改。

| 指标（三轮中位数） | 旧 UP 版 | 首次适配版 | 新版变化 |
| --- | ---: | ---: | ---: |
| 短上下文固定 2048 tokens 生成 | 46.26 tok/s | 41.80 tok/s | 慢 9.6% |
| 95K 首 token | 181.26 s | 223.12 s | 多等 23.1% |
| 95K 预填充 | 524.44 tok/s | 426.00 tok/s | 慢 18.8% |
| 缓存追问首 token | 0.70 s | 0.84 s | 多等约 0.14 s |
| 95K 历史固定 2048 tokens 生成 | 32.00 tok/s | 33.15 tok/s | 中位数快 3.6% |

长历史固定生成的旧版范围为 31.34–33.53 tok/s，新版为 31.42–33.30 tok/s，
范围重叠，不能认定新版存在稳定优势。固定输出数包含思考 tokens；这些用例在思考
阶段达到长度上限，不代表代码任务质量。记录提取回答的生成速度中位数为旧版
58.46、新版 55.35 tok/s，但输出长度不同，故不作为固定长度吞吐的替代。

六轮的参数、请求、冷缓存和固定输出长度检查全部通过，记录提取共 36/36 正确。
整卡显存采样峰值均约 21595 MiB。上述首次适配版没有整体性能提升：短上下文生成和
长文首次响应落后，长历史生成速度接近。此前不同参数的单次结果不能取代本次对照。
完整命令、原始结果、检查和波动范围保存在工作区
`benchmarks/same-params-20261011/`，摘要为其中的 `report.txt`。

## 性能退步定位与优化

在相同 32K 请求上用 Nsight Systems 对照两个发行版，原适配版的注意力 GPU 总耗时
为 11.721 秒，旧 UP 为 7.837 秒；GDN recurrence 为 3.153 秒，旧 UP 为 1.248 秒。
主要 Q3/Q4/Q5/Q6 量化矩阵内核耗时接近。记录位于
`benchmarks/perf-diagnosis/attribution.json`，原始 trace 与逐阶段统计保存在同目录。
这些带剖析开销的结果用于定位；最终速度取不带 profiler 的同参数端到端测试。

原因及处理：

- 最初为满足 64 KiB 共享内存限制，同时缩小了 query 与 KV 块，还关闭了 FP16 P×V。
  实际只缩小 KV 块即可满足限制；恢复 64 行 query、启用经数值验证的 FP16 P×V，
  并优化其 INT4 解码后，可减少反复加载 KV 的成本。
- 通用 BF16/TF32 全范围兼容代码进入 GDN 私有矩阵乘法后，极小值触发整 warp 标量
  回退；展开的标量 shuffle 代码也增大热循环。GDN 改用本节前述有界转换与原生 MMA，
  全范围回退单独保留。BF16/TF32 通用基元、公共输出和状态误差标准未降低。
- 固定输出数量仍不保证相同 MTP 工作量：数值路径改变可能使生成内容及接受率改变。
  短生成的原退步同时伴随接受率从旧版约 41.8% 降至约 35.4%；最终对照同时记录
  接受率、思考 token 数和生成吞吐，不把接受率变化冒充单个矩阵内核的速度提升。

本次没有回退 Fusion 的运行时、取消恢复或 KVMem 算法。新的数值路径不承诺与首次
适配版逐 token 一致；算子独立数学验证与模型功能回归分别验收，不能相互替代。

优化版的相关数值验证均已通过：sm_75 基元、GDN 双阶段、GDN gating、FP16 双阶段
状态；日志为 `upgrade-staging/test-perf-cold-fallback.log`。注意力及 GQA 打包路径的
完整数学验证记录在 `test-perf-shared-update.log`；此后注意力实现未改变。
新的私有基元采用明确的转换误差上界；既有通用基元测试和公开 GDN 输出/状态标准未改。

最终模型回归为 `upgrade-staging/validation/perf-final-text` 与 `perf-final-vision`：
算术、工具调用、服务确认 HTTP 499 后两次重试、95K 前后部记录全部通过；视觉模式
三张图在温度 0 下全部字段匹配。文本和视觉 overlay 的 95K 首 token 分别为
181.65、181.82 秒。这是功能回归中的单次结果，正式性能结论以六轮对照为准。
运行时和测试器在打包前核对相同二进制及设备配置哈希。

## 优化后同参数复测（2026-10-11）

重新执行旧、新、新、旧、旧、新六轮；每版三轮，每轮独立重启和相同预热。
模型、所有服务参数、请求 JSON、采样种子、冷缓存要求及固定 2048-token 输出均相同。
自动算子配置预先校准并冻结，测试中未变化；两个发行版使用各自的 CUDA 运行库。

| 指标（三轮中位数） | 旧 UP 版 | 优化版 | 优化版变化 |
| --- | ---: | ---: | ---: |
| 短上下文固定 2048 tokens 生成 | 46.12 tok/s | 46.09 tok/s | -0.1% |
| 95K 首 token | 182.45 s | 181.54 s | -0.5% |
| 95K 预填充 | 521.01 tok/s | 523.67 tok/s | +0.5% |
| 缓存追问首 token | 0.70 s | 0.69 s | -2.3% |
| 95K 历史固定 2048 tokens 生成 | 32.44 tok/s | 32.43 tok/s | 约 0% |

在本机这组负载上，原先的明确退步已消除，生成速度达到旧版水平，冷启动略快。
相较前一轮首次适配版，95K 首 token 从 223.12 秒降至 181.54 秒，短生成从
41.80 提升至 46.09 tok/s。两轮独立测试的原始结果均保留，不能据此保证所有任务都更快。

旧版长历史生成范围为 31.40–36.07 tok/s，优化版为 32.42–32.46 tok/s；旧版的 MTP
接受率也随轮次明显变化。因此不能把某一轮 36.07 对 32.42 的差异当作稳定内核退步。
短生成范围为旧版 46.08–46.15、优化版 46.07–46.11 tok/s，两者重叠。
所有固定长度生成在本轮均输出 2048 个思考 tokens，不能从这些吞吐数字推导代码质量。

九项审计全部通过：六轮完成、请求一致、服务参数一致、NInfer 环境一致、95K 冷缓存、
固定输出长度、记录提取、预期缓存命中、路由配置未变化。记录提取共 36/36 正确。
整卡显存采样峰值最高为 21594 MiB，两版高负载 SM 频率中位数均为 1935 MHz。
报告和原始数据：`benchmarks/same-params-optimized-20261011/`。
发行包的 `comparison/` 附带报告、清单、聚合统计及审计；二进制和配置哈希与验证一致。

## KVMem 小池结果与适用范围

首次适配版的 `validation/ring-32k` 使用 17920 GPU KV、8192 MiB host KV、window 16384、retrieve
8192，逻辑上下文仍为 102400。输入 31998 tokens，超过物理 GPU KV 池。日志及检查
确认 ring、检索和页面恢复都实际运行；算术、工具调用、取消重试和后部三条记录通过。
但追问早期记录只正确找回 birch，aurora 与 cedar 被回答为不存在：**仅 1/3 正确**。
采样全卡峰值 19455 MiB。该结果不足以推荐小池长期使用，不能把 HTTP 200 当作
上下文完整性证明。本次未更改 Fusion 的检索算法，也未宣称修复这个问题。

默认启动器因此保留 102400 全 GPU KV、host KV 0，即 KVMem ring 关闭。
引擎仍包含 KVMem 能力，复现实验可运行（结果目录名需为新的名称）：

```powershell
.conversion-venv/Scripts/python.exe upgrade-staging/validate-sm75.py ring-recheck --ring
```

## 使用与回退（历史perf包）

发行目录：`D:\AI\ninfer-fusion-kvmem\releases\fusion-43c39c6-sm75-perf`。
从工作区双击以下任一新脚本：

- `start-fusion-sm75-coder390-100k-lan.bat`：纯文本。
- `start-fusion-sm75-coder390-100k-vision-overlay-lan.bat`：文本与视觉 overlay。

地址仍为 `http://10.0.0.10:3682/v1`，model ID 为 `qwen3.8-27b`。脚本会检查现存服务
和 LAN 地址，使用随包设备路由配置；采样默认 temperature 1、top_p 0.95、top_k 20，
reasoning xhigh、MTP4。客户端可覆盖这些请求参数；精确图片转录建议请求温度 0。
包中包含 CUDA 12 运行库、FFmpeg、curl、许可证、验证清单与源码适配补丁。

旧根目录 `ninfer-serve.exe` 和 `start-coder390-q8mtp-*` 脚本保持原样。回退时关闭新版
服务后使用旧脚本即可。首次适配版仍保存在 `releases/fusion-43c39c6-sm75`，更新前的
Fusion 启动脚本保存在 `upgrade-staging/before-performance-launchers`；如需恢复这些脚本，
先复制回工作区根目录再运行，以保留 `%~dp0` 路径语义。
测试结束会停止自己的进程，交付时不会留下后台模型服务。
本次没有创建 Git 提交；修改位于 `upgrade-staging/fusion-sm75` 的独立源码副本。


## 2026-10-12：SM75 prompt KV 预取（历史prefetch包）

该轮本地验证包为 `fusion-43c39c6-sm75-prefetch`，上一包保留。`prompt_i8.cuh` 将 SM75
INT8-key 下一tile原始KV读取提前，原barrier后发布；没有改量化、GDN、分块或MTP默认。
chunk384/MTP4/RK8V4/FP16GDN/102400及相同模型、DLL/profile下六轮复测，95K首token
中位数182.342→178.631秒（时延下降2.04%）。生成读数的范围及轨迹另行记录。
公开Op oracle、页/tile边界、384 graph replay与95K历史检查通过；95K FP64 oracle仅
抽首尾完整query token，GPU仍执行384列，不能称为全量FP64逐元素验证。
短/约9K前缀质量均无新增失败，视觉/工具/取消/95K检索通过；长输入并非逐token一致。
该轮EXE哈希、发布记录与完整结论在仓库根`sm75/results/20261012-engine/deployment.json`和
`sm75/results/20261012-engine/decision.txt`；上文为历史记录，不赋予任意新编译版本同等验证。

<!-- t5-c0-recheck-20261013 -->
## T5 C0 有界复核：未采纳

当前默认仍为 `fusion-43c39c6-sm75-q5load`，默认参数、模型和量化格式未变。
修订并冻结预热协议后，只追加一次有界局部对照：2048 次预热、600 次测量。
95K/T5 的 append + partial + ungated reducer 耗时中位下降 **14.20%**，
控制噪声 0.216%、漂移 0.051%，满足预定门槛；这不是模型/API 提速结论。

质量门槛未通过：短题和约9K回归各保持444/447及原失败项；完整长代码修复答案出现
语法错误，141项因源码加载失败而得0/141（原q5为141/141）。下一道长代码补全在生成中
报 `target returned a token outside the public token domain`，没有完整答案可评分。
后续8项文本和3项视觉未运行，不能记作通过。尚未确定失败与C0的因果关系。

因此没有更新默认包；本轮产品和测试改动已按修改前字节恢复，保留原有本地修改、
候选补丁和全部失败证据。没有重试生成、继续性能测试或降低推理预算。
完整协议、范围、审计及后续诊断见 [最终决定](../../../sm75/results/20261013-t5-c0-recheck/decision.txt)。

<!-- q5load-final-20261013 -->
## Q5 packed-header解码优化：当前本地验证包

当前验证包为 `fusion-43c39c6-sm75-q5load`，上一包 `fusion-43c39c6-sm75-q6load` 保留用于回退。
保留先前Q3和Q6优化。默认仍为 **chunk384 / MTP4 / RK8V4 / FP16 GDN / 102400全GPU KV / 并发1**；模型、量化、DLL和profile不变。

改动范围：仅启用 SM75/cc750、Q5_K、T5、非融合、N5120、K6144或17408、dense row stride且base16对齐的packed-header读取；保留原浮点运算、归约顺序、Q3/Q6路径及所有回退。默认chunk384/MTP4/RK8V4/FP16 GDN/102400全GPU KV/并发1、采样、模型、量化、profile和DLL不变。q5load是实现包名，不是模型重新量化。

| 指标：中位数 [最小, 最大] | q6load | q5load | 中位数变化 |
| --- | ---: | ---: | ---: |
| 短上下文2048 token，API tok/s | n=3，48.5394 [47.5608, 48.6729] | n=2，48.6936 [48.6919, 48.6954] | +0.318% |
| 短上下文2048 token，内部ms/轮 | n=3，54.8144 [54.7372, 54.9619] | n=2，54.7906 [54.7893, 54.7918] | -0.043% |
| 短上下文2048 token，产出token/轮 | n=3，2.6688 [2.6043, 2.6688] | n=2，2.6688 [2.6688, 2.6688] | +0.000% |
| 95K后2048 token，API tok/s | n=2，33.9291 [32.3320, 35.5261] | n=2，36.3021 [34.0949, 38.5093] | +6.994% |
| 95K后2048 token，内部ms/轮 | n=2，68.1109 [68.0896, 68.1321] | n=2，67.9658 [67.9558, 67.9758] | -0.213% |
| 95K后2048 token，产出token/轮 | n=2，2.3115 [2.2034, 2.4196] | n=2，2.4679 [2.3182, 2.6176] | +6.766% |
| 冷95K首token，秒 | n=3，176.8621 [176.8353, 176.9598] | n=2，176.8257 [176.7072, 176.9441] | -0.021% |

原计划ABBA四次尝试，04-old基线的long-decode报错：`target returned a token outside the public token domain`。原ABBA没有完整通过。
只补一次q6load冷启动，并完整重放warmup和四个原请求以重建缓存上下文：共五次尝试、一次失败。原失败及04-old已完成的短生成、95K首问/追问均保留。
短生成及95K首问/追问基线n=3、候选n=2；长生成基线n=2、候选n=2。失败的长请求没有有效完成时延样本。
这组有限且不平衡的样本只支持所测范围，不构成统计显著性或所有任务加速的证明。内部轮时与API吞吐分别报告；每轮产出会影响吞吐。
基线失败评估：原ABBA的04-old确认为q6load基线，在long-decode触发target returned a token outside the public token domain。输入哈希前后不变，原phase/error及此前成功请求完整保留。只补一次相同q6load冷启动与原warmup/四请求序列，补测全部完成且输入未变；总计5次尝试、1次失败，不能记为四轮ABBA全通过。此错误未在唯一补测重现，根因未确定；候选两次成功与补测成功均不能证明错误修复或候选可靠性更高。采用依据是独立的Q5数值/机器码资格、局部性能收益和质量无新增失败，不是这次错误。按用户限制停止扩展调查，历史未复现波动继续跳过。
错误机理尚未确认；补测成功不证明错误已修复，也不证明候选可靠性更高。

采用解释：采用范围是经过数值资格验证的小幅解码热点优化。两个真实Q5尺寸的三个交替计时对均同向，私有内核中位时延分别下降5.8655%和3.1137%；生产SASS确认元数据窄加载合并，未改变DP4A/浮点运算计数或增加spill，Q3/Q6受测机器码保持一致。按真实调用数估算的条件收益约0.201ms/轮，不能当作整机承诺。保留全部有效观测后，短生成每轮54.814368→54.790579ms（-0.043399%），长生成68.110873→67.965823ms（-0.212962%）；唯一文本和speculative/result汇总均匹配的跨版本对05-old与02/03-new，短轮分别下降0.309469%和0.314092%，支持有限局部收益，不能证明逐轮轨迹一致或统计显著性。API短生成48.539431→48.693648tok/s（+0.317715%），长生成33.929087→36.302109tok/s（+6.994065%），但长请求每轮产出同时增加6.766376%，没有匹配文本对，因此不将约7%吞吐差异归因于本优化。负向观察全部保留：短请求TTFT中位0.281977→0.310169s（+28.192ms/+9.997968%）；95K首问API 62.292342→57.098518tok/s（-8.337821%），追问61.891095→60.442285tok/s（-2.340903%），对应每轮产出下降8.467138%/2.506251%，但每轮时延分别下降0.127093%/0.166368%。这些轨迹不匹配，不能认定API变化由Q5加载引起，也不宣称所有指标改善。接受所测首次响应代价与不确定性，以保留确定的算术等价热点改进和有限轮时收益；旧q6load保留回退，不再增加计时次数。独立短/9K答案和447代码逐项回归无新增失败，视觉、工具、取消恢复、95K事实通过。
既定五项主要信号的回退评估（其他负向观察已在采用说明和限制中逐项披露）：{}

私有Q5内核时延下降：K6144约 5.9%、K17408约 3.1%。该结果不是完整公共Op或整机收益。
新公共算子数值/路由资格：15 Q5 cases/19 replay phases; public BF16-input independent FP64 oracle, small N all rows and real N5120 eight full rows, all outputs finite/guarded. Original Q3/Q6 and all 15 formats pass. Private probe separately verifies all FP32/BF16 outputs exactly. Source and SASS independently reviewed.

短上下文：q6load答案 3/3、代码 444/447；q5load答案 3/3、代码 444/447。
- 短上下文/topological-sort：基线失败 []，候选失败 []，新增 []，改善 []。
- 短上下文/shortest-subarray：基线失败 []，候选失败 []，新增 []，改善 []。
- 短上下文/merge-patch：基线失败 [2, 49, 88]，候选失败 [2, 49, 88]，新增 []，改善 []。
约9K前缀：q6load答案 3/3、代码 444/447；q5load答案 3/3、代码 444/447。
- 约9K前缀/topological-sort：基线失败 []，候选失败 []，新增 []，改善 []。
- 约9K前缀/shortest-subarray：基线失败 []，候选失败 []，新增 []，改善 []。
- 约9K前缀/merge-patch：基线失败 [2, 49, 88]，候选失败 [2, 49, 88]，新增 []，改善 []。

视觉、工具、取消恢复与95K原始事实：arithmetic=通过；tool_call=通过；cancel_mid_prefill=通过；server_confirmed_cancel=通过；cancel_retry_1=通过；cancel_retry_2=通过；vision_dashboard=通过；vision_chinese-ui=通过；vision_code-editor-dense=通过；long_cold_prompt=通过；long_later_records=通过；long_earlier_records=通过。

q6load质量基线复用已绑定身份的原始输出；候选完整重跑，双方保存代码均独立CPU重评分。
有限回归未观察到新增失败，不证明通用能力等价或逐token一致。
按用户要求，本轮跳过历史性能波动排查；历史记录完整保留，本次优化不宣称修复那些波动。

限制：
- 性能为有限且不平衡样本：短生成及95K首问/追问old n=3/new n=2，长生成old n=2/new n=2；全部有效样本和一次失败保留，不作统计显著性或普遍加速结论。
- 短请求首次响应中位增加28.192ms；95K首问/追问API吞吐分别下降8.337821%/2.340903%。这些回退与正向结果一同披露。轮时和吞吐不同，变化的输出/接受轨迹限制归因。
- 01-old短请求采样SM频率中位1950MHz，其余多为1935MHz；每2秒采样不是锁频证明。长请求无跨版本文本匹配对。
- 私有核全FP32/BF16逐元素一致；公共Op真实N5120的独立FP64 oracle覆盖八个完整输出行，另有全输出finite/canary及小矩阵全行检查，不冒称真实大矩阵全元素FP64覆盖。
- 短/9K双方均3/3答案、444/447代码用例；merge-patch既有失败[2,49,88]不变。限定回归无观察到能力下降，不证明通用能力等价或所有输出逐token相同。
- 一次基线token域错误未在唯一补测复现，根因仍未知。本次不声称修复该错误或历史性能波动；不改变默认参数以换取速度。

EXE SHA256：`d55bd249bab9703090cc0d1f425f74489d0e5d93519b6cce0e86026880b11fc3`。

[本轮证据与结论](../../../sm75/results/20261013-q5decode/decision.txt)。
<!-- /q5load-final-20261013 -->

<!-- q6load-final-20261013 -->
## Q6 decode加载优化：2026-10-13历史包

该轮验证包为 `fusion-43c39c6-sm75-q6load`，上一包 `fusion-43c39c6-sm75-q3load` 保留用于回退。
默认仍为 **chunk 384 / MTP4 / RK8V4 / FP16 GDN / 102400 全 GPU KV / 并发1**，模型、DLL与profile不变。

仅优化 SM75 Q6_K 非融合 T5、N5120、K6144或17408的稠密四字节对齐权重读取。
保留解码位值、DP4A、浮点运算及归约顺序，原Q3路径的生产指令保持一致。

| 指标（六轮，各版三次，中位数） | q3load | q6load | 变化 |
| --- | ---: | ---: | ---: |
| 短上下文固定2048 token，API tok/s | 48.3387 | 48.5538 | +0.445% |
| 短上下文固定2048 token，内部 ms/round | 55.1934 | 54.9462 | -0.448% |
| 95K后固定2048 token，API tok/s | 33.6926 | 33.9090 | +0.642% |
| 95K后固定2048 token，内部 ms/round | 68.6044 | 68.0794 | -0.765% |

保留全部样本。API吞吐取决于轮时与MTP产出率；各自完整范围、输出匹配对见归档。
03-new/04-old的长生成文本与MTP汇总一致，双方SM采样均1935MHz，轮时下降约0.58%。
05-old长生成采样1920MHz，其匹配候选为1935MHz，更大的差值不能全归内核。
私有Q6内核约16%/17%的时延下降不是整机或完整公共Op收益。

短输入和约9K前缀，两版均为三道答案3/3、代码444/447，保留merge-patch索引[2,49,88]失败，无新增失败。
公共Op独立FP64 oracle、三图精确转录、工具、取消恢复及95K前后记录检查通过。
实际N5120的FP64 oracle检查8个完整输出行；小形状全行，并检查全输出有限值、guard及输入不变。
有限回归不证明完整能力等价或逐token一致。

历史大慢窗在相同输出/MTP计数下多出约13.68秒设备等待。该轮六轮未复现，具体触发源仍未确证。
512MiB静态显存压力未复现慢窗，2560轮固定logits/CTA顺序扰动采样无不匹配；这些结论仅覆盖测试条件。
不能将显存相关性直接认定为WDDM换页，也不能声称本次Q6优化修复了大慢窗或首次输出分叉。

EXE SHA256：`b9bf940b8d2f33182e8e873c6e464c27f33f47603bb25ccd47fa6c9d2aff428a`。
[本轮证据与结论](../../../sm75/results/20261013-decode/decision.txt)。
<!-- /q6load-final-20261013 -->

<!-- q3load-final-20261012 -->
## Q3 decode加载优化：2026-10-12历史包与验证结果

该轮验证包为 `fusion-43c39c6-sm75-q3load`，上一包 `fusion-43c39c6-sm75-prefetch` 保留用于回退。
默认仍为 **chunk 384 / MTP4 / RK8V4 / FP16 GDN / 102400 全 GPU KV / 并发1**。

仅对 SM75、Q3_K fused T5、K5120、2200字节稠密行及四字节对齐权重启用加载优化；
其余路径保留原实现。没有改模型量化、激活精度边界、DP4A计算或归约顺序。

| 指标 | prefetch | q3load | 变化 |
| --- | ---: | ---: | ---: |
| 公共SwiGLU eager（ms/Op） | 0.319368 | 0.265985 | -16.715%时延 |
| 公共SwiGLU graph（ms/Op） | 0.316876 | 0.266359 | -15.942%时延 |
| 短上下文固定2048 token生成（tok/s） | 46.370 | 48.109 | +3.749% |
| 95K后固定2048 token生成（tok/s） | 32.664 | 35.417 | +8.429% |
| 95K冷缓存首token（s） | 177.830 | 178.113 | +0.159% |

公共算子为7次/版本的CUDA event中位数（完整BF16输入→SwiGLU→BF16输出），
约16%的算子时延下降不能当作整机提升。整机是同参数六轮、每版三次的API计时中位数，
与日志累计decode时间口径分开；固定输出包括思考token，不代表任务解答质量。

整机读数范围（最小值～最大值）：
- short-decode：prefetch 46.355～46.466；q3load 47.995～48.309。
- long-decode：prefetch 31.119～33.306；q3load 27.176～38.069。
- long-first：prefetch 177.022～179.022；q3load 177.906～184.165。

短输入：prefetch 5/6题、444/447代码用例；q3load 5/6题、444/447代码用例。
短输入/prefetch保留失败：merge-patch，用例索引 [2, 49, 88]。
短输入/q3load保留失败：merge-patch，用例索引 [2, 49, 88]。
约9K前缀：prefetch 5/6题、444/447代码用例；q3load 5/6题、444/447代码用例。
约9K前缀/prefetch保留失败：merge-patch，用例索引 [2, 49, 88]。
约9K前缀/q3load保留失败：merge-patch，用例索引 [2, 49, 88]。
未观察到新增质量失败；既有失败如上保留，不能宣称所有题通过或完整能力等价。
公共独立FP64 oracle、三图精确转录、工具、取消恢复与95K前后记录检查均通过。
实际N17408的FP64 oracle检查8个完整输出行，同时检查全输出有限值、guard及输入不变；
小形状检查全部输出行，不能称实际大形状全元素FP64验证。

整轮GPU时钟/温度采样不能证明精确decode阶段时钟一致。输出与MTP接受轨迹见归档，
不能把接受率或输出变化全部归因于单个内核。
EXE SHA256：`152d02bcfa3cc5c5bb62e40a01f6cd4ee162f9d4d7228f5db5d9581b45aedb3b`。
历史决定见 [Q3 加载优化](../../../sm75/results/20261012-decode/decision.txt)；完整逐轮诊断保留在本机归档。
<!-- /q3load-final-20261012 -->
