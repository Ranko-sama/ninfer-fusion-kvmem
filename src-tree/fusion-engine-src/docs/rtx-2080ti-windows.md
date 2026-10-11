# RTX 2080 Ti / Windows 本机适配

2026-10-11：本机 sm_75 构建及 100K 全 GPU KV 文本/视觉验证通过。
KVMem 小池检索未通过早期记录检查，发行启动器默认使用全 GPU KV。

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
  TF32 调用使用 FP32 运算，不能直接将任意 TF32 输入缩窄为 FP16。
- 不支持的异步共享内存拷贝改为同步拷贝，保留部分拷贝补零与同步语义。
- BF16 成对转换与采样归约使用 sm_75 支持的实现。
- Prompt 注意力使用较小计算块，并修正块大小改变后的 KV 页号和页内偏移。
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

## 验证记录

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
此前引用的旧版 34.59 tok/s 来自另一项固定输出 2048 tokens 的长思考测试；新版尚未
复测该项，不能与此处的 58.59 tok/s 直接比较。以上均为单次结果，输出长度不同，
且文本 chunk 从旧版 512 改为 384，不能把全部差异归因于单一代码修改。

## 同参数性能补测（2026-10-11）

在 RTX 2080 Ti 22GB 上，新旧发行版各测 3 轮，顺序为旧、新、新、旧、旧、新；
每轮重启清空 KV，执行相同预热和完全相同的请求。统一使用 chunk 384、
`NINFER_PREFILL_ALIGN=0`、102400 全 GPU KV、RK8V4、FP16 GDN、MTP4、并发 1，
采样参数及请求随机种子也相同。各版本保留自身 CUDA 库和校准后的 auto 算子配置，
因此这是发行版整体性能比较，不能单独归因到某个内核或上游修改。

| 指标（三轮中位数） | 旧 UP 版 | 本次适配版 | 新版变化 |
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
整卡显存采样峰值均约 21595 MiB。结论是新版没有整体性能提升：短上下文生成和
长文首次响应落后，长历史生成速度接近。此前不同参数的单次结果不能取代本次对照。
完整命令、原始结果、检查和波动范围保存在工作区
`benchmarks/same-params-20261011/`，摘要为其中的 `report.txt`。

## KVMem 小池结果与适用范围

`validation/ring-32k` 使用 17920 GPU KV、8192 MiB host KV、window 16384、retrieve
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

## 使用与回退

发行目录：`D:\AI\ninfer-fusion-kvmem\releases\fusion-43c39c6-sm75`。
从工作区双击以下任一新脚本：

- `start-fusion-sm75-coder390-100k-lan.bat`：纯文本。
- `start-fusion-sm75-coder390-100k-vision-overlay-lan.bat`：文本与视觉 overlay。

地址仍为 `http://10.0.0.10:3682/v1`，model ID 为 `qwen3.8-27b`。脚本会检查现存服务
和 LAN 地址，使用随包设备路由配置；采样默认 temperature 1、top_p 0.95、top_k 20，
reasoning xhigh、MTP4。客户端可覆盖这些请求参数；精确图片转录建议请求温度 0。
包中包含 CUDA 12 运行库、FFmpeg、curl、许可证、验证清单与源码适配补丁。

旧根目录 `ninfer-serve.exe` 和 `start-coder390-q8mtp-*` 脚本保持原样。回退时关闭新版
服务后使用旧脚本即可。测试结束会停止自己的进程，交付时不会留下后台模型服务。
本次没有创建 Git 提交；修改位于 `upgrade-staging/fusion-sm75` 的独立源码副本。
