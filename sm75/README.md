# RTX 2080 Ti 适配与维护

本分支基于 `1314521gjy/ninfer-fusion-kvmem`，由 `Ranko-sama/ninfer-fusion-kvmem`
维护 sm75 / Windows 兼容和性能优化。保留上游 LICENSE、NOTICE 和第三方署名。
原项目的最新说明仍在仓库根 README；本文件负责本 Fork 的使用和升级流程。

## 当前基线与范围

- 上游基线、历史验证 EXE 哈希、工具链及硬件见 [baseline.json](baseline.json)。
- 硬件是 RTX 2080 Ti **22528 MiB（22GB 改装卡）**、68 SM。100K 全 GPU KV 配置不能直接套用到 11GB 卡。
- 实测模型为 Qwen3.8-27B-Coder390 Q3LynnStyle / Q8MTP；仓库不包含模型、运行库或 EXE。
- 使用 RK8V4、FP16 GDN、MTP4、chunk 384、并发 1、100K 全 GPU KV；KVMem ring 关闭。
- 没有重做模型量化或更改模型架构、KVMem 检索算法；改动包含核心 CUDA 算子实现。
- 数值边界、兼容路径、历史实验和小池限制见[引擎适配记录](../src-tree/fusion-engine-src/docs/rtx-2080ti-windows.md)。该文中的本机路径属于历史记录，跨目录操作使用本指南。

2026-10-11 同参数六轮（旧、新、新、旧、旧、新）中位数：

| 指标 | 旧 UP | 本分支历史验证包 |
| --- | ---: | ---: |
| 短上下文固定 2048 token 生成 | 46.12 tok/s | 46.09 tok/s |
| 95K 冷缓存首 token | 182.45 s | 181.54 s |
| 95K 历史固定 2048 token 生成 | 32.44 tok/s | 32.43 tok/s |

两版基本持平，不能将不同任务、输出长度或 MTP 接受率的结果直接比较。
文本/视觉回归包括工具调用、取消后重试、95K 前后记录和三张精确转录图。
2026-10-12 小型质量对照为两版各 5/6 题完全通过，代码各 444/447 用例通过；
两版 6 题的正文和推理文本相同。JSON merge patch 新增嵌套 null 的 3 个失败是两版共同的失败，未隐藏。
这是短上下文、温度 0、medium 推理的小样本，不代表所有任务能力等效。
聚合读数和验收结果保存在 [results/20261012](results/20261012)。

已知限制：小 GPU KV 池检索早期记录的历史实验只有 1/3 正确，本分支未修复该问题。
新增工具并不把历史二进制的验证自动赋予后续重编译版本；每次源码更新后重新记录验证结果。

## 仓库与分支

- `origin`：自己的 Fork，用于推送自己的修改。
- `upstream`：`https://github.com/1314521gjy/ninfer-fusion-kvmem.git`，用于读取上游。
- `main`：保持上游历史，不混入本机适配。
- `sm75`：持续维护的适配分支；发布过的提交不改写历史。
- `update/<日期>`：升级候选分支；通过检查后再合回 `sm75`。
- 标签 `sm75-v0.1.0`：首次整理后的源码基线，不是一个新编译 EXE 的验证声明。

新机器克隆后：

```powershell
git clone -b sm75 https://github.com/Ranko-sama/ninfer-fusion-kvmem.git
cd ninfer-fusion-kvmem
git remote add upstream https://github.com/1314521gjy/ninfer-fusion-kvmem.git
```

## 构建与启动

准备 Visual Studio 2022 C++ Build Tools / Windows SDK、CUDA 12.8、Python 3.11、
CMake >=3.28、Ninja，以及含 FFmpeg / curl 的 vcpkg `installed/x64-windows` 依赖树。
本机历史依赖版本在引擎适配记录中；脚本检查已有依赖，不自动安装或下载。
将以下示例中的依赖、模型和引擎路径替换为自己的路径：

```powershell
./tools/sm75/build.ps1 -Action check -DependencyRoot D:/sdk -Python python
./tools/sm75/build.ps1 -Action configure -DependencyRoot D:/sdk -Python python
./tools/sm75/build.ps1 -Action build -DependencyRoot D:/sdk
./tools/sm75/build.ps1 -Action test -DependencyRoot D:/sdk
./tools/sm75/serve.ps1 -Engine ./build-sm75/apps/ninfer-serve.exe -Model D:/models/model.ninfer
```

默认构建目录为仓库内 `build-sm75`。升级候选使用新的 `-BuildDir`，不要覆盖已验证缓存。
运行 EXE 需要对应 CUDA/FFmpeg/curl DLL；历史发行包带有这些依赖，新编译文件要按构建依赖配置 PATH 或在旁边放置 DLL。
`serve.ps1` 默认仅监听 127.0.0.1；需要局域网访问时显式传 `-BindAddress`。
`-Vision` 启用视觉 overlay，`-PrintCommand` 只展示配置而不启动服务。
不同模型必须满足架构和 MTP 组件要求。这里保存的 device profile 仅是这张卡的已测基线，升级后应重新校准候选配置。

## 验证入口

所有 GPU 测试顺序运行，不要并行争抢显卡。测试使用 3683/3684 端口；端口已占用就报错，
只清理自己创建的服务。测试结果写入被 Git 忽略的 `artifacts` 或显式输出目录。
先完成构建脚本的 `test`，再做模型功能回归：

```powershell
python tools/sm75/validate.py candidate-text --engine D:/candidate/ninfer-serve.exe --model D:/models/model.ninfer --device-profile-path sm75/device-profiles.json --long
python tools/sm75/validate.py candidate-vision --engine D:/candidate/ninfer-serve.exe --model D:/models/model.ninfer --device-profile-path sm75/device-profiles.json --long --vision
```

仅需快速检查时省略 `--long`；仍验证算术、工具调用和取消重试。
`--ring` 是已知有失败历史的小池诊断，不能作为默认配置。

质量小测（约几分钟，第一次克隆可以先用 `--self-test` 检查题库，无须 GPU）：

```powershell
python tools/sm75/quality.py --self-test
python tools/sm75/quality.py --model D:/models/model.ninfer --baseline-exe D:/stable/ninfer-serve.exe --candidate-exe D:/candidate/ninfer-serve.exe --baseline-profile D:/stable/device-profiles.json --candidate-profile sm75/device-profiles.json --output artifacts/quality-candidate
```

代码在受限命名空间和有超时的子进程中执行；这不是操作系统安全沙箱，仅用于本地可信测试。
对输出截断、格式错误和判分器不支持的代码，应查看原始结果，不能只读总分。

正式性能对照使用同一模型、固定请求、独立重启、每版三轮和原来的 95K 夹具：

```powershell
python tools/sm75/performance.py perf-candidate --old-executable D:/stable/ninfer-serve.exe --new-executable D:/candidate/ninfer-serve.exe --model D:/models/model.ninfer --old-profile D:/stable/device-profiles.json --new-profile sm75/device-profiles.json
python tools/sm75/report-performance.py artifacts/perf-candidate
```

性能输出达到固定长度不等于完成代码任务；质量小测另行判分。`old` 和 `new` 是对照标签，
后续升级用当前稳定包作为 `old`，不要永远只与最早的 UP 版比较。
复现历史对照需要自行保留旧 UP EXE 和对应 profile；本 Fork 没有发布其二进制或冒充其源码。

## 跟进上游

手动检查，不自动合并或自动部署：

```powershell
python tools/sm75/check-upstream.py
git fetch upstream
git log --oneline main..upstream/main
```

有新提交时，先确保工作区干净，再建立升级候选：

```powershell
git switch main
git merge --ff-only upstream/main
git push origin main
git switch sm75
git switch -c update/2026xxxx
git merge upstream/main
```

有冲突时根据新上游语义解决，不一概选择 ours/theirs。上游已接纳的兼容修改应该删除本地重复实现。
合并后检查相关数值契约、重新构建、校准并运行上面的验证。更新 `sm75/baseline.json` 的
上游提交、候选 EXE/profile 哈希和对应结果，提交并推送候选分支供审查。
只有验证通过后才将候选合入 `sm75`、打新标签并切换日常启动器。
已公开的 `sm75` 使用合并更新，不强制推送或改写历史。

Git 仓库只存源码、脚本、固定公开夹具和精简结果。模型、大型日志、CUDA/DLL、编译目录、
本机凭据和完整私有配置不提交。需要分发运行包时使用 Releases，并附相应许可证、构建来源与验证范围。
