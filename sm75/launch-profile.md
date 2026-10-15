# Coder390 Q8MTP 的 100K 启动配置

已采用的启动入口保存在仓库根目录：
[`start-fusion-sm75-coder390-100k-lan.bat`](../start-fusion-sm75-coder390-100k-lan.bat)。
该文件原样保留清理工作区前的启动参数。默认运行如下：

```bat
start-fusion-sm75-coder390-100k-lan.bat
```

| 配置 | 保留值 |
| --- | --- |
| 模型 | `C:\QWEN\Qwen3.8-27B-Coder390-EfficientThink-Q3LynnStyle-Q8MTP.ninfer` |
| 运行包 | `releases\fusion-43c39c6-sm75-q5load\ninfer-serve.exe` |
| API 地址、模型 ID | `http://10.0.0.10:3682/v1`、`qwen3.8-27b` |
| 上下文、GPU KV 容量 | 均为 `102400` tokens |
| KV 类型、主机 KV | `rk8v4`、`0 MiB` |
| Prefill | chunk `384`、`NINFER_PREFILL_ALIGN=0` |
| GDN 状态 | FP16 |
| 推测解码 | MTP，默认 `4` 个 draft tokens；ngram `0` |
| 默认输出上限、推理强度 | `94208` tokens（含推理）、`xhigh` |
| 并发、共享前缀、设备状态槽 | `1`、`0`、`0` |
| 采样 | temperature `1`、top_p `.95`、top_k `20` |
| 设备 profile、CUDA Graph allowance | `auto`、`128 MiB` |
| KVMem ring | 关闭，保持 full target attention |

BAT 支持可选的 draft 数 `2`、`3`、`4` 和 `--no-pause`，默认仍是 `4`。
客户端请求可以覆盖服务端默认采样和输出参数。换机器时需要调整 BAT 中的 `HOST` 和模型路径。
启动前会清除继承的 `NINFER_*` 实验环境变量，并设置运行包内的 `device-profiles.json`。

源码仓库不包含 EXE、DLL 或模型权重。BAT 需要上述运行包目录及其依赖 DLL；
本次保存启动配置没有构建或发布新二进制。已提交的源码修复不会自动进入旧运行包。

使用其他明确指定的引擎路径时，可以使用现有的
[`tools/sm75/serve.ps1`](../tools/sm75/serve.ps1)。以下命令只显示配置，不加载模型：

```powershell
.\tools\sm75\serve.ps1 -Engine 'D:\path\to\ninfer-serve.exe' `
  -Model 'C:\QWEN\Qwen3.8-27B-Coder390-EfficientThink-Q3LynnStyle-Q8MTP.ninfer' `
  -BindAddress '10.0.0.10' -Port 3682 -PrintCommand
```

`serve.ps1` 默认从仓库的 `sm75/device-profiles.json` 读取设备配置；其内容与采用包的 profile 一致。
`sm75/runtime-command.json` 保留的是验证工具使用的命令模板，包括独立的测试端口和请求日志占位符。
实际 LAN 启动参数以根目录 BAT 为准。
