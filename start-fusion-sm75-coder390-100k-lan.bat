@echo off
setlocal EnableExtensions
set "DRAFT_TOKENS=4"
set "TASK_NO_PAUSE="
if not "%3"=="" goto usage
if "%~1"=="" (
    if not "%2"=="" goto usage
    goto arguments_ready
)
if /i "%~1"=="--no-pause" (
    if not "%2"=="" goto usage
    set "TASK_NO_PAUSE=1"
    goto arguments_ready
)
if not "%~1"=="2" if not "%~1"=="3" if not "%~1"=="4" goto usage
set "DRAFT_TOKENS=%~1"
if "%~2"=="" goto arguments_ready
if /i not "%~2"=="--no-pause" goto usage
set "TASK_NO_PAUSE=1"
:arguments_ready
cd /d "%~dp0"
rem Local Fusion 43c39c6 sm75 build; settings reproduced from validation.
rem Coder390 Q8MTP: author 100K context, xhigh reasoning and sampling defaults.
rem Bind only the main LAN address. Update HOST if this computer's LAN IP changes.
set "ENGINE=%~dp0releases\fusion-43c39c6-sm75-q5load\ninfer-serve.exe"
set "MODEL=C:\QWEN\Qwen3.8-27B-Coder390-EfficientThink-Q3LynnStyle-Q8MTP.ninfer"
set "HOST=10.0.0.10"
set "PORT=3682"
rem Isolate this tested profile from inherited experimental NInfer environment flags.
for /f "tokens=1 delims==" %%V in ('set NINFER_ 2^>nul') do set "%%V="
set "NINFER_DEVICE_PROFILES=%~dp0releases\fusion-43c39c6-sm75-q5load\device-profiles.json"
set "NINFER_PREFILL_ALIGN=0"
if not exist "%ENGINE%" goto missing_engine
if not exist "%MODEL%" goto missing_model
powershell.exe -NoProfile -Command "if ((Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue) -or (Get-NetTCPConnection -State Listen -LocalPort %PORT% -ErrorAction SilentlyContinue)) { exit 1 }"
if errorlevel 1 goto already_running
powershell.exe -NoProfile -Command "if (-not (Get-NetIPAddress -AddressFamily IPv4 -IPAddress '%HOST%' -ErrorAction SilentlyContinue | Where-Object AddressState -eq Preferred)) { exit 1 }"
if errorlevel 1 goto missing_address
echo Model: "%MODEL%"
echo OpenAI API base: http://%HOST%:%PORT%/v1
echo Model ID: qwen3.8-27b. API key: not required.
echo Total context: 102400 tokens. GPU KV: 102400 tokens, RK8V4. RAM KV: 0 MiB.
echo Prefill chunk: 384 tokens, automatic alignment disabled. GDN state: FP16.
echo Reasoning effort: xhigh. Sampling: temperature 1.0, top_p 0.95, top_k 20.
echo MTP: %DRAFT_TOKENS% draft tokens. KVMem: OFF. Full target attention throughout the context.
echo Default output limit: 94208 tokens including reasoning, bounded by remaining context.
echo Client request settings override these server defaults.
echo Press Ctrl+C in this window to stop.
"%ENGINE%" "%MODEL%" --model-id qwen3.8-27b --max-context 102400 --kv-capacity 102400 --kv-dtype rk8v4 --host-kv-mib 0 --prefill-chunk 384 --device-state-slots 0 --gdn-state-fp16 --spec mtp --draft-tokens %DRAFT_TOKENS% --ngram-draft-tokens 0 --default-max-tokens 94208 --default-reasoning-effort xhigh --max-concurrency 1 --max-shared-prefixes 0 --temperature 1 --top-p .95 --top-k 20 --device-profile auto --cuda-graph-allowance-mib 128 --host "%HOST%" --port %PORT%
set "TASK_EXIT=%ERRORLEVEL%"
echo Engine exited with code %TASK_EXIT%.
goto finish
:already_running
echo [ERROR] An inference service is already running or port %PORT% is occupied. No second model was loaded.
set "TASK_EXIT=4"
goto finish
:missing_address
echo [ERROR] LAN address %HOST% is unavailable. Update HOST in this BAT to this computer's LAN IPv4 address.
set "TASK_EXIT=5"
goto finish
:missing_engine
echo [ERROR] Missing engine: "%ENGINE%"
set "TASK_EXIT=2"
goto finish
:missing_model
echo [ERROR] Missing model: "%MODEL%"
set "TASK_EXIT=3"
:finish
if not defined TASK_NO_PAUSE pause
exit /b %TASK_EXIT%
:usage
echo [ERROR] Expected an optional draft count of 2, 3 or 4 and optional --no-pause.
echo Usage: "%~nx0" [2^|3^|4] [--no-pause]
exit /b 1
