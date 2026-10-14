param(
    [Parameter(Mandatory=$true)][string]$Engine,
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$Profile='',
    [string]$BindAddress='127.0.0.1',
    [int]$Port=3682,
    [int]$Context=102400,
    [ValidateSet(2,3,4)][int]$DraftTokens=4,
    [switch]$Vision,
    [switch]$PrintCommand
)
$ErrorActionPreference='Stop'
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$Profile) { $Profile=Join-Path $repoRoot 'sm75/device-profiles.json' }
$Engine=(Resolve-Path -LiteralPath $Engine).Path
$Model=(Resolve-Path -LiteralPath $Model).Path
$Profile=(Resolve-Path -LiteralPath $Profile).Path
$engineArgs=@($Model,'--host',$BindAddress,'--port',"$Port",'--model-id','qwen3.8-27b',
    '--max-context',"$Context",'--kv-capacity',"$Context",'--kv-dtype','rk8v4','--host-kv-mib','0',
    '--prefill-chunk','384','--device-state-slots','0','--gdn-state-fp16','--spec','mtp','--draft-tokens',"$DraftTokens",
    '--ngram-draft-tokens','0','--default-max-tokens','94208','--default-reasoning-effort','xhigh',
    '--max-concurrency','1','--max-shared-prefixes','0','--temperature','1','--top-p','.95','--top-k','20',
    '--device-profile','auto','--cuda-graph-allowance-mib','128')
if ($Vision) { $engineArgs+=@('--vision','--vision-residency','overlay','--vision-max-merged','1024') }
if ($PrintCommand) {
    [ordered]@{executable=$Engine;arguments=$engineArgs;environment=@{NINFER_DEVICE_PROFILES=$Profile;NINFER_PREFILL_ALIGN='0'}} | ConvertTo-Json -Depth 5
    exit 0
}
$savedEnvironment=@{}
Get-ChildItem Env:NINFER_* | ForEach-Object { $savedEnvironment[$_.Name]=$_.Value; Remove-Item -LiteralPath "Env:$($_.Name)" }
try {
    $env:NINFER_DEVICE_PROFILES=$Profile
    $env:NINFER_PREFILL_ALIGN='0'
    & $Engine @engineArgs
    if ($LASTEXITCODE -ne 0) { throw "Engine exited with code $LASTEXITCODE" }
} finally {
    Get-ChildItem Env:NINFER_* | ForEach-Object { Remove-Item -LiteralPath "Env:$($_.Name)" }
    foreach ($key in $savedEnvironment.Keys) { Set-Item -LiteralPath "Env:$key" -Value $savedEnvironment[$key] }
}
