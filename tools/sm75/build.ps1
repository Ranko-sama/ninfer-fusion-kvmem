param(
    [ValidateSet('check','configure','build','test')][string]$Action='check',
    [string]$BuildDir='',
    [string]$CudaRoot='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8',
    [string]$DependencyRoot=$env:VCPKG_ROOT,
    [string]$Python='python',
    [int]$Jobs=4,
    [string[]]$Targets=@('ninfer-serve','ninfer_tests','ninfer-calibrate')
)
$ErrorActionPreference='Stop'
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$BuildDir) { $BuildDir=Join-Path $repoRoot 'build-sm75' }
$BuildDir=[IO.Path]::GetFullPath($BuildDir)
$sourceRoot=Join-Path $repoRoot 'src-tree/fusion-engine-src'
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (!(Test-Path -LiteralPath $vswhere)) { throw 'Install Visual Studio Build Tools with C++ and Windows SDK.' }
$vsRoot=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vsRoot) { throw 'No Visual Studio C++ toolchain found.' }
$vcvars=Join-Path $vsRoot 'VC/Auxiliary/Build/vcvars64.bat'
$importedNames=[System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
cmd /d /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$' -and $importedNames.Add($matches[1])) { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}
if ($LASTEXITCODE -ne 0) { throw 'vcvars64 failed.' }
$cmakeBin=Join-Path $vsRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin'
$ninjaBin=Join-Path $vsRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja'
$env:CUDA_PATH=$CudaRoot
$env:PATH="$CudaRoot/bin;$cmakeBin;$ninjaBin;$BuildDir/apps;$BuildDir/tests;$env:PATH"
if (!(Test-Path -LiteralPath "$CudaRoot/bin/nvcc.exe")) { throw 'CUDA compiler missing; pass -CudaRoot.' }
if (!$DependencyRoot) { throw 'Pass -DependencyRoot with a prepared vcpkg tree (installed/x64-windows).' }
$env:VCPKG_ROOT=[IO.Path]::GetFullPath($DependencyRoot)
$env:VCPKG_TARGET_TRIPLET='x64-windows'
$dependencyTree=Join-Path $env:VCPKG_ROOT 'installed/x64-windows'
foreach ($item in @('include/libavformat/avformat.h','lib/avformat.lib','lib/avcodec.lib','lib/avutil.lib','lib/swscale.lib','share/curl/CURLConfig.cmake')) {
    if (!(Test-Path -LiteralPath (Join-Path $dependencyTree $item))) { throw "Missing dependency: $item" }
}
$pythonPath=(Get-Command $Python -ErrorAction Stop).Source
foreach ($tool in @('cmake','ninja','cl','nvcc')) { Get-Command $tool -ErrorAction Stop | Select-Object Name,Source }
Write-Output "Source: $sourceRoot"
Write-Output "Build: $BuildDir"
switch ($Action) {
    'check' { exit 0 }
    'configure' {
        # Use a NEW build directory after an upstream update; never overwrite the validated cache.
        & cmake -S $sourceRoot -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 "-DCMAKE_CUDA_COMPILER=$CudaRoot/bin/nvcc.exe" "-DCMAKE_PREFIX_PATH=$dependencyTree" "-DPython3_EXECUTABLE=$pythonPath" -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=OFF
    }
    'build' { & cmake --build $BuildDir --target $Targets -j $Jobs }
    'test' { & ctest --test-dir $BuildDir --output-on-failure -R '^ninfer_(sm75_primitives|gdn_gating_proj|gdn_state_fp16_two_stage|gated_delta_net_two_stage|softmax_attention|softmax_attention_pack_gqa)_test$' }
}
if ($LASTEXITCODE -ne 0) { throw "$Action failed with exit code $LASTEXITCODE" }
