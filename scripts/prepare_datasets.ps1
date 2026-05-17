param(
    [string]$Workspace = ".",
    [ValidateSet("cicids2017-friday-smoke", "cicids2017", "cicids2017-all", "ustc_tfc2016", "ustc", "all")]
    [string]$Dataset = "cicids2017",
    [string]$Proxy = "",
    [string]$HfEndpoint = "https://hf-mirror.com",
    [switch]$SkipDownload,
    [switch]$ForceBuild,
    [switch]$RunSmokeTrain,
    [switch]$SkipValidate,
    [switch]$SkipAudit,
    [int]$MaxWorkers = 5,
    [int]$UstcWorkers = 4,
    [int]$MaxPacketsToRead = 0,
    [int]$MaxPacketsToSkip = 0,
    [int]$MaxPacketsPerFlow = 32,
    [int]$FlowTimeoutSeconds = 120,
    [bool]$CloseOnTcpFlags = $false,
    [string]$CicidsViews = "masked_header_packet",
    [string]$UstcSourceRoot = "data\raw\USTC-TFC2016\extracted\USTC-TFC2016-master"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Workspace

if ($Proxy) {
    $env:HTTP_PROXY = $Proxy
    $env:HTTPS_PROXY = $Proxy
}
$env:HF_ENDPOINT = $HfEndpoint

if (Get-Command uv -ErrorAction SilentlyContinue) {
    $python = @("uv", "run", "python")
} elseif (Test-Path ".venv\Scripts\python.exe") {
    $python = @(".venv\Scripts\python.exe")
} elseif (Test-Path ".venv/bin/python") {
    $python = @(".venv/bin/python")
} else {
    throw "Could not find uv or project .venv Python."
}

$argsList = @(
    "scripts/prepare_data.py",
    "--dataset", $Dataset,
    "--max-workers", "$MaxWorkers",
    "--ustc-workers", "$UstcWorkers",
    "--max-packets-to-skip", "$MaxPacketsToSkip",
    "--max-packets-per-flow", "$MaxPacketsPerFlow",
    "--flow-timeout-seconds", "$FlowTimeoutSeconds",
    $(if ($CloseOnTcpFlags) { "--close-on-tcp-flags" } else { "--no-close-on-tcp-flags" }),
    "--cicids-views", $CicidsViews,
    "--ustc-source-root", $UstcSourceRoot
)

if ($MaxPacketsToRead -gt 0) {
    $argsList += @("--max-packets-to-read", "$MaxPacketsToRead")
}
if ($SkipDownload) { $argsList += "--skip-download" }
if ($ForceBuild) { $argsList += "--force" }
if ($RunSmokeTrain) { $argsList += "--run-smoke-train" }
if ($SkipValidate) { $argsList += "--skip-validate" }
if ($SkipAudit) { $argsList += "--skip-audit" }

$pythonArgs = @()
if ($python.Length -gt 1) {
    $pythonArgs += $python[1..($python.Length - 1)]
}
$pythonArgs += $argsList
& $python[0] @pythonArgs
