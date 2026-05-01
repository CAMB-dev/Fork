param(
    [string]$Workspace = ".",
    [int]$MaxPacketsToRead = 250000,
    [int]$MaxPacketsToSkip = 0,
    [int]$MaxPacketsPerFlow = 16,
    [int]$MaxPerMajor = 2000,
    [string]$StartTime = "",
    [string]$EndTime = "",
    [string]$Proxy = "http://127.0.0.1:7890",
    [switch]$SkipDownload,
    [switch]$ForceBuild,
    [switch]$RunTrain
)

$ErrorActionPreference = "Stop"

Push-Location $Workspace
try {
    if ($Proxy) {
        $env:HTTP_PROXY = $Proxy
        $env:HTTPS_PROXY = $Proxy
    }
    $argsList = @(
        "run",
        "python",
        "scripts/cicids2017_friday_smoke.py",
        "--max-packets-to-read",
        "$MaxPacketsToRead",
        "--max-packets-to-skip",
        "$MaxPacketsToSkip",
        "--max-packets-per-flow",
        "$MaxPacketsPerFlow",
        "--max-per-major",
        "$MaxPerMajor"
    )
    if ($StartTime) {
        $argsList += @("--start-time", $StartTime)
    }
    if ($EndTime) {
        $argsList += @("--end-time", $EndTime)
    }
    if ($SkipDownload) {
        $argsList += "--skip-download"
    }
    if ($ForceBuild) {
        $argsList += "--force-build"
    }
    if ($RunTrain) {
        $argsList += "--run-train"
    }
    & uv @argsList
}
finally {
    Pop-Location
}
