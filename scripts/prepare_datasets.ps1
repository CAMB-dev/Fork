param(
    [string]$Workspace = ".",
    [ValidateSet("cicids2017-friday-smoke", "ustc_tfc2016", "ustc", "all")]
    [string]$Dataset = "cicids2017-friday-smoke",
    [string]$Proxy = "",
    [string]$HfEndpoint = "https://hf-mirror.com",
    [switch]$SkipDownload,
    [switch]$ForceBuild,
    [switch]$RunSmokeTrain,
    [switch]$SkipValidate,
    [switch]$SkipAudit,
    [int]$MaxPacketsToRead = 250000,
    [int]$MaxPacketsToSkip = 0,
    [int]$MaxPacketsPerFlow = 16,
    [int]$MaxPerMajor = 2000,
    [string]$LabelFileContains = "",
    [string]$StartTime = "",
    [string]$EndTime = "",
    [string]$UstcSourceRoot = "data\raw\USTC-TFC2016\extracted\USTC-TFC2016-master",
    [string]$UstcMergedPath = "data\processed\ustc_tfc2016\merged\payload_only.parquet",
    [string]$UstcLabelSplitDir = "data\processed\ustc_tfc2016\split_label_stratified",
    [string]$UstcSourceSplitDir = "data\processed\ustc_tfc2016\split_source_file"
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host ""
    Write-Host "==== $Name ====" -ForegroundColor Cyan
    & $Command
}

function Test-CommandExists {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Invoke-ValidateSplit {
    param([string]$SplitDir)
    foreach ($split in @("train", "val", "test")) {
        $path = Join-Path $SplitDir "$split.parquet"
        if (Test-Path -LiteralPath $path) {
            Invoke-Step "Validate $path" {
                uv run traffic-bert data validate --input-path $path
            }
        }
    }
}

function Invoke-AuditSplit {
    param(
        [string]$SplitDir,
        [string]$OutputPath
    )
    $train = Join-Path $SplitDir "train.parquet"
    $val = Join-Path $SplitDir "val.parquet"
    $test = Join-Path $SplitDir "test.parquet"
    if ((Test-Path -LiteralPath $train) -and (Test-Path -LiteralPath $val) -and (Test-Path -LiteralPath $test)) {
        Invoke-Step "Audit $SplitDir" {
            uv run python scripts/audit_processed_split.py `
                --train-path $train `
                --val-path $val `
                --test-path $test `
                --output-path $OutputPath
        }
    }
}

function Invoke-CicidsFridaySmoke {
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
    if ($LabelFileContains) {
        $argsList += @("--label-file-contains", $LabelFileContains)
    }
    if ($SkipDownload) {
        $argsList += "--skip-download"
    }
    if ($ForceBuild) {
        $argsList += "--force-build"
    }
    if ($RunSmokeTrain) {
        $argsList += "--run-train"
    }
    Invoke-Step "Prepare CICIDS2017 Friday smoke" {
        & uv @argsList
    }

    if (-not $SkipValidate) {
        Invoke-ValidateSplit "data\processed\cicids2017\smoke"
    }
    if (-not $SkipAudit) {
        Invoke-AuditSplit "data\processed\cicids2017\smoke" "artifacts\cicids2017_smoke\split_audit.json"
    }
}

function Invoke-UstcPipeline {
    if (-not (Test-Path -LiteralPath $UstcSourceRoot)) {
        Write-Warning "USTC source root not found: $UstcSourceRoot"
        Write-Warning "Place/extract USTC-TFC2016 under that path or pass -UstcSourceRoot. Skipping USTC."
        return
    }

    Invoke-Step "Preprocess USTC-TFC2016 PCAP files" {
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts/preprocess_ustc_tfc2016.ps1 `
            -Workspace . `
            -SourceRoot $UstcSourceRoot `
            -MaxPacketsPerFlow $MaxPacketsPerFlow
    }

    $files = Get-ChildItem -Path "data\processed\ustc_tfc2016\files" -Filter *.parquet -ErrorAction SilentlyContinue
    if (-not $files) {
        throw "No USTC processed parquet files found under data\processed\ustc_tfc2016\files"
    }

    New-Item -ItemType Directory -Force -Path (Split-Path $UstcMergedPath -Parent) | Out-Null
    $mergeArgs = @("run", "traffic-bert", "data", "merge")
    foreach ($file in $files) {
        $mergeArgs += @("--input-path", $file.FullName)
    }
    $mergeArgs += @("--output-path", $UstcMergedPath)
    Invoke-Step "Merge USTC processed files" {
        & uv @mergeArgs
    }

    Invoke-Step "Create USTC label-stratified split" {
        uv run traffic-bert data sample-stratified `
            --input-path $UstcMergedPath `
            --output-dir $UstcLabelSplitDir `
            --stratify-column major_label `
            --split-stratify-column source_label `
            --group-column flow_id `
            --max-per-class 1000000000
    }

    Invoke-Step "Create USTC source-file split" {
        uv run traffic-bert data split `
            --input-path $UstcMergedPath `
            --output-dir $UstcSourceSplitDir `
            --group-column source_file
    }

    if (-not $SkipValidate) {
        Invoke-ValidateSplit $UstcLabelSplitDir
        Invoke-ValidateSplit $UstcSourceSplitDir
    }
    if (-not $SkipAudit) {
        Invoke-AuditSplit $UstcLabelSplitDir "artifacts\ustc_tfc2016\split_label_stratified_audit.json"
        Invoke-AuditSplit $UstcSourceSplitDir "artifacts\ustc_tfc2016\split_source_file_audit.json"
    }
}

Push-Location $Workspace
try {
    if (-not (Test-CommandExists "uv")) {
        throw "uv is required but was not found in PATH"
    }
    if ($Proxy) {
        $env:HTTP_PROXY = $Proxy
        $env:HTTPS_PROXY = $Proxy
    }
    if ($HfEndpoint) {
        $env:HF_ENDPOINT = $HfEndpoint
    }

    if ($Dataset -in @("cicids2017-friday-smoke", "all")) {
        Invoke-CicidsFridaySmoke
    }
    if ($Dataset -in @("ustc_tfc2016", "ustc", "all")) {
        Invoke-UstcPipeline
    }
}
finally {
    Pop-Location
}
