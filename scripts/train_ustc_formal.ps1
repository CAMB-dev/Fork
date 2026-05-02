param(
    [string]$Workspace = ".",
    [string]$TrainPath = "data/processed/ustc_tfc2016/split_label_stratified/train.parquet",
    [string]$ValPath = "data/processed/ustc_tfc2016/split_label_stratified/val.parquet",
    [string]$TestPath = "data/processed/ustc_tfc2016/split_label_stratified/test.parquet",
    [string]$OutputRoot = "artifacts/ustc_formal",
    [ValidateSet("baseline", "bert-supervised", "bert-mlm", "full")]
    [string]$Plan = "bert-supervised",
    [string]$View = "payload_only",
    [string]$Device = "auto",
    [int]$Seed = 42,
    [int]$MaxLength = 512,
    [int]$Stride = 384,
    [int]$MaxWindows = 4,
    [int]$BaselineEpochs = 3,
    [int]$BaselineBatchSize = 16,
    [int]$ClassifierEpochs = 3,
    [int]$ClassifierBatchSize = 4,
    [int]$MlmEpochs = 1,
    [int]$MlmBatchSize = 4,
    [switch]$SkipBaseline,
    [switch]$SkipClassifier,
    [switch]$RunMlm,
    [switch]$SkipEval,
    [switch]$SkipAudit,
    [switch]$Resume
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

function Require-Path {
    param([string]$PathValue)
    if (-not (Test-Path $PathValue)) {
        throw "Required path not found: $PathValue"
    }
}

Push-Location $Workspace
try {
    Require-Path $TrainPath
    Require-Path $ValPath
    Require-Path $TestPath

    $BaselineDir = Join-Path $OutputRoot "neural_baseline_cnn"
    $MlmDir = Join-Path $OutputRoot "mlm"
    $ClassifierDir = Join-Path $OutputRoot "classifier"

    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

    if (-not $SkipAudit) {
        Invoke-Step "Audit processed split" {
            uv run python scripts/audit_processed_split.py `
                --train-path $TrainPath `
                --val-path $ValPath `
                --test-path $TestPath `
                --output-path (Join-Path $OutputRoot "split_audit.json")
        }
    }

    $DoBaseline = (-not $SkipBaseline) -and (@("baseline", "full") -contains $Plan)
    $DoMlm = $RunMlm -or (@("bert-mlm", "full") -contains $Plan)
    $DoClassifier = (-not $SkipClassifier) -and (@("bert-supervised", "bert-mlm", "full") -contains $Plan)

    if ($DoBaseline) {
        $baselineCheckpoint = Join-Path $BaselineDir "neural_baseline.pt"
        if ($Resume -and (Test-Path $baselineCheckpoint)) {
            Write-Host "Skipping CNN baseline; checkpoint exists: $baselineCheckpoint" -ForegroundColor Yellow
        }
        else {
            Invoke-Step "Train CNN baseline" {
                uv run traffic-bert train neural-baseline `
                    --train-path $TrainPath `
                    --val-path $ValPath `
                    --output-dir $BaselineDir `
                    --view $View `
                    --model cnn `
                    --epochs $BaselineEpochs `
                    --batch-size $BaselineBatchSize `
                    --max-length $MaxLength `
                    --stride $Stride `
                    --max-windows $MaxWindows `
                    --device $Device `
                    --seed $Seed
            }
        }

        if (-not $SkipEval) {
            Invoke-Step "Evaluate CNN baseline" {
                uv run traffic-bert eval neural-baseline `
                    --data-path $TestPath `
                    --checkpoint $baselineCheckpoint `
                    --view $View `
                    --batch-size $BaselineBatchSize `
                    --stride $Stride `
                    --max-windows $MaxWindows `
                    --device $Device `
                    --output-dir (Join-Path $BaselineDir "test_eval")
            }
        }
    }

    $InitBertArgs = @()
    if ($DoMlm) {
        $mlmCheckpoint = Join-Path $MlmDir "mlm.pt"
        if ($Resume -and (Test-Path $mlmCheckpoint)) {
            Write-Host "Skipping MLM; checkpoint exists: $mlmCheckpoint" -ForegroundColor Yellow
        }
        else {
            Invoke-Step "Train MLM" {
                uv run traffic-bert train mlm `
                    --train-path $TrainPath `
                    --output-dir $MlmDir `
                    --view $View `
                    --epochs $MlmEpochs `
                    --batch-size $MlmBatchSize `
                    --max-length $MaxLength `
                    --stride $Stride `
                    --max-windows $MaxWindows `
                    --device $Device `
                    --seed $Seed
            }
        }
        $InitBertArgs = @("--init-bert-checkpoint", $mlmCheckpoint)
    }

    if ($DoClassifier) {
        $classifierCheckpoint = Join-Path $ClassifierDir "classifier.pt"
        if ($Resume -and (Test-Path $classifierCheckpoint)) {
            Write-Host "Skipping classifier; checkpoint exists: $classifierCheckpoint" -ForegroundColor Yellow
        }
        else {
            Invoke-Step "Train Byte-BERT classifier" {
                uv run traffic-bert train classifier `
                    --train-path $TrainPath `
                    --val-path $ValPath `
                    --output-dir $ClassifierDir `
                    --view $View `
                    --epochs $ClassifierEpochs `
                    --batch-size $ClassifierBatchSize `
                    --max-length $MaxLength `
                    --stride $Stride `
                    --max-windows $MaxWindows `
                    --device $Device `
                    --seed $Seed `
                    @InitBertArgs
            }
        }

        if (-not $SkipEval) {
            $thresholdPath = Join-Path $ClassifierDir "minor_thresholds.json"
            Invoke-Step "Calibrate classifier thresholds" {
                uv run traffic-bert eval calibrate-thresholds `
                    --data-path $ValPath `
                    --checkpoint $classifierCheckpoint `
                    --output-path $thresholdPath `
                    --view $View `
                    --batch-size $ClassifierBatchSize `
                    --max-length $MaxLength `
                    --stride $Stride `
                    --max-windows $MaxWindows `
                    --device $Device
            }

            Invoke-Step "Evaluate Byte-BERT classifier" {
                uv run traffic-bert eval classifier `
                    --data-path $TestPath `
                    --checkpoint $classifierCheckpoint `
                    --view $View `
                    --batch-size $ClassifierBatchSize `
                    --max-length $MaxLength `
                    --stride $Stride `
                    --max-windows $MaxWindows `
                    --device $Device `
                    --output-dir (Join-Path $ClassifierDir "test_eval")
            }
        }
    }

    Write-Host ""
    Write-Host "Done. Outputs written to $OutputRoot" -ForegroundColor Green
}
finally {
    Pop-Location
}
