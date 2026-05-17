param(
    [string]$Workspace = ".",
    [string]$TrainPath = "data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/train.parquet",
    [string]$ValPath = "data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/val.parquet",
    [string]$TestPath = "data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/test.parquet",
    [string]$OutputRoot = "artifacts/cicids2017_masked_header_submode_group_formal_cap32_notcpclose_conn_b128_w4_sqrt_weighted",
    [string]$CicidsRawDir = "data/raw/CICIDS2017",
    [string]$CicidsProcessedDir = "data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose",
    [string]$CicidsMergedPath = "data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_all.parquet",
    [string]$CicidsAttackCoveragePath = "artifacts/cicids2017_coverage_masked_header_cap32_notcpclose/attack_coverage.json",
    [string]$CicidsRequiredSplitParent = "data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose",
    [string]$CicidsSplitAuditPath = "artifacts/cicids2017_all_masked_header_cap32_notcpclose/submode_stratified_group_split_audit.json",
    [string]$CicidsRequiredView = "masked_header_packet",
    [bool]$CicidsRequiredKeepEmptyPayload = $true,
    [double]$CicidsLabelMaxTimeDeltaSeconds = 900,
    [ValidateSet("baseline", "bert-supervised", "bert-mlm", "full")]
    [string]$Plan = "bert-supervised",
    [string]$View = "masked_header_packet",
    [string]$Device = "auto",
    [int]$Seed = 42,
    [int]$MaxLength = 512,
    [int]$Stride = 384,
    [int]$MaxWindows = 2,
    [int]$BaselineEpochs = 3,
    [int]$BaselineBatchSize = 32,
    [int]$ClassifierEpochs = 3,
    [int]$ClassifierBatchSize = 128,
    [int]$ClassifierNumWorkers = 4,
    [double]$ClassifierLearningRate = 0.00003,
    [ValidateSet("none", "balanced", "sqrt_balanced")]
    [string]$MajorClassWeighting = "sqrt_balanced",
    [double]$MajorClassWeightCap = 20,
    [switch]$UseConnectionTokens = $true,
    [switch]$UseContextTokens,
    [switch]$UseContextFeatures,
    [string]$SemanticCodebookPath = "",
    [int]$MlmEpochs = 1,
    [int]$MlmBatchSize = 16,
    [switch]$SkipBaseline,
    [switch]$SkipClassifier,
    [switch]$RunMlm,
    [switch]$SkipEval,
    [switch]$SkipAudit,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$ConnectionTokenArgs = @()
if ($UseConnectionTokens) {
    $ConnectionTokenArgs = @("--use-connection-tokens")
}
$ContextTokenArgs = @()
if ($UseContextTokens) {
    $ContextTokenArgs = @("--use-context-tokens")
}
$ContextFeatureArgs = @()
if ($UseContextFeatures) {
    $ContextFeatureArgs = @("--use-context-features")
}
$SemanticCodebookArgs = @()
if ($SemanticCodebookPath) {
    $SemanticCodebookArgs = @("--semantic-codebook-path", $SemanticCodebookPath)
}

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

function Invoke-CicidsFormalGate {
    param([string]$AuditPath = "")
    $gateArgs = @(
        "run",
        "python",
        "scripts/verify_formal_dataset.py",
        "--dataset",
        "cicids2017",
        "--train-path",
        $TrainPath,
        "--val-path",
        $ValPath,
        "--test-path",
        $TestPath,
        "--raw-dir",
        $CicidsRawDir,
        "--processed-dir",
        $CicidsProcessedDir,
        "--merged-path",
        $CicidsMergedPath,
        "--attack-coverage-path",
        $CicidsAttackCoveragePath,
        "--required-split-parent",
        $CicidsRequiredSplitParent,
        "--required-view",
        $CicidsRequiredView,
        "--required-keep-empty-payload",
        "$CicidsRequiredKeepEmptyPayload",
        "--required-window-scope",
        "pcap",
        "--required-csv-time-offset-hours",
        "3.0",
        "--required-cic-label-max-time-delta-seconds",
        "$CicidsLabelMaxTimeDeltaSeconds",
        "--output-path",
        (Join-Path $OutputRoot "formal_dataset_gate.json")
    )
    if ($AuditPath) {
        $gateArgs += @("--audit-path", $AuditPath)
    }
    else {
        $gateArgs += "--allow-missing-audit"
    }
    & uv @gateArgs
}

Push-Location $Workspace
try {
    Require-Path $TrainPath
    Require-Path $ValPath
    Require-Path $TestPath
    if ($SemanticCodebookPath) {
        Require-Path $SemanticCodebookPath
    }
    if ($SkipAudit) {
        throw "Formal CICIDS2017 training cannot skip split audit."
    }

    $BaselineDir = Join-Path $OutputRoot "neural_baseline_cnn"
    $MlmDir = Join-Path $OutputRoot "mlm"
    $ClassifierDir = Join-Path $OutputRoot "classifier"

    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    Invoke-Step "Verify CICIDS2017 formal dataset preflight" {
        Invoke-CicidsFormalGate
    }

    $auditPath = Join-Path $OutputRoot "split_audit.json"
    if (Test-Path $CicidsSplitAuditPath) {
        Invoke-Step "Reuse existing split audit" {
            Copy-Item -LiteralPath $CicidsSplitAuditPath -Destination $auditPath -Force
        }
    }
    else {
        Invoke-Step "Audit processed split" {
            uv run python scripts/audit_processed_split.py `
                --train-path $TrainPath `
                --val-path $ValPath `
                --test-path $TestPath `
                --ignore-source-file-overlap-for-eligibility `
                --fail-on-warnings `
                --output-path $auditPath
        }
    }
    Invoke-Step "Verify CICIDS2017 formal dataset audit gate" {
        Invoke-CicidsFormalGate $auditPath
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
                    --num-workers $ClassifierNumWorkers `
                    --learning-rate $ClassifierLearningRate `
                    --max-length $MaxLength `
                    --stride $Stride `
                    --max-windows $MaxWindows `
                    --device $Device `
                    --seed $Seed `
                    --major-class-weighting $MajorClassWeighting `
                    --major-class-weight-cap $MajorClassWeightCap `
                    @ConnectionTokenArgs `
                    @ContextTokenArgs `
                    @ContextFeatureArgs `
                    @SemanticCodebookArgs `
                    @InitBertArgs
            }
        }

        if (-not $SkipEval) {
            $classifierEvalCheckpoint = Join-Path $ClassifierDir "classifier.best.pt"
            if (-not (Test-Path $classifierEvalCheckpoint)) {
                $classifierEvalCheckpoint = $classifierCheckpoint
            }
            Write-Host "Classifier eval checkpoint: $classifierEvalCheckpoint" -ForegroundColor Yellow

            $thresholdPath = Join-Path $ClassifierDir "minor_thresholds.json"
            $classifierPredictionsDir = Join-Path $ClassifierDir "predictions"
            $splitDir = Split-Path -Parent $TrainPath
            New-Item -ItemType Directory -Force -Path $classifierPredictionsDir | Out-Null
            Invoke-Step "Calibrate classifier thresholds" {
                uv run traffic-bert eval calibrate-thresholds `
                    --data-path $ValPath `
                    --checkpoint $classifierEvalCheckpoint `
                    --output-path $thresholdPath `
                    --view $View `
                    --batch-size $ClassifierBatchSize `
                    --num-workers $ClassifierNumWorkers `
                    --max-length $MaxLength `
                    --stride $Stride `
                    --max-windows $MaxWindows `
                    --device $Device `
                    @ConnectionTokenArgs `
                    @ContextTokenArgs `
                    @ContextFeatureArgs `
                    @SemanticCodebookArgs
            }

            Invoke-Step "Evaluate Byte-BERT classifier on validation with predictions" {
                uv run traffic-bert eval classifier `
                    --data-path $ValPath `
                    --checkpoint $classifierEvalCheckpoint `
                    --view $View `
                    --batch-size $ClassifierBatchSize `
                    --num-workers $ClassifierNumWorkers `
                    --max-length $MaxLength `
                    --stride $Stride `
                    --max-windows $MaxWindows `
                    --device $Device `
                    @ConnectionTokenArgs `
                    @ContextTokenArgs `
                    @ContextFeatureArgs `
                    @SemanticCodebookArgs `
                    --output-dir (Join-Path $ClassifierDir "val_eval") `
                    --output-predictions-path (Join-Path $classifierPredictionsDir "val_predictions.parquet")
            }

            Invoke-Step "Evaluate Byte-BERT classifier on test with predictions" {
                uv run traffic-bert eval classifier `
                    --data-path $TestPath `
                    --checkpoint $classifierEvalCheckpoint `
                    --view $View `
                    --batch-size $ClassifierBatchSize `
                    --num-workers $ClassifierNumWorkers `
                    --max-length $MaxLength `
                    --stride $Stride `
                    --max-windows $MaxWindows `
                    --device $Device `
                    @ConnectionTokenArgs `
                    @ContextTokenArgs `
                    @ContextFeatureArgs `
                    @SemanticCodebookArgs `
                    --output-dir (Join-Path $ClassifierDir "test_eval") `
                    --output-predictions-path (Join-Path $classifierPredictionsDir "test_predictions.parquet")
            }

            Invoke-Step "Analyze Bot host-window risk layer" {
                uv run python scripts/analyze_bot_host_windows.py `
                    --split-dir $splitDir `
                    --prediction-dir $classifierPredictionsDir `
                    --output-path (Join-Path $OutputRoot "bot_host_window_analysis.json") `
                    --target-label botnet_malware `
                    --window-seconds 30 60 300 `
                    --min-recall 0.80 `
                    --min-f1 0.75
            }
        }
    }

    Invoke-Step "Summarize formal run artifacts" {
        uv run python scripts/summarize_formal_run.py `
            --output-root $OutputRoot `
            --write-json (Join-Path $OutputRoot "formal_run_summary.json") `
            --fail-on-acceptance
    }

    Write-Host ""
    Write-Host "Done. Outputs written to $OutputRoot" -ForegroundColor Green
}
finally {
    Pop-Location
}
