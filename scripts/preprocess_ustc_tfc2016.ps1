param(
    [string]$Workspace = (Get-Location).Path,
    [string]$SourceRoot = 'data\raw\USTC-TFC2016\extracted\USTC-TFC2016-master',
    [string]$OutputDir = 'data\processed\ustc_tfc2016\files',
    [string]$LogDir = 'data\processed\ustc_tfc2016\logs',
    [string]$ProgressPath = 'data\processed\ustc_tfc2016\preprocess_progress.jsonl',
    [int]$MaxPacketsPerFlow = 16,
    [int]$HeartbeatSeconds = 30,
    [int64]$AffinityMask = 3
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $Workspace

$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'

try {
    $currentProcess = [System.Diagnostics.Process]::GetCurrentProcess()
    $currentProcess.PriorityClass = 'BelowNormal'
    $currentProcess.ProcessorAffinity = [IntPtr]$AffinityMask
} catch {
}

$root = Resolve-Path -LiteralPath $SourceRoot
New-Item -ItemType Directory -Force -Path $OutputDir, $LogDir | Out-Null

function Write-ProgressRecord($record) {
    Add-Content -LiteralPath $ProgressPath -Value ($record | ConvertTo-Json -Compress)
}

$pcaps = Get-ChildItem -LiteralPath $root -Recurse -Filter *.pcap | Sort-Object Length, FullName
Write-ProgressRecord ([ordered]@{
    event = 'batch_start'
    time = (Get-Date).ToString('o')
    total = $pcaps.Count
})

foreach ($pcap in $pcaps) {
    $kind = if ($pcap.FullName -like '*\Benign\*') { 'Benign' } else { 'Malware' }
    $safeName = "$kind`_$($pcap.BaseName -replace '[\\/:*?""<>| ]', '_')"
    $outputPath = Join-Path $OutputDir ($safeName + '.parquet')
    $stdoutPath = Join-Path $LogDir ($safeName + '.stdout.log')
    $stderrPath = Join-Path $LogDir ($safeName + '.stderr.log')

    if (Test-Path -LiteralPath $outputPath) {
        Write-ProgressRecord ([ordered]@{
            event = 'skip_existing'
            time = (Get-Date).ToString('o')
            file = $pcap.Name
            output = $outputPath
        })
        continue
    }

    Write-ProgressRecord ([ordered]@{
        event = 'start'
        time = (Get-Date).ToString('o')
        file = $pcap.Name
        kind = $kind
        size_bytes = $pcap.Length
        output = $outputPath
    })

    $args = @(
        'run', 'traffic-bert', 'data', 'build',
        '--input-path', $pcap.FullName,
        '--output-path', $outputPath,
        '--source-dataset', 'ustc-tfc2016',
        '--split', 'train',
        '--views', 'payload_only',
        '--max-packets-per-flow', "$MaxPacketsPerFlow",
        '--no-keep-empty-payload'
    )
    if ($kind -eq 'Benign') {
        $args += @('--label-source', 'static', '--static-label', 'BENIGN')
    } else {
        $args += @('--label-source', 'filename')
    }

    $start = Get-Date
    $process = Start-Process `
        -FilePath 'uv' `
        -ArgumentList $args `
        -WorkingDirectory (Get-Location) `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -PassThru
    try {
        $process.PriorityClass = 'BelowNormal'
        $process.ProcessorAffinity = [IntPtr]$AffinityMask
    } catch {
    }

    while ($true) {
        $process.Refresh()
        if ($process.HasExited) {
            break
        }
        Start-Sleep -Seconds $HeartbeatSeconds
        $process.Refresh()
        try {
            $cpu = [Math]::Round($process.CPU, 2)
        } catch {
            $cpu = $null
        }
        Write-ProgressRecord ([ordered]@{
            event = 'heartbeat'
            time = (Get-Date).ToString('o')
            file = $pcap.Name
            seconds = [Math]::Round(((Get-Date) - $start).TotalSeconds, 2)
            cpu = $cpu
        })
    }

    $process.WaitForExit()
    $process.Refresh()
    $exitCode = $process.ExitCode
    $seconds = [Math]::Round(((Get-Date) - $start).TotalSeconds, 2)
    $statsPath = [System.IO.Path]::ChangeExtension($outputPath, '.stats.json')
    $rows = $null
    $hasOutput = Test-Path -LiteralPath $outputPath
    $hasStats = Test-Path -LiteralPath $statsPath
    if ($hasStats) {
        try {
            $rows = (Get-Content -LiteralPath $statsPath -Raw | ConvertFrom-Json).rows
        } catch {
        }
    }

    $looksComplete = $hasOutput -and $hasStats -and ($null -ne $rows) -and ($rows -gt 0)
    if (($exitCode -eq 0) -or (($null -eq $exitCode) -and $looksComplete)) {
        Write-ProgressRecord ([ordered]@{
            event = 'done'
            time = (Get-Date).ToString('o')
            file = $pcap.Name
            seconds = $seconds
            exit_code = $exitCode
            rows = $rows
            output = $outputPath
        })
    } else {
        Write-ProgressRecord ([ordered]@{
            event = 'failed'
            time = (Get-Date).ToString('o')
            file = $pcap.Name
            seconds = $seconds
            exit_code = $exitCode
            output = $outputPath
            stderr = $stderrPath
        })
        exit $(if ($null -ne $exitCode) { $exitCode } else { 1 })
    }
}

Write-ProgressRecord ([ordered]@{
    event = 'batch_done'
    time = (Get-Date).ToString('o')
})
