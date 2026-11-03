$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\precise_arrangement'
$taskRun = 'D:\precise_arrangement\experiments\entry_conditioned_rope_path\manual_r1_20261103_v1'
$taskPython = 'D:\precise_arrangement\vision_training\.venv\Scripts\python.exe'
$taskReceipts = @()
function Wait-TaskInput([string] $taskRelative) {
    $taskDeadline = [DateTime]::UtcNow.AddMinutes(50)
    $taskFile = Join-Path $taskRun $taskRelative
    while (-not (Test-Path -LiteralPath $taskFile)) {
        if ([DateTime]::UtcNow -gt $taskDeadline) { throw ('Dependency timeout: '+$taskFile) }
        $taskPrior = Join-Path $taskRun 'unet_execution_receipts.json'
        if (Test-Path -LiteralPath $taskPrior) {
            $taskSteps = @(Get-Content -LiteralPath $taskPrior -Raw | ConvertFrom-Json)
            if (@($taskSteps | Where-Object { $_.exit_code -ne 0 }).Count -gt 0) { throw 'U-Net dependency failed' }
        }
        Start-Sleep -Seconds 5
    }
    $taskInput = Get-Content -LiteralPath $taskFile -Raw | ConvertFrom-Json
    if ($taskInput.exit_code -ne 0) { throw ('Dependency exit failure: '+$taskFile) }
}
Wait-TaskInput 'inference\fixed\receipt.json'
foreach ($taskMode in @('prepare-pairs','train','infer-fixed','infer-gallery')) {
    if ($taskMode -eq 'infer-gallery') { Wait-TaskInput 'inference\gallery\receipt.json' }
    $taskStarted = [DateTime]::UtcNow
    $taskTimer = [System.Diagnostics.Stopwatch]::StartNew()
    $taskLog = Join-Path $taskRun ('r1_'+$taskMode+'.log')
    if (Test-Path -LiteralPath $taskLog) { throw ('Existing log: '+$taskLog) }
    & $taskPython -B (Join-Path $taskRun 'r1_pipeline.py') $taskMode --workers 4 2>&1 | Tee-Object -FilePath $taskLog
    $taskCode = $LASTEXITCODE; $taskTimer.Stop()
    $taskReceipts += [PSCustomObject]@{mode=$taskMode;command=@($taskPython,'-B',(Join-Path $taskRun 'r1_pipeline.py'),$taskMode,'--workers','4');started_utc=$taskStarted.ToString('o');elapsed_seconds=$taskTimer.Elapsed.TotalSeconds;exit_code=$taskCode;log=$taskLog}
    $taskReceipts | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskRun 'r1_execution_receipts.json') -Encoding utf8
    if ($taskCode -ne 0) { throw ('R1 step failed: '+$taskMode+' exit'+$taskCode) }
}
