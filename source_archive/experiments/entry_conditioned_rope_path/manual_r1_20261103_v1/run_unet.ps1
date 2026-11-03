$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\precise_arrangement'
$taskRun = 'D:\precise_arrangement\experiments\entry_conditioned_rope_path\manual_r1_20261103_v1'
$taskPython = 'D:\precise_arrangement\vision_training\.venv\Scripts\python.exe'
$taskReceipts = @()
foreach ($taskMode in @('smallcheck','train','select','infer-fixed','infer-gallery','fit-stats')) {
    $taskStarted = [DateTime]::UtcNow
    $taskTimer = [System.Diagnostics.Stopwatch]::StartNew()
    $taskLog = Join-Path $taskRun ($taskMode + '.log')
    if (Test-Path -LiteralPath $taskLog) { throw ('Existing log: ' + $taskLog) }
    & $taskPython -B (Join-Path $taskRun 'train_manual.py') $taskMode 2>&1 | Tee-Object -FilePath $taskLog
    $taskCode = $LASTEXITCODE
    $taskTimer.Stop()
    $taskReceipts += [PSCustomObject]@{mode=$taskMode;command=@($taskPython,'-B',(Join-Path $taskRun 'train_manual.py'),$taskMode);started_utc=$taskStarted.ToString('o');elapsed_seconds=$taskTimer.Elapsed.TotalSeconds;exit_code=$taskCode;log=$taskLog}
    $taskReceipts | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskRun 'unet_execution_receipts.json') -Encoding utf8
    if ($taskCode -ne 0) { throw ('U-Net step failed: '+$taskMode+' exit'+$taskCode) }
}
