$ErrorActionPreference='Stop'
Set-Location -LiteralPath 'D:\precise_arrangement'
$taskRoot='D:\precise_arrangement\experiments\entry_conditioned_rope_path\manual_u1_p1_20261103_v1'
$taskReceipt=Join-Path $taskRoot 'render_receipt.json'
$taskDeadline=[DateTime]::UtcNow.AddMinutes(15)
while (-not (Test-Path -LiteralPath $taskReceipt)) {
    if ([DateTime]::UtcNow -gt $taskDeadline) { throw 'Render receipt dependency timeout' }
    Start-Sleep -Seconds 3
}
$taskRender=Get-Content -LiteralPath $taskReceipt -Raw | ConvertFrom-Json
if ($taskRender.exit_code -ne 0 -or @($taskRender.rows).Count -ne 292) { throw 'Full render did not pass' }
$taskLog=Join-Path $taskRoot 'audit_u1_p1_repair_003.log'
if (Test-Path -LiteralPath $taskLog) { throw 'Preserve existing audit log' }
$taskStarted=[DateTime]::UtcNow
$taskTimer=[System.Diagnostics.Stopwatch]::StartNew()
& 'D:\precise_arrangement\vision_prototype\.venv\Scripts\python.exe' -B (Join-Path $taskRoot 'audit_u1_p1.py') --workers 4 2>&1 | Tee-Object -FilePath $taskLog
$taskExit=$LASTEXITCODE
$taskTimer.Stop()
[PSCustomObject]@{command=@('vision_prototype/.venv/Scripts/python.exe','-B','experiments/entry_conditioned_rope_path/manual_u1_p1_20261103_v1/audit_u1_p1.py','--workers','4');started_utc=$taskStarted.ToString('o');seconds=$taskTimer.Elapsed.TotalSeconds;actual_exit_code=$taskExit;log=$taskLog} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskRoot 'audit_execution_receipt.json') -Encoding utf8
if ($taskExit -ne 0) { throw ('Independent audit exit'+$taskExit) }
