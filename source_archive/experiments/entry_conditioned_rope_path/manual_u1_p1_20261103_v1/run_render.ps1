$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\precise_arrangement'
$taskRun = 'D:\precise_arrangement\experiments\entry_conditioned_rope_path\manual_u1_p1_20261103_v1'
$taskPython = 'D:\precise_arrangement\vision_prototype\.venv\Scripts\python.exe'
$taskLog = Join-Path $taskRun 'render_all.log'
if (Test-Path -LiteralPath $taskLog) { throw 'Render log already exists; do not replace prior run.' }
$taskStart = [DateTime]::UtcNow
$taskTimer = [System.Diagnostics.Stopwatch]::StartNew()
& $taskPython -B (Join-Path $taskRun 'render_four.py') --run all_001 --workers 4 2>&1 | Tee-Object -FilePath $taskLog
$taskCode = $LASTEXITCODE
$taskTimer.Stop()
$taskReceipt = [PSCustomObject]@{command=@($taskPython,'-B',(Join-Path $taskRun 'render_four.py'),'--run','all_001','--workers','4');started_utc=$taskStart.ToString('o');elapsed_seconds=$taskTimer.Elapsed.TotalSeconds;exit_code=$taskCode;log=$taskLog;render_receipt=(Join-Path $taskRun 'render_receipt.json')}
$taskReceipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskRun 'render_execution_receipt.json') -Encoding utf8
if ($taskCode -ne 0) { throw ('Render failed exit'+$taskCode) }
