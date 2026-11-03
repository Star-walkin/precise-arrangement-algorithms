$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\precise_arrangement'
$renderRun = 'D:\precise_arrangement\experiments\entry_conditioned_rope_path\manual_r1_20261103_v1'
$renderPython = 'D:\precise_arrangement\vision_prototype\.venv\Scripts\python.exe'
$renderScript = Join-Path $renderRun 'render_results.py'
$renderLog = Join-Path $renderRun 'render_results.log'
if (Test-Path -LiteralPath $renderLog) { throw 'Preserve existing render log; use a new run.' }
$renderStarted = [DateTime]::UtcNow
$renderTimer = [System.Diagnostics.Stopwatch]::StartNew()
& $renderPython -B $renderScript --workers 4 2>&1 | Tee-Object -FilePath $renderLog
$renderCode = $LASTEXITCODE
$renderTimer.Stop()
[PSCustomObject]@{command=@($renderPython,'-B',$renderScript,'--workers','4');started_utc=$renderStarted.ToString('o');elapsed_seconds=$renderTimer.Elapsed.TotalSeconds;exit_code=$renderCode;log=$renderLog;output_directory='C:\precise_arrangement_results\manual_r1_20261103_v1\final_292'} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $renderRun 'render_execution_receipt.json') -Encoding utf8
exit $renderCode
