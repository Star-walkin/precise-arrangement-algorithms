param([Parameter(Mandatory=$true)][string]$RunName)
$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
$projectRoot = Split-Path (Split-Path $experimentRoot -Parent) -Parent
if ($RunName -notmatch '^run_[0-9]{8}_[0-9]{3}$') { throw 'Use a new run_YYYYMMDD_NNN name.' }
$targetRun = Join-Path $experimentRoot ('runs/' + $RunName)
if (Test-Path -LiteralPath $targetRun) { throw 'Run exists; preserve it and choose a new name.' }
& (Join-Path $projectRoot 'vision_prototype/.venv/Scripts/python.exe') -B (Join-Path $experimentRoot 'run_supervision_experiment.py') --run $targetRun
if ($LASTEXITCODE -ne 0) { throw ('Experiment exited ' + $LASTEXITCODE) }
