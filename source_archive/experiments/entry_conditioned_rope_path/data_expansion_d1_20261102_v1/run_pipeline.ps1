$ErrorActionPreference='Stop'
$taskRoot='D:\precise_arrangement\experiments\entry_conditioned_rope_path\data_expansion_d1_20261102_v1'
$trainPython='D:\precise_arrangement\vision_training\.venv\Scripts\python.exe'
$imagePython='D:\precise_arrangement\vision_prototype\.venv\Scripts\python.exe'
$records=New-Object System.Collections.Generic.List[object]
function Run-TaskStep([string]$Name,[string]$Python,[string[]]$Arguments) {
    $tick=[Diagnostics.Stopwatch]::StartNew()
    $started=[DateTime]::UtcNow.ToString('o')
    $log=Join-Path $taskRoot ($Name+'.log')
    if(Test-Path -LiteralPath $log){throw ('Preserve previous log '+$log)}
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $log
    $code=$LASTEXITCODE
    $records.Add([pscustomobject]@{name=$Name;executable=$Python;arguments=$Arguments;started_utc=$started;seconds=$tick.Elapsed.TotalSeconds;exit_code=$code;log=$log})
    $records | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskRoot 'execution_receipts.json') -Encoding utf8
    if($code -ne 0){throw ('Step failed '+$Name+' exit '+$code)}
}
Run-TaskStep 'smallcheck' $trainPython @('-B',"$taskRoot\train_infer.py",'smallcheck')
Run-TaskStep 'train_D0' $trainPython @('-B',"$taskRoot\train_infer.py",'train','--branch','D0')
Run-TaskStep 'train_D1' $trainPython @('-B',"$taskRoot\train_infer.py",'train','--branch','D1')
Run-TaskStep 'select_common' $trainPython @('-B',"$taskRoot\train_infer.py",'select')
Run-TaskStep 'inference_fixed' $trainPython @('-B',"$taskRoot\train_infer.py",'infer-fixed')
Run-TaskStep 'inference_gallery' $trainPython @('-B',"$taskRoot\train_infer.py",'infer-gallery','--branch','D1')
Run-TaskStep 'evaluate_fixed' $imagePython @('-B',"$taskRoot\evaluation.py",'run')
Run-TaskStep 'evaluate_train_fit' $trainPython @('-B',"$taskRoot\evaluation.py",'train-fit')
Run-TaskStep 'render_292' $imagePython @('-B',"$taskRoot\render_export.py",'--workers','4')
Run-TaskStep 'compare_fixed' $imagePython @('-B',"$taskRoot\comparison.py")
Run-TaskStep 'audit_complete' $trainPython @('-B',"$taskRoot\audit.py",'--workers','4')
