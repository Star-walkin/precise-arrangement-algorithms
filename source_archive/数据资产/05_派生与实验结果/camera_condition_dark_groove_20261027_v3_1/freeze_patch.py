"""结构修复的独立冻结；不覆盖0.0的协议、快照、失败运行。"""
from datetime import datetime,timezone
import json
from data_io import HERE,sources,sha,read_json,dump
from groove_field import PARAMETERS as FIELD
from ordered_paths import PARAMETERS as FIT
parent=HERE.parent/'camera_condition_dark_groove_20261027_v3'
target=HERE/'freeze_all.json'
if target.exists():raise FileExistsError(target)
old=read_json(parent/'freeze_all.json')
assert json.loads(json.dumps(FIELD))==old['field_parameters']
assert json.loads(json.dumps(FIT))==old['fit_parameters']
for filename in ['data_io.py','run.py','groove_field.py','audit.py']:
    assert sha(HERE/filename)==sha(parent/filename)
check=read_json(HERE/'repair_check.json')
dump(target,{'version':'DR-D3.0.1','name':'暗沟约束 D3','final_run':'all_001',
    'frozen_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sources(),
    'field_parameters':FIELD,'fit_parameters':FIT,
    'parent_freeze_sha256':sha(parent/'freeze_all.json'),
    'parent_failed_audit_sha256':sha(parent/'all_001/independent_audit.json'),
    'repair_check_sha256':sha(HERE/'repair_check.json'),
    'scope':'Only actual overlapping-neighbor spacing guard repair; same input400 and image/fitting parameters. No further image-guided tuning.',
    'equipment_access':False,'reserved_access':False,'ground_truth_accuracy':None})
print({'version':'DR-D3.0.1','source_count':len(sources()),'sha256':sha(target)})
