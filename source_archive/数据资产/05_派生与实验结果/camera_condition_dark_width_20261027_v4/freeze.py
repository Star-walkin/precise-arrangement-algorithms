"""固定三轮开发后的D4源码；全量运行前只执行一次。"""
from datetime import datetime,timezone
from data_io import HERE,read_json,dump,sha,sources

if (HERE/'freeze_all.json').exists():raise FileExistsError('freeze already exists')
audit=read_json(HERE/'pilot_003/audit.json')
assert audit['passed']==12 and audit['failed']==0 and not audit['global_errors']
assert audit['limited_body_cases']==0
current=sources()
assert current==read_json(HERE/'pilot_003/run_context.json')['source_sha256']
dump(HERE/'freeze_all.json',{
    'version':'DR-D4.0.0','name':'沟宽筛选 D4',
    'frozen_utc':datetime.now(timezone.utc).isoformat(),
    'source_sha256':current,'chosen_pilot':'pilot_003',
    'pilot_audit_sha256':sha(HERE/'pilot_003/audit.json'),
    'mechanism_check_sha256':sha(HERE/'selfcheck_records/final_algorithm_synthetic_001.json'),
    'fixed_inputs':400,'changes_after_freeze':'No visual parameter tuning on full400.',
    'basis':'Fixed pilot visual comparison: less fine-strand beading and local waviness; weak-evidence long paths rejected. Incomplete side edges and ambiguous texture remain.',
    'acceptance_scope':'Experimental image outputs for review, not whole-image correctness or physical rope identity.',
    'equipment_access':False,'reserved_access':False})
print({'source_count':len(current),'freeze_sha256':sha(HERE/'freeze_all.json')})
