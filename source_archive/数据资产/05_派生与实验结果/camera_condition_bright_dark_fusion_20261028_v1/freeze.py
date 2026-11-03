from datetime import datetime,timezone
from fusion_io import HERE,read_json,dump,sha,sources,baseline_unchanged
assert not (HERE/'freeze_all.json').exists()
a=read_json(HERE/'pilot_001/audit.json')
assert a['passed']==12 and a['failed']==0 and not a['global_errors']
assert sources()==read_json(HERE/'pilot_001/run_context.json')['source_sha256'] and baseline_unchanged()
dump(HERE/'freeze_all.json',{'version':'RB-F1.0.0','frozen_utc':datetime.now(timezone.utc).isoformat(),
 'source_sha256':sources(),'pilot_audit_sha256':sha(HERE/'pilot_001/audit.json'),
 'visual_diagnosis_sha256':sha(HERE/'pilot_001/记录/qa_visual/visual_diagnosis.json'),
 'task':'Merge existing accepted C3 bright and D4 dark results on fixed400. No refit or bridge.',
 'known_inherited_problem':'ruler_left_05 old SAM union surface misses visible physical surface above incoming rope; preserved, not silently expanded.',
 'equipment_access':False,'reserved_access':False})
print({'freeze_sha256':sha(HERE/'freeze_all.json')})
