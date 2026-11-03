"""给父版本追加失败结论，不覆盖任何预测、协议、快照或评价。"""
from data_io import HERE,read_json,dump,sha
parent=HERE.parent/'camera_condition_dark_groove_20261027_v3'
target=parent/'experiment_report.json'
if target.exists():raise FileExistsError(target)
audit=read_json(parent/'all_001/independent_audit.json')
evidence=['freeze_all.json','all_001/summary.json','all_001/independent_audit.json',
          'all_001/qa_visual_additional.json','pilot_003/qa_visual_final.json','ablation_comparison.json']
dump(target,{'version':'DR-D3.0.0','status':'development output retained; full structural acceptance failed',
    'summary':read_json(parent/'all_001/summary.json'),
    'audit':{'planned':400,'passed':audit['passed'],'failed':audit['failed'],'global_errors':audit['global_errors']},
    'failed_cases':[c for c in audit['cases'] if c['status']=='failed'],
    'reason':'Global neighboring row IDs fail to capture new local neighbors when an intermediate path ends; two images have close curves.',
    'fixed_additional_visual_review':'12 images/26 components:3 qualitative improvements,6 mixed,3 ordinary-cyan coverage regressions; not truth accuracy.',
    'follow_up_version':'DR-D3.0.1','follow_up_directory':str(HERE),
    'follow_up_scope':'Only overlapping-pair spacing guard repair, no image or fit parameter tuning; see separate source freeze and full audit.',
    'evidence_sha256':{p:sha(parent/p) for p in evidence},
    'equipment_access':False,'reserved_access':False,'ground_truth_accuracy':None})
print({'retained_parent_report_sha256':sha(target)})
