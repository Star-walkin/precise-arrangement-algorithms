"""为结构错误修复建立独立版本；原D3.0.0及失败产物不修改。"""
from data_io import HERE,ROOT,read_json,dump,sha
parent=HERE.parent/'camera_condition_dark_groove_20261027_v3'
target=HERE/'protocol.json'
if target.exists():raise FileExistsError(target)
p=read_json(parent/'protocol.json')
p.update({'version':'DR-D3.0.1','name':'暗沟约束 D3',
   'parent_experiment':str(parent),'parent_freeze_sha256':sha(parent/'freeze_all.json'),
   'parent_failed_audit_sha256':sha(parent/'all_001/independent_audit.json'),
   'patch_reason':'Global neighboring IDs are not necessarily local visible neighbors after an intermediate path ends. Check every overlapping pair under the same spacing rule.',
   'patch_scope':'Structural correctness repair only; no image/fit thresholds or model tuning after400. Preserve original3 pilots, control and398/400 failed audit in parent.',
   'budget':'Two previously failing cases as regression checks; same400 replay after source freeze; no additional image-guided parameter optimization.'})
dump(target,p)
dump(HERE/'index_before.json',{str(q):sha(q) for q in [ROOT/'数据资产/README.md',HERE.parent/'README.md',ROOT/'工程导航/05_实验版本档案/README.md']})
dump(HERE/'repair_protocol.json',{'version':'DR-D3.0.1','scope':p['patch_scope'],
    'fixed_failure_cases':['images/right_camera/dual_20260924_111143__pair0135__right.png',
      'images/left_camera/L_run_stop_run_20260922_01__f004322__left.png'],
    'required':'Unchanged 400 denominator and yellow coordinates; no local close pairs; preserve parent failed audit.'})
print({'version':p['version'],'protocol_sha256':sha(target)})
