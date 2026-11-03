"""Post-experiment inventory only. No input images, labels or inference are loaded."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[4]
RUN=ROOT/'vision_geometry_lab/outputs/v27_material_observability/v27_20261006_002'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
result={'created_utc':datetime.now(timezone.utc).isoformat(),
 'protocol_sha256':sha(RUN/'protocol.json'),
 'new_tests':{'command':r'.\vision_prototype\.venv\Scripts\python.exe -m pytest vision_geometry_lab/experiments/v27_material_observability -q --junitxml=vision_geometry_lab/outputs/v27_material_observability/v27_20261006_002/tests.xml','passed':114,'exit_code':0},
 'v26_regression':{'command':r'.\vision_prototype\.venv\Scripts\python.exe -m pytest vision_geometry_lab/experiments/v26_causal_material -q --junitxml=vision_geometry_lab/outputs/v27_material_observability/v27_20261006_002/v26_regression_tests.xml','passed':105,'exit_code':0},
 'launcher_validation':{'command':r'.\vision_geometry_lab\复核V27材料与换层.cmd --validate-only','exit_code':0},
 'conditional_branches':{'CoTracker3':'not_executed_license_applicability_unconfirmed_no_weight_loaded',
 'candidate_separability':'not_executed_no_independent_material_correspondence'},
 'accuracy':None,'metric_velocity_m_s':None,'t2':None,'hardware_connected':False,
 'output_sha256':{},'document_sha256':{},'inventory_excludes':['this receipt','mutable blind_review/annotations.json']}
for path in sorted(RUN.rglob('*')):
 if not path.is_file() or path.name in ('delivery_receipt.json','annotations.json'):continue
 result['output_sha256'][path.relative_to(RUN).as_posix()]=sha(path)
for name in ('vision_geometry_lab/V27_材料运动可辨识性与最小真值闭环.md','vision_geometry_lab/V27_最小补证与采样清单.md','vision_geometry_lab/experiments/v27_material_observability/盲化材料与换层复核使用说明.md'):
 result['document_sha256'][name]=sha(ROOT/name)
with (RUN/'delivery_receipt.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({'inventoried_output_files':len(result['output_sha256']),'documents':len(result['document_sha256'])}))
