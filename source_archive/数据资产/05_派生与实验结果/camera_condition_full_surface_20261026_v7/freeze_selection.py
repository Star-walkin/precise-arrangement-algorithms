"""在400图运行前冻结选定参数、固定38图证据及输入规则。"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    path=HERE/'freeze_all_001.json'
    if path.exists():raise FileExistsError(path)
    pilot=HERE/'pilot_002';evaluation=HERE/'evaluation_002'
    s=read(pilot/'summary.json');a=read(pilot/'independent_audit.json');e=read(evaluation/'summary.json')
    assert s['planned']==s['completed']==38 and s['failed']==0 and s['source_unchanged']
    assert a['planned']==a['passed']==38 and a['failed']==0 and not a['global_errors']
    assert e['metric_gate_pass'] and e['source_unchanged']
    context=read(pilot/'run_context.json')
    for p,h in context['source_sha256'].items():assert sha(Path(p))==h
    assert e['source_sha256_start']['full_surface_core.py']==sha(HERE/'full_surface_core.py')
    paths=[pilot/'summary.json',pilot/'independent_audit.json',pilot/'run_context.json',
           evaluation/'summary.json',evaluation/'run_context.json',HERE/'protocol.json',
           HERE/'protocol_revision_002.json',HERE/'revision_notes.json',HERE/'evaluation_protocol.json']
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'version':'RB-C4.0.0',
            'selected_pilot':str(pilot),'selected_evaluation':str(evaluation),
            'source_sha256':context['source_sha256'],'evidence_sha256':{str(p):sha(p) for p in paths},
            'bright_hit_new':e['bright']['overall']['v7']['all_point_hit_rate'],
            'bright_hit_c3':e['bright']['overall']['c3']['all_point_hit_rate'],
            'edge_hit_new':e['edge']['overall']['v7']['all_point_hit_rate'],
            'edge_prediction_equal_simple':e['edge']['predictions_identical_to_simple_when_no_dark_candidates'],
            'decision':'Run full400 with frozen second revision. Pixel/structure and bright nonregression passed; dark/tangency accuracy not established. Keep SAM semantic failure ruler_left_05 and all first-round outputs.',
            'first_run_not_overwritten':True,'remaining_tuning_rounds':0}
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Frozen all_001 selection')
if __name__=='__main__':main()
