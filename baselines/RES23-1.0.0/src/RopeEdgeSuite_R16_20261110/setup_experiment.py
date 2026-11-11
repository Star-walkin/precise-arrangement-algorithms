from pathlib import Path
import json, shutil
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'RopeEdgeSuite_R15_20261109'
for name in ['common.py','boundary_bg9.py','camera_cg2.py','plan.json']:
    target=ROOT/name
    if not target.exists():shutil.copy2(OLD/name,target)
rejection=dict(status='rejected_candidate',reason='SG3 temporal drift onto yellow flange specular reflection',
    video='dual_20260923_113745__left',confirmed_bad_frames=[397,400],
    action='Stopped SG3+BG9 replay; retained intermediate SG3/BG5 computation for paired branch experiment',
    baseline_SG2_current_frame_candidate_visually_correct=True,
    next='SG4 conservative branch priority and independent current entrance verification for fallback')
(OLD/'rejected_candidate.json').write_text(json.dumps(rejection,ensure_ascii=False,indent=2),encoding='utf8')
log=json.loads((OLD/'iteration_log.json').read_text(encoding='utf8'))
if not any(t['experiment']=='SG3 reflection regression' for t in log['trials']):
    log['trials'].append(dict(experiment='SG3 reflection regression',result=rejection))
(OLD/'iteration_log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf8')
