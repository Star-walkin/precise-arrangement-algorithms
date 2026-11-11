from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'RopeEdgeSuite_R16_20261110'
names=['common.py','boundary_bg9.py','camera_cg2.py','plan.json','render.py','run_one.py','run_batch.py',
    'engine.py','run_fresh.py','test_fresh_parity.py','compare_results.py','build_results.py','make_review_sheet.py',
    'inspect_progress.py','test_sg4.py','tracker_sg4.py']
for name in names:
    outname=name.replace('sg4','sg5')
    if (ROOT/outname).exists():continue
    text=(OLD/name).read_text(encoding='utf8')
    for a,b in [('RES-1.6','RES-1.7'),('RES16','RES17'),('SG4','SG5'),('sg4','sg5')]:text=text.replace(a,b)
    (ROOT/outname).write_text(text,encoding='utf8')
rejection=dict(status='rejected_candidate',reason='Blind strict-branch priority inherited original SG2 false rope texture on 113305 frames 499/540',
    next='Verify both branches with current fixed-entry measurements before selection or identity update',
    retained='all partial videos and logs; R15 independent base computation kept running')
(OLD/'rejected_candidate.json').write_text(json.dumps(rejection,ensure_ascii=False,indent=2),encoding='utf8')
