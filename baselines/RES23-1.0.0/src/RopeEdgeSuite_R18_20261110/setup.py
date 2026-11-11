from pathlib import Path
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'RopeEdgeSuite_R17_20261110'
names=['common.py','boundary_bg9.py','camera_cg2.py','plan.json','inventory_all_sources.json','control_scope.json','control_audit.json',
    'render.py','run_one.py','run_batch.py','engine.py','run_fresh.py','test_fresh_parity.py','compare_results.py',
    'build_results.py','make_review_sheet.py','inspect_progress.py','test_sg5.py','tracker_sg5.py','sg5_regression_test.json']
for name in names:
    target=ROOT/name
    if target.exists():continue
    text=(OLD/name).read_text(encoding='utf8')
    if name.endswith('.py') and name not in ['boundary_bg9.py','tracker_sg5.py','common.py','camera_cg2.py']:
        for a,b in [('RES-1.7','RES-1.8'),('RES17','RES18'),('BG9','BG10'),('from boundary_bg9 import Boundary','from boundary_bg10 import Boundary'),("ROOT/'boundary_bg9.py'","ROOT/'boundary_bg10.py',ROOT/'boundary_bg9.py'")]:
            text=text.replace(a,b)
    target.write_text(text,encoding='utf8')
