import shutil
from common import *
names=['common.py','boundary_bg9.py','boundary_bg10.py','camera_cg2.py','tracker_sg5.py','engine.py','run_one.py','run_batch.py','run_fresh.py','render.py','plan.json']
out=ROOT/'frozen';out.mkdir(exist_ok=True);hashes={}
for name in names:
    p=ROOT/name;shutil.copy2(p,out/name);hashes[str(p)]=sha(p)
for p in [R12/'tracker_sg2.py',R11/'region_bg5_refined.py',PARENT/'RopeEdgeSuite_R15_20261109/tracker_sg3.py']:
    hashes[str(p)]=sha(p)
dump(out/'hashes.json',dict(algorithm='RES-1.8/SG5-BG10-CG2',files=hashes,
    unchanged_SG5_component_sha_matches_R17=sha(ROOT/'tracker_sg5.py')==sha(PARENT/'RopeEdgeSuite_R17_20261110/tracker_sg5.py')))
dump(PARENT/'RopeEdgeSuite_R17_20261110/superseded.json',dict(reason='BG10 improves contaminated initial shape; SG5 component unchanged',
    initial_window_result=dict(video='dual_20260923_113305__right',frames=80,BG9_passes=56,BG10_passes=78),
    current_folder=str(ROOT),partial_outputs_not_final=True))
