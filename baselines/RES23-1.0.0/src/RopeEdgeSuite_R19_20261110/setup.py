from pathlib import Path
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'RopeEdgeSuite_R18_20261110'
for name in ['common.py','boundary_bg9.py','boundary_bg10.py','tracker_sg5.py','plan.json','render.py','run_one.py']:
    if (ROOT/name).exists():continue
    text=(OLD/name).read_text(encoding='utf8')
    if name in ['render.py','run_one.py']:
        text=text.replace('RES-1.8','RES-1.8.1').replace('RES18','RES181').replace('BG10','BG11')
        text=text.replace('from boundary_bg10 import Boundary','from boundary_bg11 import Boundary')
        text=text.replace("ROOT/'boundary_bg10.py'","ROOT/'boundary_bg11.py',ROOT/'boundary_bg10.py'")
    (ROOT/name).write_text(text,encoding='utf8')
