"""Current-image P1/H6 probes, no inherited target predictions."""
import sys,json,time,hashlib
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
ASSETS=ROOT.parent; LC1=ASSETS/'LC1_locked_column_20261105_v1'
sys.path.insert(0,str(LC1))
from lc1_p1_adapter import P1H6Adapter
SOURCE=ROOT.parents[3]/'双相机同步/dual_20260924_111500/left_DA9627065.avi'
def main():
    out=ROOT/'p1_probes';out.mkdir(exist_ok=True)
    stem='dual_20260924_111500__pair0015__left'
    mp=ASSETS/'camera_condition_sam2_20261009_v1/all_001/body_branches_002/masks/left_camera'/f'{stem}.png'
    pp=ASSETS/'camera_condition_bright_rope_boundaries_20261023_v2/all_002/boundaries/records/left_camera'/f'{stem}.json'
    mask=cv2.imdecode(np.fromfile(mp,np.uint8),0)
    pitch=json.loads(pp.read_text(encoding='utf8'))['record']['pitch_px']
    model=ROOT.parents[3]/'临时文件/step_0750.pt'
    engine=P1H6Adapter(mask,pitch,model,threads=2,camera='left')
    cap=cv2.VideoCapture(str(SOURCE));summary=[]
    for f in [220,245,260,290,320]:
        dest=out/f'{f:06d}.json'
        if dest.exists():r=json.loads(dest.read_text(encoding='utf8'))
        else:
            cap.set(1,f);ok,im=cap.read();assert ok
            tick=time.perf_counter();r=engine.step(im)
            r.update(frame_index=f,source_pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest(),elapsed=time.perf_counter()-tick)
            dest.write_text(json.dumps(r,ensure_ascii=False,allow_nan=False),encoding='utf8')
            for k,h in enumerate(engine.last_heatmaps):np.savez_compressed(out/f'{f:06d}_E{k}.npz',response=h)
        row=dict(frame=f,status=r['status'],whole=r['whole_path_candidate_supported'],reasons=r['reasons'],elapsed=r.get('elapsed'))
        summary.append(row);print(json.dumps(row),flush=True)
    cap.release();(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
if __name__=='__main__':main()
