"""Two predeclared open images, raw A7 + F3 full-surface D4 timing only."""
import sys,json,time,importlib.util,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.append(str(ROOT/'vision_prototype/.venv/Lib/site-packages'))
import cv2,numpy as np
cv2.setNumThreads(1)
A=ROOT/'数据资产/05_派生与实验结果'
A7=A/'camera_condition_free_straight_20261030_v7'
sys.path.insert(0,str(A7))
import straight_detector
spec=importlib.util.spec_from_file_location('video_f3_observation',A/'camera_condition_full_surface_joint_20261028_v3/surface_observation.py')
f3=importlib.util.module_from_spec(spec);spec.loader.exec_module(f3)
def read(p):return cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_COLOR if p.parent.parent.name=='images' else cv2.IMREAD_GRAYSCALE)
def js(p):return json.loads(p.read_text(encoding='utf8'))
def main():
 out=HERE/'benchmark_001';out.mkdir(exist_ok=False)
 results=[]
 for role,ident in [('right','dual_20260923_101517__pair0015__right'),('left','L_run_20260923_01__f000015__left')]:
  image_path=A/'camera_condition_images_20261009_v2/images'/f'{role}_camera'/(ident+'.png')
  image=cv2.imdecode(np.fromfile(image_path,np.uint8),cv2.IMREAD_COLOR)
  surface_path=A/'camera_condition_sam2_20261009_v1/all_001/body_branches_002/masks'/f'{role}_camera'/(ident+'.png')
  surface=cv2.imdecode(np.fromfile(surface_path,np.uint8),cv2.IMREAD_GRAYSCALE)
  c3=js(A/'camera_condition_grouped_conic_20261026_v6/all_001/boundaries/records'/f'{role}_camera'/(ident+'.json'))
  data=dict(original=image,surface=surface,pitch=float(c3['record']['pitch_px']),c3_record=c3,c3_exclusion=np.zeros(surface.shape,np.uint8))
  t=time.perf_counter();a7=straight_detector.detect(data);a7time=time.perf_counter()-t
  print(ident,'A7',round(a7time,2),a7['status'],len(a7['candidates']),flush=True)
  t=time.perf_counter();d=f3.detect_surface(data);ftime=time.perf_counter()-t
  np.savez_compressed(out/(ident+'_fields.npz'),surface=surface,merged=d['fit']['smooth_mask'],inferred=d['fit']['inferred_mask'])
  row=dict(id=ident,source=str(image_path),a7_seconds=a7time,a7_status=a7['status'],a7_candidates=len(a7['candidates']),full_D4_seconds=ftime,
   full_D4_curves=len(d['fit']['curves']),full_D4_points=len(d['points']),stages=d['diagnostics']['stage_seconds'],pitch=data['pitch'],
   caveat='Same-frame SAM2 and C3 pitch/shape only; no old A1/A2 entry. Full D4 observer, not complete historical F3 fusion.')
  results.append(row);print(json.dumps(row,ensure_ascii=True),flush=True)
  (out/'timings.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
