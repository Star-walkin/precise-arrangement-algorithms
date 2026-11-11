"""Authorized two-development-video real B750/H6 current-frame smoke."""
from pathlib import Path
import hashlib,json,sys,time,traceback
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lc1_p1_adapter import P1H6Adapter
ASSETS=ROOT.parent
MODEL=Path(r"E:\study\program\济宁\临时文件\step_0750.pt")
DEV=[
 ("right113305","dual_20260923_113305","right",r"E:\study\program\济宁\双相机同步\dual_20260923_113305\right_DA9626975.avi"),
 ("left111500","dual_20260924_111500","left",r"E:\study\program\济宁\双相机同步\dual_20260924_111500\left_DA9627065.avi"),
]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_image(p,image):
 cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,93])[1].tofile(str(p))
def run():
 out=ROOT/"p1_runtime"/"smoke_current_frame";out.mkdir(exist_ok=True)
 receipt={"adapter_sha256":sha(ROOT/"lc1_p1_adapter.py"),"model_sha256":sha(MODEL),"rows":[]}
 for name,session,camera,video in DEV:
  try:
   stem=f"{session}__pair0015__{camera}"
   mp=ASSETS/"camera_condition_sam2_20261009_v1"/"all_001"/"body_branches_002"/"masks"/f"{camera}_camera"/f"{stem}.png"
   pp=ASSETS/"camera_condition_bright_rope_boundaries_20261023_v2"/"all_002"/"boundaries"/"records"/f"{camera}_camera"/f"{stem}.json"
   mask=cv2.imdecode(np.fromfile(mp,np.uint8),cv2.IMREAD_GRAYSCALE)
   pitch=json.loads(pp.read_text(encoding="utf-8"))["record"]["pitch_px"]
   cap=cv2.VideoCapture(video);ok,bgr=cap.read();cap.release()
   assert ok and bgr.shape==(3000,4096,3)
   start=time.perf_counter()
   engine=P1H6Adapter(mask,pitch,MODEL,threads=2,camera=camera)
   record=engine.step(bgr)
   elapsed=time.perf_counter()-start
   record.update(smoke_source_video=video,smoke_frame_index=0,smoke_original_pixel_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),
                 source_mask_sha256=sha(mp),source_pitch_record_sha256=sha(pp))
   dest=out/(name+"__f0.json");dest.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
   overlay=bgr.copy()
   p=np.asarray(record["path_xy"],float).reshape(-1,2)
   seg=record["diagnostics"]["path_segment_support"]
   stage=record["diagnostics"].get("path_stage",[])
   for i in range(1,len(p)):
    if seg[i-1]:
     color=(50,235,40) if stage[i]=="free" else (230,80,230)
     cv2.line(overlay,tuple(np.rint(p[i-1]).astype(int)),tuple(np.rint(p[i]).astype(int)),color,5,cv2.LINE_AA)
    elif i%5<2:
     cv2.line(overlay,tuple(np.rint(p[i-1]).astype(int)),tuple(np.rint(p[i]).astype(int)),(0,140,255),3,cv2.LINE_AA)
   seed=record.get("seed_xy")
   if seed:
    q=np.rint(seed).astype(np.int32)
    cv2.arrowedLine(overlay,tuple(q[0]),tuple(q[-1]),(255,255,255),7,cv2.LINE_AA,tipLength=.2)
   cv2.putText(overlay,f"{name} original f0 real B750+H6 | {record['status']}",(55,90),cv2.FONT_HERSHEY_SIMPLEX,1.9,(255,255,255),4,cv2.LINE_AA)
   cv2.putText(overlay,"green/purple: current supported | orange dashed: diagnostic only",(55,155),cv2.FONT_HERSHEY_SIMPLEX,1.3,(255,255,255),3,cv2.LINE_AA)
   save_image(out/(name+"__f0_overlay.jpg"),overlay)
   for k,heat in enumerate(engine.last_heatmaps):
    np.savez_compressed(out/(name+f"__f0_B750_E{k+1}.npz"),response=heat)
    half=cv2.resize(bgr,(2048,1500),interpolation=cv2.INTER_AREA)
    hm=cv2.applyColorMap(np.rint(heat*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    half[525:]=hm
    save_image(out/(name+f"__f0_B750_E{k+1}.jpg"),half)
   row=dict(case=name,status=record["status"],seconds=elapsed,model_forward_count=record["model_forward_count"],
            seed_xy=record["seed_xy"],path_point_count=len(record["path_xy"]),line_points=len(record["line_points_xy"]),
            whole_path_candidate_supported=record["whole_path_candidate_supported"],reasons=record["reasons"],
            H6_track_count=record["diagnostics"].get("H6_track_count"),
            support=record["diagnostics"].get("whole_path_support_checks"),record=str(dest),record_sha256=sha(dest))
   receipt["rows"].append(row)
   print(json.dumps(row,ensure_ascii=False),flush=True)
  except Exception:
   error=dict(case=name,error=traceback.format_exc());receipt["rows"].append(error)
   print(json.dumps(error,ensure_ascii=False),flush=True)
 receipt["passed_runtime_smoke"]=all("error" not in r for r in receipt["rows"])
 (out/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
 return 0 if receipt["passed_runtime_smoke"] else 1
if __name__=="__main__":sys.exit(run())

