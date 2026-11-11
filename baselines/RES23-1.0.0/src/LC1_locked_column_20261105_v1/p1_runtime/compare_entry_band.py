"""Explicit entry-band/full-frame comparison and independent S1 gap audit."""
from pathlib import Path
import hashlib,json,sys,time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lc1_p1_adapter import P1H6Adapter
from lc1_s1_adapter import S1Adapter
from smoke_current import DEV,ASSETS,MODEL,save_image,sha

def nearest_segment_distances(points,segments):
 p=np.asarray(points,float).reshape(-1,2)
 distance=np.full(len(p),np.inf)
 for segment in segments:
  q=np.asarray(segment,float).reshape(-1,2)
  for a,b in zip(q[:-1],q[1:]):
   delta=b-a;den=float(delta@delta)
   if den<1.e-12:continue
   t=np.clip((p-a)@delta/den,0.,1.)
   distance=np.minimum(distance,np.linalg.norm(p-a-t[:,None]*delta,axis=1))
 return distance
def runs(flag):
 z=np.diff(np.r_[False,flag,False].astype(np.int8))
 return list(zip(np.flatnonzero(z==1),np.flatnonzero(z==-1)))
def gaps(record,source):
 p=np.asarray(record["path_xy"],float);diag=record["diagnostics"]
 stage=np.asarray(diag["path_stage"]);observed=np.asarray(diag["path_point_support"],bool)
 segments=np.asarray(diag["path_segment_support"],bool)
 answer=[]
 for a,b in runs((~segments)&(stage[:-1]=="free")&(stage[1:]=="free")):
  q=p[a:b+1];length=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum())
  if length<30:continue
  ds=nearest_segment_distances(q,source["supported_segments_xy"])
  answer.append(dict(start_path_index=int(a),end_path_index=int(b),
    start_original_xy=p[a].tolist(),end_original_xy=p[b].tolist(),
    length_original_px=length,stage="free",point_status_counts={str(k):int(v) for k,v in zip(*np.unique(np.asarray(diag["path_point_status"])[a:b+1],return_counts=True))},
    nearest_actual_S1_segment_median_distance_px=float(np.median(ds)),
    nearest_actual_S1_segment_max_distance_px=float(max(ds)),
    S1_actual_coverage_fraction_within_8px=float((ds<=8).mean()),
    S1_actual_coverage_fraction_within_12px=float((ds<=12).mean())))
 return answer
def geometry_compare(baseline,band):
 answer={}
 for stage in ["free","transition","surface"]:
  pp=[]
  for record in [baseline,band]:
   p=np.asarray(record["path_xy"],float)
   ss=np.asarray(record["diagnostics"].get("path_stage",[]))
   p=p[ss==stage];p=p[np.argsort(p[:,0])] if len(p) else p;pp.append(p)
  a,b=pp
  item=dict(full_point_count=len(a),band_point_count=len(b))
  if min(len(a),len(b))>=2:
   lo=max(a[0,0],b[0,0]);hi=min(a[-1,0],b[-1,0])
   if hi>lo:
    x=np.linspace(lo,hi,200)
    diff=np.abs(np.interp(x,a[:,0],a[:,1])-np.interp(x,b[:,0],b[:,1]))
    item.update(common_x_extent_original_px=[float(lo),float(hi)],median_y_difference_px=float(np.median(diff)),maximum_y_difference_px=float(diff.max()))
  answer[stage]=item
 return answer
def main():
 out=ROOT/"p1_runtime"/"entry_band_comparison";out.mkdir(exist_ok=True)
 report=dict(adapter_sha256=sha(ROOT/"lc1_p1_adapter.py"),
   surface_mode_default="full",explicit_mode_compared="entry_band",
   halfwidth_pitch=8.,halo_pitch=2.,cases=[])
 s1=S1Adapter(retain_step_records=True)
 for name,session,camera,video in DEV:
  stem=f"{session}__pair0015__{camera}"
  mp=ASSETS/"camera_condition_sam2_20261009_v1"/"all_001"/"body_branches_002"/"masks"/f"{camera}_camera"/f"{stem}.png"
  pp=ASSETS/"camera_condition_bright_rope_boundaries_20261023_v2"/"all_002"/"boundaries"/"records"/f"{camera}_camera"/f"{stem}.json"
  mask=cv2.imdecode(np.fromfile(mp,np.uint8),cv2.IMREAD_GRAYSCALE)
  pitch=json.loads(pp.read_text(encoding="utf-8"))["record"]["pitch_px"]
  cap=cv2.VideoCapture(video);ok,bgr=cap.read();cap.release();assert ok
  baseline_path=ROOT/"p1_runtime"/"smoke_current_frame"/(name+"__f0.json")
  baseline=json.loads(baseline_path.read_text(encoding="utf-8"))
  assert baseline["smoke_original_pixel_sha256"]==hashlib.sha256(bgr.tobytes()).hexdigest()
  engine=P1H6Adapter(mask,pitch,MODEL,threads=2,camera=camera,surface_mode="entry_band")
  tick=time.perf_counter();band=engine.step(bgr);runtime=time.perf_counter()-tick
  band_path=out/(name+"__f0_entry_band.json");band_path.write_text(json.dumps(band,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
  current_s1=s1.step(bgr,baseline["seed_xy"],seed_source=baseline["seed_source"])
  sp=out/(name+"__f0_S1_current_A7_seed.json");sp.write_text(json.dumps(current_s1,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
  overlay=bgr.copy()
  # Independent S1 blue lines are only actual supported optical segments.
  for segment in current_s1["supported_segments_xy"]:
   q=np.rint(segment).astype(np.int32)
   if len(q)>1:cv2.polylines(overlay,[q],False,(255,150,0),7,cv2.LINE_AA)
  for record,color,dashed in [(baseline,(230,70,230),True),(band,(60,235,40),False)]:
   q=np.rint(record["path_xy"]).astype(np.int32)
   for i in range(1,len(q)):
    supported=record["diagnostics"]["path_segment_support"][i-1]
    if dashed:
     if i%5<2:cv2.line(overlay,tuple(q[i-1]),tuple(q[i]),color,3,cv2.LINE_AA)
    elif supported:cv2.line(overlay,tuple(q[i-1]),tuple(q[i]),color,3,cv2.LINE_AA)
    elif i%5<2:cv2.line(overlay,tuple(q[i-1]),tuple(q[i]),(0,140,255),3,cv2.LINE_AA)
  cv2.putText(overlay,f"{name} f0 full purple dash / band supported green / band weak orange dash / S1 blue",(40,80),cv2.FONT_HERSHEY_SIMPLEX,1.15,(255,255,255),3,cv2.LINE_AA)
  save_image(out/(name+"__f0_full_band_S1_overlay.jpg"),overlay)
  row=dict(case=name,frame_index=0,source_video=video,
    current_image_pixel_sha256=baseline["smoke_original_pixel_sha256"],
    full_record_sha256=sha(baseline_path),band_record_sha256=sha(band_path),
    full_D4_seconds=baseline["diagnostics"]["timings"]["current_D4_seconds"],
    band_D4_seconds=band["diagnostics"]["timings"]["current_D4_seconds"],
    band_total_seconds=runtime,
    full_kind=baseline.get("selected_kind"),band_kind=band.get("selected_kind"),
    full_H6_track_count=baseline["diagnostics"].get("H6_track_count"),
    band_H6_track_count=band["diagnostics"].get("H6_track_count"),
    full_whole=baseline["whole_path_candidate_supported"],band_whole=band["whole_path_candidate_supported"],
    band_reasons=band["reasons"],band_mask=band["diagnostics"]["current_D4_observation_mask"],
    full_support=baseline["diagnostics"].get("whole_path_support_checks"),
    band_support=band["diagnostics"].get("whole_path_support_checks"),
    geometry_comparison=geometry_compare(baseline,band),
    full_free_gaps=gaps(baseline,current_s1),
    S1_valid=current_s1["valid"],S1_status=current_s1["s1_status"],
    S1_endpoint_original_xy=current_s1["optical_endpoint_xy"],
    S1_supported_sample_count=len(current_s1["line_points_xy"]),
    S1_current_seed=baseline["seed_xy"],S1_record_sha256=sha(sp))
  report["cases"].append(row)
  (out/"comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
  print(json.dumps(row,ensure_ascii=False),flush=True)
if __name__=="__main__":main()

