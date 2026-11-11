"""Real-model development-only multi-frame probe; never authorizes missing evidence."""
from pathlib import Path
import hashlib,json,sys,time,traceback,platform
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lc1_p1_adapter import P1H6Adapter,_plain
from lc1_s1_adapter import S1Adapter
from lc1_geometry import estimate_turn
from smoke_current import DEV,ASSETS,MODEL,save_image,sha
from compare_entry_band import nearest_segment_distances,runs

def summarize_s1_coverage(p1,s1):
 p=np.asarray(p1["path_xy"],float).reshape(-1,2)
 diag=p1["diagnostics"]
 if not len(p):return {}
 stage=np.asarray(diag["path_stage"]);state=np.asarray(diag["path_point_status"])
 original_support=np.asarray(diag["path_point_support"],bool)
 dist=nearest_segment_distances(p,s1["supported_segments_xy"])
 free=stage=="free";counted=free&(state!="entry_condition")
 s1within=dist<=8.
 union=original_support|(s1within&free)
 gap_results=[]
 for a,b in runs((~original_support[:-1]|~original_support[1:])&free[:-1]&free[1:]):
  length=float(np.linalg.norm(np.diff(p[a:b+1],axis=0),axis=1).sum())
  if length<40 or np.all(state[a:b+1]=="entry_condition"):continue
  gap_results.append(dict(path_indices=[int(a),int(b)],stage="free",length_original_px=length,
   start_original_xy=p[a].tolist(),end_original_xy=p[b].tolist(),
   S1_covered_fraction_within_8px=float(s1within[a:b+1].mean()),
   S1_covered_fraction_within_12px=float((dist[a:b+1]<=12).mean())))
 gap_results.sort(key=lambda a:-a["length_original_px"])
 return dict(
  comparison_scope="actual_finite_S1_supported_polylines_only_no_line_extrapolation",
  free_current_S1_coverage_fraction_within_8px=float(s1within[counted].mean()) if counted.any() else None,
  free_current_S1_coverage_fraction_within_12px=float((dist[counted]<=12).mean()) if counted.any() else None,
  free_P1_observed_fraction_excluding_seed=float(original_support[counted].mean()) if counted.any() else None,
  free_union_S1_P1_coverage_fraction_excluding_seed=float(union[counted].mean()) if counted.any() else None,
  longest_free_P1_gaps=gap_results[:4],
  same_material_identity_confirmed=False,
  fused_whole_path_candidate_supported=False,
  fusion_gate_applied=False,
  fusion_note="Evidence coverage diagnostic only; no new complete gate or d license inferred")
def compact_p1(p1):
 d=p1["diagnostics"]
 keys=["surface_observation_mode","current_D4_observation_mask","timings","current_A7_seed",
       "H6_track_count","current_D4_curve_count","entry_summary","ranked_candidates",
       "path_segment_support","path_point_support","path_stage","path_point_status",
       "whole_path_support_checks","H6_candidate_metrics","H6_topology","H6_termination",
       "join_start_xy","join_end_xy"]
 return {**{k:v for k,v in p1.items() if k!="diagnostics"},"diagnostics":{k:d.get(k) for k in keys}}
def draw_overlay(bgr,p1,s1,turn,case,index):
 out=bgr.copy()
 for segment in s1["supported_segments_xy"]:
  p=np.rint(segment).astype(np.int32)
  if len(p)>1:cv2.polylines(out,[p],False,(255,150,0),6,cv2.LINE_AA)
 q=np.rint(p1["path_xy"]).astype(np.int32).reshape(-1,2)
 support=p1["diagnostics"]["path_segment_support"]
 stage=p1["diagnostics"].get("path_stage",[])
 for j in range(1,len(q)):
  if support[j-1]:
   color=(40,235,40) if stage[j]=="free" else (235,70,235)
   cv2.line(out,tuple(q[j-1]),tuple(q[j]),color,4,cv2.LINE_AA)
  elif j%5<2:cv2.line(out,tuple(q[j-1]),tuple(q[j]),(0,140,255),3,cv2.LINE_AA)
 seed=p1.get("seed_xy")
 if seed:
  a,b=np.rint(seed).astype(np.int32)
  cv2.arrowedLine(out,tuple(a),tuple(b),(255,255,255),6,cv2.LINE_AA,tipLength=.2)
 if turn.get("valid") and turn.get("x") is not None:
  x=float(turn["x"]);x=4095-x if case.startswith("left") else x
  # Diagnostic cross only, never a locked reference column.
  y=float(turn["y"]);pt=(int(round(x)),int(round(y)))
  cv2.drawMarker(out,pt,(0,200,255),cv2.MARKER_TILTED_CROSS,34,3)
 cv2.putText(out,f"{case} f{index} | entry_band real B750+H6 / S1 | whole={p1['whole_path_candidate_supported']}",(45,85),cv2.FONT_HERSHEY_SIMPLEX,1.45,(255,255,255),3,cv2.LINE_AA)
 cv2.putText(out,"green/purple: current supported; orange dash: weak; blue: S1 actual; cross: fit diagnostic only",(45,145),cv2.FONT_HERSHEY_SIMPLEX,1.15,(255,255,255),3,cv2.LINE_AA)
 return out
def main():
 cv2.setNumThreads(1)
 out=ROOT/"p1_runtime"/"multiframe_probe";out.mkdir(exist_ok=True)
 output=out/"frames.jsonl"
 if output.exists():raise FileExistsError("Keep existing probe immutable; choose a new directory to repeat")
 models={};caps={};meta={}
 s1=S1Adapter()
 for case,session,camera,video in DEV:
  stem=f"{session}__pair0015__{camera}"
  mp=ASSETS/"camera_condition_sam2_20261009_v1"/"all_001"/"body_branches_002"/"masks"/f"{camera}_camera"/f"{stem}.png"
  pp=ASSETS/"camera_condition_bright_rope_boundaries_20261023_v2"/"all_002"/"boundaries"/"records"/f"{camera}_camera"/f"{stem}.json"
  mask=cv2.imdecode(np.fromfile(mp,np.uint8),cv2.IMREAD_GRAYSCALE)
  pitch=json.loads(pp.read_text(encoding="utf-8"))["record"]["pitch_px"]
  models[case]=P1H6Adapter(mask,pitch,MODEL,threads=2,camera=camera,surface_mode="entry_band")
  caps[case]=cv2.VideoCapture(video)
  assert caps[case].isOpened()
  meta[case]=dict(camera=camera,source_video=video,pitch_px=pitch,surface_mask_sha256=sha(mp),pitch_record_sha256=sha(pp),
                 source_fps=caps[case].get(cv2.CAP_PROP_FPS),source_frame_count=int(caps[case].get(cv2.CAP_PROP_FRAME_COUNT)))
 plan=[("left111500",30),("right113305",300),
       ("left111500",0),("left111500",15),("left111500",45),("left111500",60),("left111500",75),
       ("right113305",240),("right113305",270),("right113305",330),("right113305",360),("right113305",390)]
 receipt=dict(python=sys.executable,platform=platform.platform(),model_sha256=sha(MODEL),
  source_hashes={str(p):sha(p) for p in [ROOT/"lc1_p1_adapter.py",ROOT/"lc1_s1_adapter.py",ROOT/"lc1_geometry.py",Path(__file__)]},
  plan=[dict(case=c,frame_index=i) for c,i in plan],
  surface_mode="entry_band_explicit_8pitch_plus2pitch_halo_different_from_full",
  current_image_observations_only=True,historical_target_cache_read=False,source_metadata=meta,rows=[])
 (out/"run_start.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
 with output.open("x",encoding="utf-8") as handle:
  for case,index in plan:
   start=time.perf_counter()
   try:
    cap=caps[case];cap.set(cv2.CAP_PROP_POS_FRAMES,index)
    before=cap.get(cv2.CAP_PROP_POS_FRAMES);ok,bgr=cap.read();after=cap.get(cv2.CAP_PROP_POS_FRAMES)
    assert ok and bgr.shape==(3000,4096,3) and abs(before-index)<.01 and abs(after-index-1)<.01
    p1=models[case].step(bgr)
    tick=time.perf_counter();current_s1=s1.step(bgr,p1["seed_xy"],seed_source=p1.get("seed_source") or "current_frame_P1_A7_missing");s1sec=time.perf_counter()-tick
    path=np.asarray(p1["path_xy"],float).reshape(-1,2)
    canonical=path.copy()
    if meta[case]["camera"]=="left":canonical[:,0]=4095-canonical[:,0]
    turn=estimate_turn(canonical,free_side="right",pitch_px=meta[case]["pitch_px"]) if len(path)>=24 else dict(valid=False,reason="no_candidate_path")
    turn=_plain(turn)
    turn.update(coordinate_system="canonical_right_entering_xy",diagnostic_candidate_fit_only=True,
                admissible_for_d_lock=bool(p1["whole_path_candidate_supported"] and current_s1["valid"] and turn.get("valid")),
                S1_optical_endpoint_used_as_d=False)
    coverage=summarize_s1_coverage(p1,current_s1)
    row=dict(case=case,frame_index=index,time_s=index/meta[case]["source_fps"],
      source_video=meta[case]["source_video"],decoded_frame_index_before=before,decoded_frame_index_after=after,
      original_pixel_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),image_coordinate_system="original_unmirrored_xy",
      surface_mode="entry_band_explicit_8pitch_plus2pitch_halo",P1_H6=compact_p1(p1),
      S1={k:v for k,v in current_s1.items() if k!="diagnostics"},
      S1_diagnostics=current_s1["diagnostics"],S1_coverage=coverage,estimate_turn=turn,
      S1_seconds=s1sec,total_seconds=time.perf_counter()-start)
    dest=out/(case+f"__f{index:04d}_overlay.jpg")
    save_image(dest,draw_overlay(bgr,p1,current_s1,turn,case,index))
    row["overlay_file"]=str(dest);row["overlay_sha256"]=sha(dest)
    handle.write(json.dumps(_plain(row),ensure_ascii=False,allow_nan=False)+"\n");handle.flush()
    summary=dict(case=case,frame_index=index,whole_path_candidate_supported=p1["whole_path_candidate_supported"],
      P1_status=p1["status"],selected_kind=p1.get("selected_kind"),point_count=len(path),
      maximum_unsupported_run_original_px=p1["diagnostics"].get("whole_path_support_checks",{}).get("maximum_unsupported_run_original_px"),
      S1_valid=current_s1["valid"],S1_endpoint_original_xy=current_s1["optical_endpoint_xy"],
      geometric_turn_valid=turn["valid"],geometric_turn_x=turn.get("x"),
      turn_fit_reason=turn.get("reason"),admissible_for_d_lock=turn["admissible_for_d_lock"],
      reasons=p1["reasons"],total_seconds=row["total_seconds"])
   except Exception:
    summary=dict(case=case,frame_index=index,error=traceback.format_exc())
    handle.write(json.dumps(summary,ensure_ascii=False)+"\n");handle.flush()
   receipt["rows"].append(summary)
   (out/"receipt.json").write_text(json.dumps(_plain(receipt),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
   print(json.dumps(_plain(summary),ensure_ascii=False),flush=True)
 for cap in caps.values():cap.release()
 receipt.update(runtime_completed=True,whole_supported_frame_count=sum(r.get("whole_path_candidate_supported",False) for r in receipt["rows"]),
                admissible_turn_count=sum(r.get("admissible_for_d_lock",False) for r in receipt["rows"]),
                failures=sum("error" in r for r in receipt["rows"]),frames_jsonl_sha256=sha(output))
 (out/"receipt.json").write_text(json.dumps(_plain(receipt),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
if __name__=="__main__":main()

