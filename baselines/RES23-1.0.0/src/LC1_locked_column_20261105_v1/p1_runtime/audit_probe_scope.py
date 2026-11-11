"""Numerical audit only: separate supplied entry-condition extent from real gaps.

No image/model read, no original probe record rewritten. This checks the current
adapter gate against the immutable candidate coordinates/statuses from the probe.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lc1_p1_adapter import P1H6Adapter,_plain
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=ROOT/"p1_runtime"/"multiframe_probe"
start=json.loads((out/"run_start.json").read_text(encoding="utf-8"))
report=dict(
 scope="posthoc_gate_scope_audit_only_no_new_observations",
 original_probe_adapter_sha256=start["source_hashes"][str(ROOT/"lc1_p1_adapter.py")],
 current_adapter_sha256=sha(ROOT/"lc1_p1_adapter.py"),
 thresholds_changed=False,
 correction="exclude leading input entry_condition from evidence denominator and maximum gap; its rendering remains unsupported and it adds no observed length",
 original_frames_jsonl_sha256=sha(out/"frames.jsonl"),
 rows=[])
for line in (out/"frames.jsonl").read_text(encoding="utf-8").splitlines():
 row=json.loads(line)
 if "error" in row:continue
 p1=row["P1_H6"];d=p1["diagnostics"];old=d["whole_path_support_checks"]
 status=np.asarray(d["path_point_status"])
 candidate=dict(
  points_xy=((np.asarray(p1["path_xy"],float)-[0,1050])*.5).tolist(),
  supported=np.isin(status,["image_supported","joint_heat_image_supported"]).tolist(),
  stage=d["path_stage"],point_status=d["path_point_status"],
  kind=p1["selected_kind"],topology=d["H6_topology"],
  width_half_px=[old["maximum_allowed_unsupported_run_original_px"]/8.])
 ranked=d["ranked_candidates"]
 candidate["continuation_eligible"]=ranked[0].get("continuation_eligible")
 selected=p1["selected_entry"]
 entry=dict(seed=d["current_A7_seed"],
  connection=dict(score_margin=old["branch_score_margin"],ambiguous=old["ambiguous"]),
  task_direction=d["entry_summary"][selected]["task_direction"])
 raw_ranked=[(r["rank_score"],r["entry"],r["candidate"],{}) for r in ranked]
 instance=P1H6Adapter.__new__(P1H6Adapter);instance.pitch=start["source_metadata"][row["case"]]["pitch_px"]
 good,checks,reasons,segments,observed=instance._support(entry,candidate,raw_ranked)
 assert segments==d["path_segment_support"] and observed==d["path_point_support"],"Evidence changed unexpectedly"
 report["rows"].append(dict(case=row["case"],frame_index=row["frame_index"],
  original_whole_flag=p1["whole_path_candidate_supported"],scope_corrected_whole_flag=good,
  original_maximum_gap_px=old["maximum_unsupported_run_original_px"],
  input_condition_point_count_excluded=checks["input_condition_point_count_excluded"],
  maximum_actual_gap_excluding_input_condition_px=checks["maximum_unsupported_run_original_px"],
  whole_path_support_checks=checks,reasons=reasons,
  original_geometric_turn_valid=row["estimate_turn"]["valid"],
  scope_corrected_admissible_for_d_lock=bool(good and row["S1"]["valid"] and row["estimate_turn"]["valid"])))
report["whole_supported_count"]=sum(r["scope_corrected_whole_flag"] for r in report["rows"])
report["admissible_for_d_lock_count"]=sum(r["scope_corrected_admissible_for_d_lock"] for r in report["rows"])
(out/"support_scope_audit.json").write_text(json.dumps(_plain(report),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="rows"},ensure_ascii=False))
for row in report["rows"]:
 print(json.dumps({k:v for k,v in row.items() if k in ["case","frame_index","original_maximum_gap_px","maximum_actual_gap_excluding_input_condition_px","scope_corrected_whole_flag","reasons"]},ensure_ascii=False))

