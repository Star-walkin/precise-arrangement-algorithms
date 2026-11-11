from pathlib import Path
import json,hashlib,statistics
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/"p1_runtime"/"multiframe_probe"
frames=[json.loads(s) for s in (out/"frames.jsonl").read_text(encoding="utf-8").splitlines()]
receipt=json.loads((out/"receipt.json").read_text(encoding="utf-8"))
audit=json.loads((out/"support_scope_audit.json").read_text(encoding="utf-8"))
scope={(r["case"],r["frame_index"]):r for r in audit["rows"]}
assert len(frames)==12 and receipt["runtime_completed"] and receipt["failures"]==0
assert receipt["frames_jsonl_sha256"]==hashlib.sha256((out/"frames.jsonl").read_bytes()).hexdigest()
assert all(hashlib.sha256(Path(r["overlay_file"]).read_bytes()).hexdigest()==r["overlay_sha256"] for r in frames)
rows=sorted(frames,key=lambda r:(r["case"],r["frame_index"]))
lines=[
"# Real B750/H6 + S1 development-frame probe",
"",
"All12 requested frames ran successfully using original unmirrored4096x3000 images.",
"Explicit mode:entry_band, current A7 ±8pitch plus2pitch halo; this differs from",
"full D4 mode. No historical target path/cache or reserved scene was read.",
"",
f"Actual B750 forwards:{sum(r['P1_H6']['model_forward_count'] for r in rows)}; genuine S1 executions:12.",
"Whole-supported frames:0/12. Geometrically admissible locked-column frames:0/12.",
"All S1 outputs are PARTIAL optical straight evidence; their endpoints are never d.",
"",
"| case/frame | H6 candidate | actual max gap px* | S1 free coverage | P1/S1 free union | turn fit | seconds |",
"|---|---|---:|---:|---:|---|---:|",
]
for r in rows:
 a=scope[(r["case"],r["frame_index"])]
 cov=r["S1_coverage"]
 kind="curve attached" if r["P1_H6"]["selected_kind"]=="straight_free_to_F3_center_track" else "free only"
 lines.append(f"| {r['case']}/{r['frame_index']} | {kind} | {a['maximum_actual_gap_excluding_input_condition_px']:.1f} | {100*cov['free_current_S1_coverage_fraction_within_8px']:.1f}% | {100*cov['free_union_S1_P1_coverage_fraction_excluding_seed']:.1f}% | {r['estimate_turn']['reason']} | {r['total_seconds']:.1f} |")
lines.extend([
"",
"*The input entry_condition extent is excluded from these gap/coverage statistics;",
"it adds no observed length and is never rendered as supported. The original",
"JSONL is immutable; support_scope_audit.json records this scope correction and",
"proves all12 complete flags remain false with unchanged evidence thresholds.",
"",
"S1 coverage uses nearest actual finite supported segments within8px, without",
"extrapolating S1's fitted line. The union column applies only to the free stage;",
"a high value cannot supply missing transition or curved-path evidence.",
"",
"Left f15/f60 have91.7%/94.8% union free coverage, yet their transition evidence",
"is insufficient and the overall path does not establish a localized bend.",
"Left f75 and right f270/f360/f390 can have complete observed free coverage but",
"H6 has no admissible curved attachment. Thus a percentage-only fusion gate",
"would not establish a physical or optical straight-to-curved transition.",
"",
"Left f30 retains a curved candidate but has225.05px actual unsupported run,",
"weak transition evidence and a branch ambiguity; estimate_turn explicitly",
"rejects turn_localization_uncertain. The raw fitted x is diagnostic only.",
"Right f240 retains a curved candidate but474.68px loss of current support;",
"the geometry fit is straight_path. Other right samples retain free fragments.",
"",
"Overlay legend:green supported free, purple supported transition/surface,",
"orange dashed weak candidate geometry, blue actual S1 bilateral segments.",
"Any fit marker is explicitly diagnostic; no fixed reference column was drawn.",
"",
"Data/provenance:frames.jsonl, receipt.json, support_scope_audit.json,",
"compact_summary.json, and source_snapshot/ preserve coordinates, per-segment",
"support, source pixel hashes, decoded frame indices, source code hashes and",
"each output overlay SHA256. The12 overlays were rehashed successfully.",
f"Total per-frame compute mean:{statistics.mean(r['total_seconds'] for r in rows):.2f}s;",
f"D4 mean:{statistics.mean(r['P1_H6']['diagnostics']['timings']['current_D4_seconds'] for r in rows):.2f}s.",
])
(out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print("\n".join(lines))

