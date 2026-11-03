"""Read-only post-evaluation diagnosis of the fixed eight V28 images.

Reads saved candidates and evaluation references; never regenerates candidates,
changes scores or parameters. The reference-profile measurements below are
explicit posthoc observability diagnostics, not AUTO inputs or new accuracy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from vision_geometry_lab.experiments.v28_local_spool_connection import data_eval, pipeline


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def distances_to_polyline(query, polyline):
    query, line = np.asarray(query, float), np.asarray(polyline, float)
    best = np.full(len(query), np.inf)
    for a, b in zip(line[:-1], line[1:]):
        delta = b - a
        fraction = np.clip((query - a) @ delta / max(float(delta @ delta), 1e-10), 0., 1.)
        best = np.minimum(best, np.linalg.norm(query - a - fraction[:, None] * delta, axis=1))
    return best


def quantiles(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return None if not len(v) else dict(min=float(np.min(v)), median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(np.max(v)))


def fraction(values, mask=None):
    v = np.asarray(values, bool)
    if mask is not None:
        v = v[mask]
    return float(v.mean()) if len(v) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    evaluation = read(run / "evaluation.json")
    inputs = read(run / "input_manifest.json")
    assert len(inputs) == 8 and {x["sample_id"] for x in inputs} == set(data_eval.SAMPLE_IDS)
    images = {}
    references = {}
    entries = []
    for result in evaluation["rows"]:
        sid, mode = result["sample_id"], result["mode"]
        assert sid in data_eval.SAMPLE_IDS and mode in {"AUTO", "CONDITIONAL"}
        folder = run / mode / sid
        entry, pool, structure = [read(folder / (name + ".json")) for name in ("entry", "pool", "structure")]
        if sid not in references:
            references[sid] = data_eval.load_evaluation_reference(sid)
            item = next(x for x in inputs if x["sample_id"] == sid)
            assert data_eval.sha256(item["image_path"]) == item["image_sha256"]
            images[sid] = cv2.imdecode(np.fromfile(item["image_path"], np.uint8), cv2.IMREAD_COLOR)
        reference = references[sid]
        gt_seed = np.asarray(reference["free_seed"], float)
        gt_d = pipeline.unit(gt_seed[-1] - gt_seed[0])
        gt_n = np.array([-gt_d[1], gt_d[0]])
        gt_bend = np.vstack([pipeline.resample(piece, 2.) for piece in result["trusted_bend_segments"]])
        departure = np.asarray(result["GT_departure"]["point_xy"], float)
        reference_evidence = pipeline.profile_evidence(images[sid], gt_bend, reference["width_px"])
        diag = dict(sample_id=sid, mode=mode, entry_available=entry["available"], entry_reason=entry["reason"],
                    pool_contains_trusted_bend=result["candidate_pool_contains_trusted_bend_geometry"],
                    best_pool_bend_coverage_5px=result["best_pool_bend_coverage_5px"],
                    visible_connection_evaluable=result["visible_connection_evaluable"],
                    physical_identity_confirmed=False,
                    reference_evidence_posthoc=dict(
                        source="V28 profile_evidence on trusted bend only AFTER prediction/evaluation; not AUTO input",
                        available_fraction=fraction(reference_evidence["available"]),
                        bilateral_fraction=fraction(reference_evidence["supported"]),
                        contrast_gray=quantiles(reference_evidence["contrast"]),
                        saturated_fraction=quantiles(reference_evidence["saturated_fraction"]),
                        caveat="Available profiles need not establish unique material or neighbor identity"),
                    candidates=[], structures=[], failure_layers=[])
        if not entry["available"]:
            diag["failure_layers"].append("自动入口不可用，未生成候选；原始分母保留")
            diag["rope_reason"] = entry["rope"].get("reason")
            diag["arc_reason"] = entry["arc"].get("reason")
            entries.append(diag)
            continue
        seed = np.asarray(entry["free_seed"], float)
        origin = seed[-1]
        d = pipeline.unit(seed[-1] - seed[0])
        n = np.array([-d[1], d[0]])
        angle = float(np.degrees(np.arccos(np.clip(d @ gt_d, -1., 1.))))
        width = float(pool["width_px"])
        guv = np.column_stack(((gt_bend - origin) @ d, (gt_bend - origin) @ n))
        duv = [(departure - origin) @ d, (departure - origin) @ n]
        cfg = read(run / "config.json")["structure"]
        maxu = min(cfg["search_forward_widths"] * width, cfg["maximum_forward_px"])
        in_search = ((guv[:, 0] >= -cfg["search_backward_widths"] * width) &
                     (guv[:, 0] <= maxu) & (abs(guv[:, 1]) <= cfg["search_half_widths"] * width))
        diag["input_geometry"] = dict(
            angle_error_to_existing_free_seed_deg=angle,
            entry_distance_to_GT_departure_px=float(np.linalg.norm(departure - origin)),
            entry_upstream_projected_distance_px=float((departure - origin) @ gt_d),
            entry_normal_offset_from_GT_free_line_px=float((origin - gt_seed[-1]) @ gt_n),
            GT_departure_u_v_px=[float(z) for z in duv],
            GT_bend_u_range_px=[float(guv[:, 0].min()), float(guv[:, 0].max())],
            GT_bend_v_range_px=[float(guv[:, 1].min()), float(guv[:, 1].max())],
            GT_bend_in_declared_search_fraction=float(in_search.mean()),
            forward_search_px=maxu, input_width_px=width, independent_reference_width_px=reference["width_px"],
            measured_width_ratio_to_independent_reference=float(width / reference["width_px"]),
            local_structure_focus_u_px=structure["diagnostics"].get("focus_u_px"),
            local_structure_focus_source=structure["diagnostics"].get("focus_source"))
        for s in structure["structures"]:
            p = np.asarray(s["points_xy"], float)
            u = (p - origin) @ d
            measured = [value / width for value in s["widths_px"] if value is not None]
            diag["structures"].append(dict(
                id=s["id"], polarity=s["polarity"],
                u_range_px=[float(u.min()), float(u.max())],
                distance_from_trusted_bend_px=quantiles(distances_to_polyline(gt_bend, p)),
                paired_refinement_fraction=s["paired_refinement_fraction"],
                paired_candidate_fraction=s["paired_candidate_fraction"],
                width_ratio_to_free_span=quantiles(measured),
                confirmed_strand_or_neighbor_identity=False))
        for c in pool["candidates"]:
            p = np.asarray(c["points_xy"], float)
            u = (p - origin) @ d
            key = (u >= guv[:, 0].min()) & (u <= guv[:, 0].max())
            refdist = distances_to_polyline(gt_bend, p)
            ce = next(x for x in result["candidate_scores"] if x["id"] == c["id"])
            diag["candidates"].append(dict(
                id=c["id"], kind=c["kind"], structure_id=c["structure_id"],
                bend_coverage_5px=ce["bend_geometry_5px"]["covered_fraction"],
                distance_from_trusted_bend_px=quantiles(refdist),
                u_range_px=[float(u.min()), float(u.max())],
                terminal_xy=p[-1].tolist(), terminal_to_departure_px=float(np.linalg.norm(p[-1] - departure)),
                missing_forward_extent_to_bend_start_px=float(max(0., guv[:, 0].min() - u.max())),
                key_longitudinal_region_points=int(key.sum()),
                key_region_available_fraction=fraction(c["evidence"]["available"], key),
                key_region_bilateral_fraction=fraction(c["supported"], key),
                key_region_saturation=quantiles(np.asarray(c["evidence"]["saturated_fraction"])[key]),
                inferred_prefix_px=c["inferred_prefix_px"],
                visible_connection_status=ce["visible_connection_status"],
                neighbor_rope_ID_not_annotated=True))
        if in_search.mean() < .8:
            diag["failure_layers"].append("可信弯段未充分落入声明搜索范围，候选范围本身不足")
        else:
            diag["failure_layers"].append("可信弯段在声明搜索范围内；失败并非90w/侧向ROI直接排除目标")
        structs = diag["structures"]
        if structs and max(x["u_range_px"][1] for x in structs) < guv[:, 0].min():
            diag["failure_layers"].append("局部结构均提前结束于可信弯段上游；自动法兰焦点/短弧预算未把卷面弯段纳入")
        elif structs and min(x["distance_from_trusted_bend_px"]["median"] for x in structs) > reference["width_px"]:
            diag["failure_layers"].append("局部结构虽可到达相同纵向范围，但全部距可信弯段中位距离超过一倍参考宽度")
        if result["best_pool_bend_coverage_5px"] is not None and result["best_pool_bend_coverage_5px"] < .8:
            diag["failure_layers"].append("固定候选池没有覆盖80%可信关键弯段；排序器不能选出池中不存在的合格路径")
        crossing = [c for c in diag["candidates"] if c["kind"] == "new_segment_may_cross_old_rows"]
        if crossing and all(c["missing_forward_extent_to_bend_start_px"] > 0 for c in crossing):
            diag["failure_layers"].append("跨行新段候选在关键弯段之前停止")
        elif crossing and max(c["bend_coverage_5px"] for c in crossing) < .8:
            diag["failure_layers"].append("跨行候选到达关键纵向范围，但没有覆盖可信弯段；允许跨行尚未等于识别正确")
        if diag["reference_evidence_posthoc"]["available_fraction"] < .35:
            diag["failure_layers"].append("GT弯段的原图剖面可用性也低，存在证据不足；不能归为确定负证据")
        else:
            diag["failure_layers"].append("GT弯段并非整体被暗/饱和遮蔽；局部可用剖面不证明相邻路径能够唯一消歧")
        diag["failure_layers"].append("当前参考未标相邻绳ID；不能将偏离目标或暗相/窄宽候选直接命名为错邻绳或股纹")
        entries.append(diag)
    out = dict(schema="v28_posthoc_diagnostic_1", run=str(run),
               evaluation_sha256=data_eval.sha256(run / "evaluation.json"),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               scope="same eight open images, saved AUTO/CONDITIONAL pools; no rescoring/regeneration/tuning",
               notes=["GT departure is posthoc 2D departure from free line, not a physical contact point.",
                      "Longitudinal key-region fractions describe candidate evidence where GT u overlaps, not identity.",
                      "Profile evidence at GT is a posthoc observability diagnostic only."], entries=entries)
    with (run / "diagnostic_001.json").open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ["# V28 run001 事后失败诊断", "", "仅复查同8张开放图的已保存池与可信单帧参考；不生成新候选、不改权重、不重跑评分。", "",
             "所有可用输入的候选池均未达到80%可信关键弯段覆盖；因此本轮首先暴露候选生成/局部结构定位瓶颈，不能用排序变化证明关键身份改善。GT剖面可用性是事后诊断，不进入AUTO。", ""]
    for sid in data_eval.SAMPLE_IDS:
        lines += ["## " + sid, ""]
        for q in [x for x in entries if x["sample_id"] == sid]:
            if not q["entry_available"]:
                lines += [f"**{q['mode']}**：入口不可用，原因 `{q['entry_reason']}`；未生成候选。", ""]
                continue
            g = q["input_geometry"]
            best = max(q["candidates"], key=lambda c: c["bend_coverage_5px"])
            lines += [f"**{q['mode']}**：自由线方向相对已有可信自由段偏差 {g['angle_error_to_existing_free_seed_deg']:.2f}°，可见入口距事后弯段起点 {g['entry_distance_to_GT_departure_px']:.0f}px；可信弯段在声明搜索区内的比例 {g['GT_bend_in_declared_search_fraction']:.0%}。结构焦点 u={g['local_structure_focus_u_px']:.0f}px，弯段 u={g['GT_bend_u_range_px'][0]:.0f}–{g['GT_bend_u_range_px'][1]:.0f}px。",
                      f"池中最好覆盖为 {best['id']}：{best['bend_coverage_5px']:.1%}；其对可信弯段距离中位 {best['distance_from_trusted_bend_px']['median']:.1f}px / P95 {best['distance_from_trusted_bend_px']['p95']:.1f}px。GT弯段原图剖面可用 {q['reference_evidence_posthoc']['available_fraction']:.0%}，双边支持 {q['reference_evidence_posthoc']['bilateral_fraction']:.0%}。", ""]
            lines += ["- " + reason for reason in q["failure_layers"]]
            lines += [""]
    lines += ["## 解释边界", "", "没有相邻绳ID标注，因此只能判定未覆盖可信弯段、偏离目标带或证据不足。亮/暗多相及较窄光学双边可能对应绳体、间隙或股纹，本轮不能仅凭这些量确定其身份。", ""]
    with (run / "diagnostic_001.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(json.dumps(dict(entries=len(entries), fixed_images=len(inputs), output=str(run / "diagnostic_001.md")), ensure_ascii=False))


if __name__ == "__main__":
    main()
