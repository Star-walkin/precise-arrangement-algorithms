"""只读比对 D2 空模型修复前后输出；仅写 all_002/repair_equivalence.json。

不重新运行检测，也不解码原始视频。仅在原失败图读取 D1 已冻结的
暗 ROI 和增强 ROI，用于验证空预测时预览保留原有底图与轮廓。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
OLD = ROOT / "all_001"
NEW = ROOT / "all_002"
PILOT = ROOT / "pilot_006"
D1 = ROOT.parent / "camera_condition_dark_roi_20261026_v1" / "all_001"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_image(path: Path, flags: int) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def related_paths(run: Path, rel: str) -> tuple[list[Path], Path]:
    name = Path(rel)
    role = name.parent.name
    png = name.name
    stem = name.stem
    images = [run / "图像结果" / role / png]
    images.extend(run / "记录" / "masks" / mask / role / png
                  for mask in ("observed", "smooth", "inferred"))
    case = run / "记录" / "cases" / role / f"{stem}.json"
    return images, case


def case_without_version(value: dict) -> dict:
    result = dict(value)
    record = dict(result.get("record", {}))
    record.pop("version", None)
    result["record"] = record
    return result


def compare_predictions(reference: Path, candidate: Path,
                        images: list[str], *, compare_whole_case: bool) -> dict:
    checked = 0
    png_equal = 0
    case_equal = 0
    observations_equal = 0
    fit_equal = 0
    mismatches: list[dict] = []
    for rel in images:
        checked += 1
        old_pngs, old_case = related_paths(reference, rel)
        new_pngs, new_case = related_paths(candidate, rel)
        files_ok = True
        for old_path, new_path in zip(old_pngs, new_pngs):
            if not old_path.is_file() or not new_path.is_file():
                files_ok = False
                mismatches.append({"image": rel, "item": old_path.name,
                                   "reason": "missing_png"})
            elif digest(old_path) != digest(new_path):
                files_ok = False
                mismatches.append({"image": rel, "item": old_path.name,
                                   "reason": "png_sha256_differs"})
        if files_ok:
            png_equal += 1
        if not old_case.is_file() or not new_case.is_file():
            mismatches.append({"image": rel, "item": "case", "reason": "missing_json"})
            continue
        old_value, new_value = read_json(old_case), read_json(new_case)
        if old_value.get("observed_lines") == new_value.get("observed_lines"):
            observations_equal += 1
        else:
            mismatches.append({"image": rel, "item": "observed_lines",
                               "reason": "json_differs"})
        if old_value.get("fit") == new_value.get("fit"):
            fit_equal += 1
        else:
            mismatches.append({"image": rel, "item": "fit", "reason": "json_differs"})
        if compare_whole_case:
            if case_without_version(old_value) == case_without_version(new_value):
                case_equal += 1
            else:
                mismatches.append({"image": rel, "item": "case_except_record_version",
                                   "reason": "json_differs"})
    return {"cases_checked": checked, "four_pngs_identical_cases": png_equal,
            "observed_lines_identical_cases": observations_equal,
            "fit_identical_cases": fit_equal,
            "case_except_record_version_identical_cases": case_equal
            if compare_whole_case else None,
            "mismatch_count": len(mismatches), "mismatches": mismatches}


def check_repaired_failure(rel: str, old_case: dict) -> dict:
    pngs, case_path = related_paths(NEW, rel)
    result = {"image": rel, "old_status": old_case.get("record", {}).get("status"),
              "old_error": old_case.get("record", {}).get("error"),
              "new_case_exists": case_path.is_file(), "pngs_exist": all(p.is_file() for p in pngs)}
    if not result["new_case_exists"] or not result["pngs_exist"]:
        return result
    new_case = read_json(case_path)
    result["new_status"] = new_case.get("record", {}).get("status")
    result["observed_lines_empty"] = new_case.get("observed_lines") == []
    result["curves_empty"] = new_case.get("fit", {}).get("curves") == []
    result["masks_all_zero"] = {}
    for mask, path in zip(("observed", "smooth", "inferred"), pngs[1:]):
        result["masks_all_zero"][mask] = not np.any(read_image(path, cv2.IMREAD_GRAYSCALE))

    # 只读原失败图既有 D1 派生图，不访问原始采集或保留集。
    name = Path(rel)
    role, png = name.parent.name, name.name
    roi = read_image(D1 / "记录" / "masks" / "dark_roi" / role / png,
                     cv2.IMREAD_GRAYSCALE)
    base = read_image(D1 / "图像结果" / "02_对比度增强ROI" / role / png,
                      cv2.IMREAD_COLOR)
    expected = base.copy()
    contours = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(expected, contours, -1, (0, 135, 0), 1)
    expected[roi == 0] = 0
    overlay = read_image(pngs[0], cv2.IMREAD_COLOR)
    result["overlay_matches_empty_prediction_roi_render"] = bool(
        np.array_equal(overlay, expected))
    result["roi_pixel_count"] = int(np.count_nonzero(roi))
    result["overlay_nonzero_inside_roi"] = bool(np.any(overlay[roi > 0]))
    result["overlay_zero_outside_roi"] = not bool(np.any(overlay[roi == 0]))
    return result


def main() -> None:
    if not (NEW / "summary.json").is_file():
        raise SystemExit("all_002/summary.json not present; wait for the completed run")
    old_summary, new_summary = read_json(OLD / "summary.json"), read_json(NEW / "summary.json")
    old_context, new_context = read_json(OLD / "run_context.json"), read_json(NEW / "run_context.json")
    pilot_context = read_json(PILOT / "run_context.json")
    if old_context["selection"] != new_context["selection"]:
        raise ValueError("all_001 and all_002 input selections differ")
    selection = old_context["selection"]
    successful, failed = [], []
    for rel in selection:
        _, case_path = related_paths(OLD, rel)
        value = read_json(case_path)
        if value.get("record", {}).get("status") == "completed":
            successful.append(rel)
        else:
            failed.append((rel, value))
    old_to_new = compare_predictions(OLD, NEW, successful, compare_whole_case=True)
    pilot_to_new = compare_predictions(PILOT, NEW, pilot_context["selection"],
                                       compare_whole_case=True)
    repaired = [check_repaired_failure(rel, value) for rel, value in failed]
    repair_passed = (len(repaired) == 1 and
                     repaired[0].get("new_status") == "completed" and
                     repaired[0].get("observed_lines_empty") is True and
                     repaired[0].get("curves_empty") is True and
                     all(repaired[0].get("masks_all_zero", {}).values()) and
                     repaired[0].get("overlay_matches_empty_prediction_roi_render") is True and
                     repaired[0].get("overlay_nonzero_inside_roi") is True and
                     repaired[0].get("overlay_zero_outside_roi") is True)
    count_passed = (len(selection) == 400 and len(successful) == 399 and len(failed) == 1 and
                    old_summary.get("completed") == 399 and new_summary.get("completed") == 400 and
                    new_summary.get("failed") == 0)
    predictions_passed = (old_to_new["cases_checked"] == 399 and
                          old_to_new["four_pngs_identical_cases"] == 399 and
                          old_to_new["observed_lines_identical_cases"] == 399 and
                          old_to_new["fit_identical_cases"] == 399 and
                          old_to_new["case_except_record_version_identical_cases"] == 399 and
                          old_to_new["mismatch_count"] == 0 and
                          pilot_to_new["cases_checked"] == 12 and
                          pilot_to_new["four_pngs_identical_cases"] == 12 and
                          pilot_to_new["observed_lines_identical_cases"] == 12 and
                          pilot_to_new["fit_identical_cases"] == 12 and
                          pilot_to_new["case_except_record_version_identical_cases"] == 12 and
                          pilot_to_new["mismatch_count"] == 0)
    report = {"check": "DR-D2.1.1 empty-C3-model repair equivalence",
              "status": "passed" if count_passed and predictions_passed and repair_passed
              else "needs_review",
              "scope": "Existing open-development outputs only; no algorithm rerun, original video decode, reserved set, or device access",
              "all_001_completed": old_summary.get("completed"),
              "all_002_completed": new_summary.get("completed"),
              "successful_399_equivalence": old_to_new,
              "pilot_006_12_equivalence": pilot_to_new,
              "repaired_failure": repaired,
              "checks": {"expected_case_counts": count_passed,
                         "predictions_unchanged": predictions_passed,
                         "single_empty_case_repaired": repair_passed},
              "interpretation": "Byte-identical output validates this guard repair only; visual boundary accuracy and physical identity remain unverified"}
    output = NEW / "repair_equivalence.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"],
                      "successful": len(successful), "failed_old": len(failed),
                      "pilot_cases": pilot_to_new["cases_checked"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
