"""独立审计 v3 文件与像素契约；不导入分割算法或运行设备。

用法：python audit_v3.py --run <既有 v3 运行目录> --output <新的 JSON 路径>
审计读取固定开放开发图以检查原图哈希和 RGB 保真；不会复制或重算模型输出。
文件/像素完整性通过，不等于逐根钢丝绳边界准确率或材料身份验收。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
DATASET = ASSETS / "camera_condition_images_20261009_v2"
V1 = ASSETS / "camera_condition_bright_rope_boundaries_20261009_v1" / "all_001"
MANIFEST = DATASET / "manifest.csv"
QA_PLAN = V1.parent / "qa_plan.csv"
PROTOCOL = HERE / "protocol.json"


def sha256(path: Path) -> str:
    """按字节流核验，避免解码或改写素材。"""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_image(path: Path, flag: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if image is None:
        raise ValueError(f"cannot decode {path}")
    return image


def binary_errors(name: str, mask: np.ndarray, shape: tuple[int, int]) -> list[str]:
    errors = []
    if mask.shape != shape or mask.dtype != np.uint8:
        errors.append(f"{name}: wrong shape or dtype")
    elif np.any((mask != 0) & (mask != 255)):
        errors.append(f"{name}: values outside 0/255")
    return errors


def safe_points(lines: list[dict], field: str, mask: np.ndarray) -> tuple[int, list[str]]:
    """逐点核对 JSON 坐标与掩码中心；避免负索引误通过。"""
    count = 0
    errors = []
    h, w = mask.shape
    for line_index, line in enumerate(lines):
        values = line.get(field, [])
        if not isinstance(values, list):
            errors.append(f"line {line_index}: {field} is not a list")
            continue
        for point_index, point in enumerate(values):
            count += 1
            if (not isinstance(point, (list, tuple)) or len(point) != 2 or
                    any(isinstance(v, bool) or not isinstance(v, int) for v in point)):
                errors.append(f"line {line_index} point {point_index}: invalid {field} coordinate")
                continue
            x, y = point
            if not (0 <= x < w and 0 <= y < h and mask[y, x] == 255):
                errors.append(f"line {line_index} point {point_index}: {field} absent from raster")
        if len(errors) >= 30:
            errors.append(f"{field}: further point errors omitted after 30")
            break
    return count, errors


def expected_paths(run: Path, row: dict, qa: set[str]) -> dict[str, Path]:
    rel = Path(row["output_file"])
    role, name = rel.parent.name, rel.name
    paths = {
        "reference_roi": run / "roi" / "reference_masks" / role / name,
        "effective_roi": run / "roi" / "effective_masks" / role / name,
        "effective_rgb": run / "roi" / "effective_masked" / role / name,
        "observed": run / "boundaries" / "observed" / role / name,
        "inferred": run / "boundaries" / "inferred" / role / name,
        "supported_rope": run / "boundaries" / "incoming_rope_occlusion" / role / name,
        "unknown": run / "boundaries" / "unknown_exclusion" / role / name,
        "inferred_under_occlusion": run / "boundaries" / "inferred_under_occlusion" / role / name,
        "record": run / "boundaries" / "records" / role / name.replace(".png", ".json"),
        "overlay": run / "overlays" / role / name.replace(".png", ".jpg"),
    }
    if row["output_file"] in qa:
        paths["qa_comparison"] = run / "qa_compare" / role / name.replace(".png", ".jpg")
    return paths


def audit_one(run: Path, row: dict, qa: set[str]) -> tuple[dict, list[dict]]:
    """一张图出错仍返回失败记录，使后续所有输入继续接受审计。"""
    errors: list[str] = []
    file_rows: list[dict] = []
    rel = Path(row["output_file"])
    role, name = rel.parent.name, rel.name
    paths = expected_paths(run, row, qa)
    for kind, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty {kind}: {path}")
            file_rows.append({"kind": kind, "path": str(path), "status": "missing"})
        else:
            file_rows.append({"kind": kind, "path": str(path), "status": "present",
                              "bytes": path.stat().st_size, "sha256": sha256(path)})
    source_path = DATASET / rel
    if not source_path.is_file():
        errors.append(f"missing source image: {source_path}")
        source = None
    else:
        source_hash = sha256(source_path)
        if source_hash != row["output_sha256"]:
            errors.append("source image SHA-256 differs from fixed manifest")
        source = read_image(source_path, cv2.IMREAD_COLOR)
        if source.shape != (int(row["height"]), int(row["width"]), 3):
            errors.append("source image dimensions differ from fixed manifest")
    old_roi_path = V1 / "roi" / "masks" / role / name
    if not old_roi_path.is_file():
        errors.append(f"missing authoritative v1 reference ROI: {old_roi_path}")
    elif paths["reference_roi"].is_file():
        if sha256(paths["reference_roi"]) != sha256(old_roi_path):
            errors.append("reference ROI bytes differ from v1")
    # 缺文件时保留全部缺失清单；不让一张图阻断其它图。
    if errors and any("missing" in error or "empty" in error for error in errors):
        finding = {"output_file": row["output_file"], "camera_role": row["camera_role"],
                   "source_group": row["source_group"], "status": "fail", "errors": errors}
        return finding, file_rows
    try:
        reference = read_image(paths["reference_roi"], cv2.IMREAD_GRAYSCALE)
        effective = read_image(paths["effective_roi"], cv2.IMREAD_GRAYSCALE)
        rgb = read_image(paths["effective_rgb"], cv2.IMREAD_COLOR)
        observed = read_image(paths["observed"], cv2.IMREAD_GRAYSCALE)
        inferred = read_image(paths["inferred"], cv2.IMREAD_GRAYSCALE)
        rope = read_image(paths["supported_rope"], cv2.IMREAD_GRAYSCALE)
        unknown = read_image(paths["unknown"], cv2.IMREAD_GRAYSCALE)
        under = read_image(paths["inferred_under_occlusion"], cv2.IMREAD_GRAYSCALE)
        read_image(paths["overlay"], cv2.IMREAD_COLOR)
        if "qa_comparison" in paths:
            read_image(paths["qa_comparison"], cv2.IMREAD_COLOR)
    except Exception as exc:
        errors.append(f"image decode: {type(exc).__name__}: {exc}")
        return ({"output_file": row["output_file"], "camera_role": row["camera_role"],
                 "source_group": row["source_group"], "status": "fail", "errors": errors},
                file_rows)
    shape = source.shape[:2]
    for kind, mask in (("reference ROI", reference), ("effective ROI", effective),
                       ("observed", observed), ("inferred", inferred),
                       ("supported rope", rope), ("unknown", unknown),
                       ("inferred under occlusion", under)):
        errors.extend(binary_errors(kind, mask, shape))
    if rgb.shape != source.shape or rgb.dtype != np.uint8:
        errors.append("effective RGB wrong shape or dtype")
    # 形状不符时不能逐像素广播；形状错误已经单独记录。
    if not any("wrong shape" in error for error in errors):
        ref = reference > 0
        eff = effective > 0
        obs = observed > 0
        inf = inferred > 0
        sup = rope > 0
        unk = unknown > 0
        if np.any(sup & ~ref) or np.any(unk & ~ref):
            errors.append("supported/unknown exclusion outside reference ROI")
        if np.any(sup & unk):
            errors.append("supported and unknown exclusions overlap")
        if not np.array_equal(eff, ref & ~sup & ~unk):
            errors.append("effective ROI differs from reference minus both exclusions")
        if np.any(obs & ~eff):
            errors.append("observed pixels outside effective ROI")
        # 紫色允许在有证据的活动绳遮挡里，但必须与黄色、unknown、ROI 外互斥。
        if np.any(inf & (~ref | unk | obs)):
            errors.append("inferred pixels overlap observed/unknown or leave reference ROI")
        if not np.array_equal(under > 0, inf & sup):
            errors.append("inferred_under_occlusion is not exact inferred ∩ supported_rope")
        if rgb.shape == source.shape:
            if np.any(rgb[~eff] != 0):
                errors.append("effective RGB outside effective ROI is not black")
            if not np.array_equal(rgb[eff], source[eff]):
                errors.append("effective RGB inside effective ROI differs from source")
    try:
        content = json.loads(paths["record"].read_text(encoding="utf-8"))
        record, lines = content["record"], content["lines"]
        if not isinstance(record, dict) or not isinstance(lines, list):
            raise ValueError("record or lines has wrong type")
        for field, expected in (("output_file", row["output_file"]),
                                ("camera_role", row["camera_role"]),
                                ("source_sha256", row["output_sha256"]),
                                ("dataset_use_status", row["dataset_use_status"])):
            if record.get(field) != expected:
                errors.append(f"record {field} differs from input manifest")
        if old_roi_path.is_file() and record.get("reference_roi_sha256") != sha256(old_roi_path):
            errors.append("record reference ROI hash mismatch")
        for field, mask in (("observed_pixels", observed),
                            ("inferred_pixels", inferred),
                            ("inferred_under_occlusion_pixels", under),
                            ("rope_pixels", rope), ("unknown_pixels", unknown)):
            if record.get(field) != int(np.count_nonzero(mask)):
                errors.append(f"record {field} count differs from raster")
        if record.get("fragment_or_family_count") != len(lines):
            errors.append("record line count differs from lines list")
        n_observed, observed_point_errors = safe_points(lines, "observed_points_xy", observed)
        errors.extend(observed_point_errors)
        _, inferred_point_errors = safe_points(lines, "inferred_points_xy", inferred)
        errors.extend(inferred_point_errors)
        if record.get("v3_observed_points") != n_observed:
            errors.append("record v3_observed_points differs from line point count")
        if np.any(observed) and n_observed == 0:
            errors.append("observed raster nonempty but records contain no observed points")
        if np.any(inferred) and not any(line.get("inferred_points_xy") for line in lines):
            errors.append("inferred raster nonempty but records contain no inferred points")
        # 双向匹配：v3 黄点画半径 1 的采样圆，紫点则各画一个像素。
        # 重新从记录生成位图能抓出“记录点都在图上、但图上另藏未记录块”的问题。
        if not observed_point_errors and effective.shape == observed.shape:
            rebuilt_observed = np.zeros_like(observed)
            for line in lines:
                for x, y in line.get("observed_points_xy", []):
                    cv2.circle(rebuilt_observed, (x, y), 1, 255, -1, cv2.LINE_8)
            rebuilt_observed[effective == 0] = 0
            if not np.array_equal(rebuilt_observed, observed):
                errors.append("observed raster differs from points rendered with radius 1")
        if not inferred_point_errors:
            rebuilt_inferred = np.zeros_like(inferred)
            for line in lines:
                for x, y in line.get("inferred_points_xy", []):
                    rebuilt_inferred[y, x] = 255
            if not np.array_equal(rebuilt_inferred, inferred):
                errors.append("inferred raster differs from recorded inferred points")
    except Exception as exc:
        errors.append(f"record parse/check: {type(exc).__name__}: {exc}")
    finding = {"output_file": row["output_file"], "camera_role": row["camera_role"],
               "source_group": row["source_group"],
               "status": "pass" if not errors else "fail", "errors": errors,
               "observed_pixels": int(np.count_nonzero(observed)),
               "inferred_pixels": int(np.count_nonzero(inferred)),
               "inferred_under_occlusion_pixels": int(np.count_nonzero(under))}
    return finding, file_rows


def read_plan(run: Path) -> tuple[list[dict], set[str], dict, list[str]]:
    """输入固定集合与日志逐项对账，保证失败图仍在分母中。"""
    errors = []
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest_hash, qa_hash = sha256(MANIFEST), sha256(QA_PLAN)
    if manifest_hash != protocol["input_manifest_sha256"]:
        errors.append("input manifest hash differs from protocol")
    if qa_hash != protocol["fixed_qa_plan_sha256"]:
        errors.append("QA plan hash differs from protocol")
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    with QA_PLAN.open("r", encoding="utf-8-sig", newline="") as stream:
        qa_rows = list(csv.DictReader(stream))
    qa = {row["output_file"] for row in qa_rows}
    if len(manifest) != 400 or Counter(row["camera_role"] for row in manifest) != \
            {"left": 200, "right": 200}:
        errors.append("manifest does not list exactly 200 left and 200 right inputs")
    names = [row["output_file"] for row in manifest]
    if len(set(names)) != len(names):
        errors.append("duplicate output_file in input manifest")
    if len(qa_rows) != 38 or len(qa) != 38 or not qa.issubset(names):
        errors.append("fixed QA plan does not select 38 distinct manifest inputs")
    for row in manifest:
        if ("20260925" in row["source_file"] + row["output_file"] or
                any(token in row["dataset_use_status"].lower()
                    for token in ("reserved", "heldout"))):
            errors.append(f"reserved input in manifest: {row['output_file']}")
    context = json.loads((run / "run_context.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    mode = context.get("mode")
    if mode not in ("pilot", "all"):
        raise ValueError("run context lacks pilot/all mode")
    selection = manifest if mode == "all" else [row for row in manifest if row["output_file"] in qa]
    if len(selection) != (400 if mode == "all" else 38):
        errors.append("selected input count differs from fixed mode")
    if (context.get("input_manifest_sha256") != manifest_hash or
            context.get("qa_plan_sha256") != qa_hash or
            context.get("protocol_sha256") != sha256(PROTOCOL)):
        errors.append("run context input/protocol hashes differ from current fixed files")
    if summary.get("mode") != mode or summary.get("planned") != len(selection):
        errors.append("run summary mode/planned differs from fixed selection")
    log_rows = []
    with (run / "run_log.jsonl").open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                item = json.loads(line)
                log_rows.append(item)
            except Exception as exc:
                errors.append(f"run log line {line_number} malformed: {exc}")
    expected = {row["output_file"] for row in selection}
    logged = [row.get("output_file") for row in log_rows]
    missing = sorted(expected - set(logged))
    extra = sorted(set(logged) - expected, key=str)
    if missing:
        errors.append(f"run log omitted {len(missing)} selected inputs: {missing[:15]}")
    if extra:
        errors.append(f"run log has {len(extra)} unselected inputs: {extra[:15]}")
    if len(logged) != len(set(logged)):
        errors.append("run log has duplicate output_file entries")
    if len(log_rows) != len(selection):
        errors.append("run log line count differs from selected input count")
    complete = sum(item.get("status") == "completed" for item in log_rows)
    if summary.get("completed") != complete or summary.get("failed") != len(log_rows)-complete:
        errors.append("run summary completed/failed counts disagree with run log")
    for item in log_rows:
        if item.get("status") != "completed":
            errors.append(f"run failed input retained: {item.get('output_file')}: {item.get('error')}")
    return selection, qa, {"mode": mode, "summary": summary,
                            "run_context_sha256": sha256(run / "run_context.json"),
                            "run_summary_sha256": sha256(run / "summary.json")}, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {output}")
    if not run.is_dir():
        raise NotADirectoryError(run)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "audit_script_sha256": sha256(Path(__file__)),
        "meaning": "file, input, pixel and record integrity only; not semantic boundary accuracy",
        "global_errors": [], "findings": [], "file_manifest": [],
    }
    try:
        selection, qa, metadata, plan_errors = read_plan(run)
        result.update(metadata)
        result["global_errors"].extend(plan_errors)
        for index, row in enumerate(selection, 1):
            try:
                finding, files = audit_one(run, row, qa)
            except Exception as exc:
                finding = {"output_file": row["output_file"],
                           "camera_role": row["camera_role"],
                           "source_group": row["source_group"],
                           "status": "fail",
                           "errors": [f"unexpected audit exception: {type(exc).__name__}: {exc}",
                                      traceback.format_exc()]}
                files = []
            finding["index"] = index
            result["findings"].append(finding)
            for file_item in files:
                file_item["output_file"] = row["output_file"]
                result["file_manifest"].append(file_item)
            if index % 25 == 0 or finding["status"] == "fail":
                print(f"audit {index}/{len(selection)} {row['output_file']}: "
                      f"{finding['status']}", flush=True)
        result["planned"] = len(selection)
        result["passed"] = sum(item["status"] == "pass" for item in result["findings"])
        result["failed"] = len(selection) - result["passed"]
        result["role_passed"] = dict(Counter(item["camera_role"] for item in result["findings"]
                                              if item["status"] == "pass"))
    except Exception as exc:
        result["global_errors"].append(f"fatal setup: {type(exc).__name__}: {exc}")
        result["global_errors"].append(traceback.format_exc())
        result.setdefault("planned", 0)
        result.setdefault("passed", 0)
        result.setdefault("failed", 0)
    result["file_manifest_rows"] = len(result["file_manifest"])
    result["exit_code"] = int(bool(result["global_errors"] or result["failed"]))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("planned", "passed", "failed", "file_manifest_rows", "exit_code")},
                     ensure_ascii=False), flush=True)
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
