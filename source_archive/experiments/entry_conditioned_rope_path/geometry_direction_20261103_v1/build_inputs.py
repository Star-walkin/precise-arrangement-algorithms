"""Freeze D1's 292 source images and E1 entries for the geometry experiment.

This script reads JSON metadata only. It does not decode images, inspect labels,
or load any network response. An invalid E1 remains in the output as a failure
row, so downstream rendering can still produce one result for every source.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D1 = ROOT / "experiments" / "entry_conditioned_rope_path" / "data_expansion_d1_20261102_v1"
INDEX_PATH = D1 / "final_292_index.json"
PROTOCOL_PATH = D1 / "protocol.json"
OUTPUT_PATH = HERE / "inputs.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def validate_xy(value: object, shape_hw: object) -> str | None:
    if not isinstance(shape_hw, list) or len(shape_hw) != 2:
        return "missing_or_invalid_image_shape"
    height, width = shape_hw
    if not all(isinstance(v, int) and v > 0 for v in (height, width)):
        return "missing_or_invalid_image_shape"
    if not isinstance(value, list) or len(value) != 2:
        return "missing_or_invalid_e1_points"
    points: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            return "missing_or_invalid_e1_points"
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in point):
            return "non_numeric_e1_point"
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            return "non_finite_e1_point"
        if not (0 <= x < width and 0 <= y < height):
            return "e1_point_outside_original_image"
        points.append((x, y))
    if math.dist(*points) <= 1e-6:
        return "coincident_e1_points"
    return None


def main() -> None:
    index = read_json(INDEX_PATH)
    protocol = read_json(PROTOCOL_PATH)
    old_rows = index["rows"]
    jobs = protocol["gallery_jobs"]
    if len(old_rows) != index["unique_source_count"] or len(old_rows) != 292:
        raise ValueError("D1 final_292_index must contain exactly 292 rows")
    source_ids = [row["source_id"] for row in old_rows]
    if len(set(source_ids)) != 292:
        raise ValueError("D1 final_292_index source IDs are not unique")
    if len({row["source_sha256"] for row in old_rows}) != 292:
        raise ValueError("D1 final_292_index source hashes are not unique")

    jobs_by_id: dict[str, list[dict]] = defaultdict(list)
    for job in jobs:
        jobs_by_id[job["id"]].append(job)

    rows: list[dict] = []
    for order_index, old in enumerate(old_rows):
        source_id = old["source_id"]
        matches = jobs_by_id.get(source_id, [])
        job = matches[0] if len(matches) == 1 else None
        reasons: list[str] = []
        if not matches:
            reasons.append("missing_protocol_gallery_job")
        elif len(matches) > 1:
            reasons.append("duplicate_protocol_gallery_job")

        shape = job.get("shape") if job else None
        bbox = job.get("bbox") if job else None
        camera_role = job.get("camera_role") if job else None
        seeds = job.get("seeds", []) if job else []
        entry_count = len(seeds) if isinstance(seeds, list) else 0
        seed = seeds[0] if entry_count and isinstance(seeds[0], dict) else None
        if job is not None:
            if job.get("image_path") != old["source"]:
                reasons.append("source_path_mismatch")
            if job.get("image_sha256") != old["source_sha256"]:
                reasons.append("source_hash_mismatch")
            if job.get("source_group") != old["source_group"]:
                reasons.append("source_group_mismatch")
            if job.get("main_display_entry_index") != 0 or old.get("entry_index") != 0:
                reasons.append("frozen_e1_index_mismatch")
            if old.get("entry_count") != entry_count:
                reasons.append("entry_count_mismatch")
            if not isinstance(bbox, list) or len(bbox) != 4:
                reasons.append("missing_or_invalid_search_bbox")
        if seed is None:
            reasons.append("missing_frozen_e1")

        xy = seed.get("xy") if seed else None
        xy_error = validate_xy(xy, shape)
        if xy_error is not None:
            reasons.append(xy_error)
        entry_source = seed.get("source") if seed else None
        if job is not None and entry_source != old.get("entry_source"):
            reasons.append("entry_source_mismatch")
        if not isinstance(entry_source, str) or not entry_source:
            reasons.append("missing_entry_source")
        source_hashes = job.get("seed_source_hashes", []) if job else []
        entry_source_hash = next(
            (item.get("sha256") for item in source_hashes if item.get("path") == entry_source),
            None,
        )
        if entry_source and entry_source_hash is None:
            reasons.append("missing_entry_source_hash_in_protocol")

        row = {
            "order_index": order_index,
            "source_id": source_id,
            "source_group": old["source_group"],
            "source_use": old["source_use"],
            "independent_blind_test": False,
            "source_path": old["source"],
            "source_sha256": old["source_sha256"],
            "source_file_exists": Path(old["source"]).is_file(),
            "camera_role": camera_role,
            "shape_hw": shape,
            "bbox_original_xyxy": bbox,
            "entry_count": entry_count,
            "e1": {
                "index": 0,
                "xy": xy,
                "source": entry_source,
                "source_sha256": entry_source_hash,
                "entry_id": seed.get("entry_id") if seed else None,
                "ambiguous": seed.get("ambiguous") if seed else None,
                "old_entry_index": seed.get("old_entry_index") if seed else None,
                "valid": not reasons,
                "failure_reasons": reasons,
            },
        }
        rows.append(row)

    train_groups = protocol["train_groups"]
    eval_groups = protocol["eval_groups"]
    eval_ids = protocol["eval_ids"]
    if set(train_groups) & set(eval_groups):
        raise ValueError("D1 training and evaluation groups overlap")

    result = {
        "schema_version": "geometry_direction_inputs_v1",
        "source_files": {
            "d1_final_292_index": {"path": str(INDEX_PATH), "sha256": sha256(INDEX_PATH)},
            "d1_protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
        },
        "source_count": len(rows),
        "entry_condition_count": sum(row["entry_count"] for row in rows),
        "row_order": "D1 final_292_index.rows order; join protocol.gallery_jobs by source_id, never array position",
        "development_evaluation_policy": {
            "development_source_groups": train_groups,
            "historical_open_evaluation_source_groups": eval_groups,
            "historical_open_evaluation_image_ids": eval_ids,
            "parameter_selection_rule": "Use only development_source_groups; exclude evaluation groups and other gallery groups until parameters are frozen.",
            "gallery_evaluation_source_group_rows_are_display_only": sum(
                row["source_group"] in set(eval_groups) for row in rows
            ),
            "reserved_media_and_labels_access": False,
            "target_labels_read_by_this_builder": False,
        },
        "e1_valid_count": sum(row["e1"]["valid"] for row in rows),
        "e1_invalid_count": sum(not row["e1"]["valid"] for row in rows),
        "source_use_counts": dict(Counter(row["source_use"] for row in rows)),
        "rows": rows,
        "mapping_sha256": hashlib.sha256(canonical_bytes(rows)).hexdigest(),
    }
    if len(jobs) != 292 or result["entry_condition_count"] != index["entry_conditions"]:
        raise ValueError("D1 gallery job or entry count changed")
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if OUTPUT_PATH.exists():
        if OUTPUT_PATH.read_bytes() != encoded:
            raise FileExistsError(f"Refusing to overwrite different input mapping: {OUTPUT_PATH}")
    else:
        with OUTPUT_PATH.open("xb") as stream:
            stream.write(encoded)
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH),
                "source_count": result["source_count"],
                "entry_condition_count": result["entry_condition_count"],
                "e1_valid_count": result["e1_valid_count"],
                "e1_invalid_count": result["e1_invalid_count"],
                "mapping_sha256": result["mapping_sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
