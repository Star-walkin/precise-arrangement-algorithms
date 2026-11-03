"""Read-only v2/v3 occlusion and purple-line audit from saved masks.

This script decodes derived binary PNG masks and saved JSON records only.  It
never opens source photographs, runs a model, or changes either algorithm.
Pixel counts describe mask coverage and classification, not segmentation
accuracy or a confirmed physical rope boundary.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DEFAULT_V2 = BASE / "camera_condition_bright_rope_boundaries_20261023_v2" / "all_002"
MASK_KEYS = {
    "v2_supported": "boundaries/incoming_rope_occlusion",
    "v2_unknown": "boundaries/unknown_exclusion",
    "v2_purple": "boundaries/inferred",
    "v3_supported": "boundaries/incoming_rope_occlusion",
    "v3_unknown": "boundaries/unknown_exclusion",
    "v3_purple": "boundaries/inferred",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise ValueError(f"Cannot decode binary mask: {path}")
    unique = np.unique(image)
    if not np.all(np.isin(unique, (0, 255))):
        raise ValueError(f"Mask is not 0/255: {path}, values={unique[:12]}")
    return image > 0


def _item(path: Path, line_number: int) -> dict:
    try:
        row = json.loads(path.read_text(encoding="utf-8").splitlines()[line_number])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid run log line {line_number + 1}: {path}") from exc
    return row


def _merge(aggregate: dict, item: dict) -> None:
    aggregate["images"] += 1
    for key in ("v2_supported_pixels", "v2_unknown_pixels", "v2_union_pixels",
                "v3_supported_pixels", "v3_unknown_pixels", "v3_union_pixels",
                "newly_excluded_pixels", "removed_exclusion_pixels",
                "unchanged_exclusion_pixels", "v2_supported_to_v3_unknown_pixels",
                "v2_unknown_to_v3_supported_pixels", "v3_supported_outside_v2_union_pixels",
                "v3_unknown_outside_v2_union_pixels", "v3_purple_pixels",
                "v3_purple_on_supported_pixels", "v3_purple_on_unknown_pixels",
                "v2_purple_pixels", "v2_purple_on_union_pixels",
                "v3_purple_on_v2_union_pixels"):
        aggregate[key] += item[key]
    for key in ("v3_supported", "v3_unknown", "newly_excluded",
                "removed_exclusion", "v3_purple_on_supported",
                "v3_purple_on_unknown"):
        aggregate["images_with_" + key] += int(item[key + "_pixels"] > 0)
    aggregate["history_called_images"] += int(item["history_called"] is True)
    aggregate["history_valid_images"] += int(item["history_valid"] is True)
    aggregate["history_reasons"][str(item["history_reason"])] += 1
    aggregate["ribbon_statuses"][str(item["ribbon_status"])] += 1


def _blank() -> dict:
    return defaultdict(int, history_reasons=Counter(), ribbon_statuses=Counter())


def _plain(value):
    if isinstance(value, Counter):
        return dict(sorted(value.items()))
    if isinstance(value, dict):
        return {key: _plain(v) for key, v in sorted(value.items())}
    return value


def _audit_one(row: dict, v3: Path, v2: Path) -> dict:
    rel = Path(row["output_file"])
    role_dir = rel.parent.name
    name = rel.name
    masks = {}
    for key, subdir in MASK_KEYS.items():
        root = v2 if key.startswith("v2") else v3
        masks[key] = _mask(root / subdir / role_dir / name)
    shape = masks["v3_supported"].shape
    if any(value.shape != shape for value in masks.values()):
        raise ValueError(f"Mask dimensions differ for {row['output_file']}")
    old_s, old_u, old_p = (masks[k] for k in ("v2_supported", "v2_unknown", "v2_purple"))
    new_s, new_u, new_p = (masks[k] for k in ("v3_supported", "v3_unknown", "v3_purple"))
    if np.any(new_s & new_u):
        raise ValueError(f"v3 supported/unknown overlap: {row['output_file']}")
    old_union = old_s | old_u
    new_union = new_s | new_u
    diag = row.get("occlusion_diagnostics", {})
    ribbon = diag.get("oriented_ribbon", {})
    result = {
        "output_file": row["output_file"],
        "camera_role": row["camera_role"],
        "source_group": row["source_group"],
        "v2_supported_pixels": int(np.count_nonzero(old_s)),
        "v2_unknown_pixels": int(np.count_nonzero(old_u)),
        "v2_union_pixels": int(np.count_nonzero(old_union)),
        "v3_supported_pixels": int(np.count_nonzero(new_s)),
        "v3_unknown_pixels": int(np.count_nonzero(new_u)),
        "v3_union_pixels": int(np.count_nonzero(new_union)),
        "newly_excluded_pixels": int(np.count_nonzero(new_union & ~old_union)),
        "removed_exclusion_pixels": int(np.count_nonzero(old_union & ~new_union)),
        "unchanged_exclusion_pixels": int(np.count_nonzero(old_union & new_union)),
        "v2_supported_to_v3_unknown_pixels": int(np.count_nonzero(old_s & new_u)),
        "v2_unknown_to_v3_supported_pixels": int(np.count_nonzero(old_u & new_s)),
        "v3_supported_outside_v2_union_pixels": int(np.count_nonzero(new_s & ~old_union)),
        "v3_unknown_outside_v2_union_pixels": int(np.count_nonzero(new_u & ~old_union)),
        "v3_purple_pixels": int(np.count_nonzero(new_p)),
        "v3_purple_on_supported_pixels": int(np.count_nonzero(new_p & new_s)),
        "v3_purple_on_unknown_pixels": int(np.count_nonzero(new_p & new_u)),
        "v2_purple_pixels": int(np.count_nonzero(old_p)),
        "v2_purple_on_union_pixels": int(np.count_nonzero(old_p & old_union)),
        "v3_purple_on_v2_union_pixels": int(np.count_nonzero(new_p & old_union)),
        "history_called": diag.get("history_module_called"),
        "history_valid": diag.get("history_valid"),
        "history_reason": diag.get("history_reason"),
        "ribbon_status": ribbon.get("status"),
        "v3_reported_inferred_under_occlusion_pixels": row.get("inferred_under_occlusion_pixels"),
    }
    reported = row.get("inferred_under_occlusion_pixels")
    if reported is not None and result["v3_purple_on_supported_pixels"] != reported:
        raise ValueError(f"Purple/occlusion count differs from run record: {row['output_file']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Completed v3 run directory, e.g. all_001")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON analysis path; existing files are not overwritten")
    parser.add_argument("--v2-run", type=Path, default=DEFAULT_V2,
                        help="Frozen v2 comparison run (default: all_002)")
    args = parser.parse_args()
    v3, v2, output = (p.resolve() for p in (args.run, args.v2_run, args.output))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    log = v3 / "run_log.jsonl"
    if not log.is_file():
        raise FileNotFoundError(log)
    run_context = v3 / "run_context.json"
    if not run_context.is_file():
        raise FileNotFoundError(run_context)
    context = json.loads(run_context.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = _blank()
    roles: dict[str, dict] = defaultdict(_blank)
    groups: dict[tuple[str, str], dict] = defaultdict(_blank)
    per_image = []
    errors = []
    for index, row in enumerate(rows, 1):
        if row.get("status") != "completed":
            errors.append({"output_file": row.get("output_file"), "reason": "v3_run_error",
                           "run_error": row.get("error")})
            continue
        try:
            item = _audit_one(row, v3, v2)
        except Exception as exc:
            errors.append({"output_file": row.get("output_file"), "reason": repr(exc)})
            continue
        per_image.append(item)
        _merge(total, item)
        _merge(roles[item["camera_role"]], item)
        _merge(groups[(item["camera_role"], item["source_group"])], item)
        if index % 40 == 0:
            print(f"Audited {index}/{len(rows)} derived-mask records", flush=True)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "saved binary masks and saved run log only; no source images decoded",
        "v3_run": str(v3), "v2_run": str(v2),
        "v3_run_context_sha256": _sha(run_context),
        "v3_run_log_sha256": _sha(log),
        "analysis_script_sha256": _sha(Path(__file__).resolve()),
        "planned_or_logged": len(rows), "audited": len(per_image), "errors": errors,
        "total": _plain(total),
        "by_camera": {role: _plain(value) for role, value in sorted(roles.items())},
        "by_camera_source_group": [
            {"camera_role": role, "source_group": group, **_plain(value)}
            for (role, group), value in sorted(groups.items())],
        "per_image": per_image,
        "interpretation": (
            "Supported pixels have local image evidence only; unknown pixels are exclusions, "
            "not confirmed free rope. Purple overlap is a raster intersection, not a physical "
            "same-rope crossing or a segmentation accuracy score. Counts of valid calls or "
            "supported images are not success rates."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "logged": len(rows),
                      "audited": len(per_image), "errors": len(errors),
                      "newly_excluded_pixels": total["newly_excluded_pixels"],
                      "removed_exclusion_pixels": total["removed_exclusion_pixels"],
                      "v3_purple_on_supported_pixels": total["v3_purple_on_supported_pixels"]},
                     ensure_ascii=False), flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
