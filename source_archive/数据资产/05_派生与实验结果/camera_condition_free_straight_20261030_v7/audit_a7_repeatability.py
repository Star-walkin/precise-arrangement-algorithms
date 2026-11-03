"""Compare saved formal PNG bytes from fixed development runs with all_001.

This is a read-only check of predictions.  It neither reruns inference nor uses
visual truth.  A new report is required; earlier failures are never overwritten.
"""
from pathlib import Path
import hashlib
import json
import sys


BASE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = BASE / "all_001"
    report_path = destination / "independent_repeatability_audit.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    target_manifest = json.loads(
        (destination / "independent_output_manifest.json").read_text("utf-8-sig")
    )
    target_map = {item["path"]: item for item in target_manifest["files"]}
    rows = []
    groups = []
    for run in ["pilot_003", "extra_001"]:
        root = BASE / run
        manifest = json.loads(
            (root / ("independent_output_manifest_visibility_001.json" if run == "pilot_003" else "independent_output_manifest.json")).read_text("utf-8-sig")
        )
        group = {"run": run, "png_count": 0, "identical_count": 0, "errors": []}
        for item in manifest["files"]:
            relative = item["path"]
            if Path(relative).suffix.lower() != ".png":
                continue
            first, second = root / relative, destination / relative
            errors = []
            same = False
            first_hash = digest(first)
            second_hash = digest(second) if second.exists() else None
            if first_hash != item["sha256"]:
                errors.append("source_manifest_hash_changed")
            target = target_map.get(relative)
            if target is None:
                errors.append("missing_from_all_manifest")
            elif second_hash != target["sha256"]:
                errors.append("all_manifest_hash_changed")
            if second.exists():
                # Compare actual bytes as well as SHA256, without decoding PNG.
                same = first.read_bytes() == second.read_bytes()
            if not same:
                errors.append("PNG_bytes_differ")
            rows.append({"run": run, "path": relative,
                         "source_sha256": first_hash,
                         "all_sha256": second_hash,
                         "bytes_equal": same, "errors": errors})
            group["png_count"] += 1
            group["identical_count"] += int(same and not errors)
            if errors:
                group["errors"].append({"path": relative, "errors": errors})
        groups.append(group)
    result = {
        "scope": "Byte equality of every formal PNG, including original-size binary masks; no accuracy claim.",
        "target_run": "all_001",
        "passed": all(not row["errors"] for row in rows),
        "compared_png_count": len(rows),
        "identical_png_count": sum(row["bytes_equal"] and not row["errors"] for row in rows),
        "groups": groups,
        "files": rows,
        "audit_source_sha256": digest(Path(__file__)),
    }
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
