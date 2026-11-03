"""C4 离线全量交付封存；只在所有固定检查通过后创建两个新 JSON。

示例：python -B finalize_delivery.py --run all_001 --pilot pilot_002 --evaluation evaluation_002
不导入图像模型或设备模块，不修改预测、协议、审计或旧报告。
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import stat
import sys
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname


HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
ROOT = ASSETS.parents[1]
DATA_MANIFEST = ASSETS / "camera_condition_images_20261009_v2/manifest.csv"
QA_PLAN = ASSETS / "camera_condition_bright_rope_boundaries_20261009_v1/qa_plan.csv"
BASELINE_MANIFEST = ROOT / "工程导航/05_实验版本档案/卷绳边界_整链基准_C3_20261026/baseline_manifest.json"
ROI_POLICY = "SAM2_v2_union_B1_bright"

# run_full_surface.process 实际每图写出的 18 个文件。固定 QA 图另外写 qa_compare。
PNG_DIRS = (
    "roi/sam_reference_masks", "roi/sam_reference_masked",
    "roi/bright_reference_masks", "boundaries/original_observed",
    "roi/surface_masks", "roi/surface_masked", "roi/effective_masks",
    "boundaries/dark_observed", "boundaries/modeled", "boundaries/inferred",
    "boundaries/simple_c3_extension", "boundaries/incoming_rope_occlusion",
    "boundaries/unknown_exclusion", "overlays",
)
JPG_DIRS = ("previews", "smooth_previews", "group_overlays")
MODEL_CORE = "full_surface_core.py"
LINK_DATA_ATTRS = {"data-evidence", "data-curve", "data-group", "data-roi", "data-full"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"File changed while hashing: {path}")
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def fixed_path(argument: Path, label: str) -> Path:
    # 仅允许指定 v7 下的直接子目录，不把任意路径递归遍历为本次运行。
    path = (HERE / argument).absolute() if not argument.is_absolute() else argument.absolute()
    require(path.parent == HERE and path.name not in {"", ".", ".."},
            f"{label} must be a direct v7 run directory: {path}")
    require(path.is_dir() and not is_reparse(path), f"{label} is missing or a reparse point: {path}")
    return path


def is_reparse(path: Path) -> bool:
    info = path.lstat()
    attrs = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def ordinary_file(path: Path, label: str) -> Path:
    require(path.is_file(), f"Missing {label}: {path}")
    require(not is_reparse(path), f"Reparse/symlink file forbidden: {path}")
    return path


def safe_walk(directory: Path) -> list[Path]:
    """不跟随 junction、符号链接或其他重解析点，且拒绝含它们的输出树。"""
    require(directory.is_dir() and not is_reparse(directory), f"Unsafe output root: {directory}")
    files: list[Path] = []
    pending = [directory]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                info = entry.stat(follow_symlinks=False)
                attrs = getattr(info, "st_file_attributes", 0)
                require(not entry.is_symlink() and not (attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)),
                        f"Reparse point in output tree: {path}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files.append(path)
                else:
                    raise RuntimeError(f"Unexpected filesystem entry: {path}")
    return sorted(files, key=lambda item: item.relative_to(directory).as_posix())


def case_paths(run: Path, output_file: str, qa: bool) -> list[Path]:
    rel = Path(output_file)
    require(len(rel.parts) == 3 and rel.parts[0] == "images" and
            rel.parts[1] in {"left_camera", "right_camera"} and rel.suffix.lower() == ".png" and
            ".." not in rel.parts, f"Unsafe/unknown output_file: {output_file}")
    role, name, stem = rel.parent.name, rel.name, rel.stem
    files = [run / part / role / name for part in PNG_DIRS]
    files += [run / "boundaries/records" / role / (stem + ".json")]
    files += [run / part / role / (stem + ".jpg") for part in JPG_DIRS]
    if qa:
        files.append(run / "qa_compare" / role / (stem + ".jpg"))
    require(len(files) == (19 if qa else 18), f"Output contract count wrong: {output_file}")
    return files


class LocalLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.articles = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "article":
            self.articles += 1
        for name, value in attrs:
            looks_like_file = bool(value and re.search(r"\.(?:png|jpe?g|webp|gif|json|csv|html)(?:[?#]|$)",
                                                       value, flags=re.IGNORECASE))
            if value is not None and (name in {"href", "src"} or name in LINK_DATA_ATTRS or
                                      (name.startswith("data-") and looks_like_file)):
                self.links.append((name, value))


def link_target(page: Path, value: str) -> Path | None:
    value = value.strip()
    if not value or value.startswith("#"):
        return None
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return Path(unquote(value)).absolute()
    parsed = urlsplit(value)
    if parsed.scheme in {"http", "https", "mailto", "javascript", "data", "blob"}:
        return None
    if parsed.scheme == "file":
        return Path(url2pathname(unquote(parsed.path))).absolute()
    require(not parsed.scheme and not parsed.netloc, f"Unsupported link URL: {value}")
    path = Path(unquote(parsed.path.replace("\\", "/")))
    return (path if path.is_absolute() else page.parent / path).absolute()


def assert_no_reparse_parent(path: Path) -> None:
    # Path.absolute 不解析 junction；逐层 lstat，防止链接核验越过它。
    for part in (path, *path.parents):
        if part.exists() and is_reparse(part):
            raise RuntimeError(f"Local link crosses a reparse point: {path} via {part}")


def check_links(pages: list[tuple[Path, int | None]], future: set[Path]) -> dict:
    checked = 0
    deferred = 0
    counts = {}
    for page, expected_articles in pages:
        ordinary_file(page, "HTML review/report")
        parser = LocalLinks()
        parser.feed(page.read_text(encoding="utf-8"))
        if expected_articles is not None:
            require(parser.articles == expected_articles,
                    f"Wrong article count: {page} has {parser.articles}, expected {expected_articles}")
        counts[str(page)] = {"articles": parser.articles, "local_links": 0}
        for attr, raw in parser.links:
            target = link_target(page, raw)
            if target is None:
                continue
            assert_no_reparse_parent(target)
            if target in future and not target.exists():
                deferred += 1
            else:
                ordinary_file(target, f"local {attr} link in {page}")
            checked += 1
            counts[str(page)]["local_links"] += 1
    return {"pages": counts, "links_checked": checked, "planned_future_links": deferred}


def source_snapshot_check(run: Path, context: dict) -> None:
    for source, expected in context["source_sha256"].items():
        original = ordinary_file(Path(source), "source")
        snapshot = ordinary_file(run / "source_snapshot" / original.name, "source snapshot")
        require(digest(original) == expected and digest(snapshot) == expected,
                f"Source/snapshot differs from run_context: {source}")
    for model, expected in context["model_sha256"].items():
        require(digest(ordinary_file(Path(model), "model")) == expected,
                f"Model differs from run_context: {model}")


def read_selection() -> tuple[list[dict], set[str]]:
    with ordinary_file(DATA_MANIFEST, "fixed manifest").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with ordinary_file(QA_PLAN, "fixed QA plan").open(encoding="utf-8-sig", newline="") as stream:
        qa = {row["output_file"] for row in csv.DictReader(stream)}
    require(len(rows) == 400 and len(qa) == 38 and
            Counter(row["camera_role"] for row in rows) == {"left": 200, "right": 200} and
            qa.issubset({row["output_file"] for row in rows}), "Fixed selection changed")
    for row in rows:
        require("20260925" not in row["source_file"] + row["output_file"] and
                not any(word in row["dataset_use_status"].lower() for word in ("reserved", "heldout")),
                f"Reserved selection forbidden: {row['output_file']}")
    return rows, qa


def file_entry(run: Path, path: Path) -> dict:
    ordinary_file(path, "run output")
    return {"relative_path": path.relative_to(run).as_posix(), "absolute_path": str(path),
            "bytes": path.stat().st_size, "sha256": digest(path)}


def write_new(path: Path, value: dict) -> None:
    # x 模式是最终拒绝覆盖门槛；没有临时覆盖/替换旧证据的路径。
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--evaluation", type=Path, required=True)
    args = ap.parse_args()
    run = fixed_path(args.run, "full run")
    pilot = fixed_path(args.pilot, "pilot")
    evaluation = fixed_path(args.evaluation, "evaluation")
    require(len({run, pilot, evaluation}) == 3 and run.name == "all_001" and
            pilot.name == "pilot_002" and evaluation.name == "evaluation_002",
            "Expected frozen all_001 / pilot_002 / evaluation_002")
    output_manifest_path = run / "output_manifest.json"
    delivery_path = HERE / "delivery_manifest.json"
    require(not output_manifest_path.exists() and not delivery_path.exists(),
            "Refuse to overwrite an existing output_manifest.json or delivery_manifest.json")

    freeze = load(ordinary_file(HERE / "freeze_all_001.json", "frozen selection"))
    require(Path(freeze["selected_pilot"]) == pilot and Path(freeze["selected_evaluation"]) == evaluation and
            freeze.get("version") == "RB-C4.0.0" and freeze.get("remaining_tuning_rounds") == 0,
            "Full run does not match frozen choice")
    for path, expected in freeze["evidence_sha256"].items():
        require(digest(ordinary_file(Path(path), "frozen evidence")) == expected,
                f"Frozen evidence changed: {path}")

    full_summary = load(ordinary_file(run / "summary.json", "full summary"))
    full_audit = load(ordinary_file(run / "independent_audit.json", "full independent audit"))
    full_context = load(ordinary_file(run / "run_context.json", "full run context"))
    pilot_summary = load(ordinary_file(pilot / "summary.json", "pilot summary"))
    pilot_audit = load(ordinary_file(pilot / "independent_audit.json", "pilot audit"))
    pilot_context = load(ordinary_file(pilot / "run_context.json", "pilot context"))
    eval_summary = load(ordinary_file(evaluation / "summary.json", "evaluation summary"))
    eval_context = load(ordinary_file(evaluation / "run_context.json", "evaluation context"))

    require(full_summary.get("planned") == 400 and full_summary.get("completed") == 400 and
            full_summary.get("failed") == 0 and full_summary.get("source_unchanged") is True and
            full_summary.get("exit_code") == 0 and full_summary.get("roles") == {"left": 200, "right": 200},
            "Full run did not complete 400/400 with unchanged sources")
    require(full_audit.get("planned") == 400 and full_audit.get("passed") == 400 and
            full_audit.get("failed") == 0 and not full_audit.get("global_errors") and
            full_audit.get("roi_policy") == ROI_POLICY,
            "Full independent audit did not pass 400/400")
    require(pilot_summary.get("planned") == 38 and pilot_summary.get("completed") == 38 and
            pilot_summary.get("failed") == 0 and pilot_summary.get("source_unchanged") is True and
            pilot_summary.get("exit_code") == 0 and
            pilot_audit.get("passed") == 38 and pilot_audit.get("failed") == 0 and
            not pilot_audit.get("global_errors"), "Frozen pilot did not pass 38/38")
    require(eval_summary.get("metric_gate_pass") is True and eval_summary.get("source_unchanged") is True and
            eval_summary.get("exit_code") == 0 and
            all(value is True for value in eval_summary.get("gates", {}).values()) and
            len(eval_summary.get("gates", {})) >= 4,
            "Frozen evaluation gate failed")
    require(full_context.get("mode") == "all" and pilot_context.get("mode") == "pilot" and
            full_context.get("roi_policy") == ROI_POLICY and pilot_context.get("roi_policy") == ROI_POLICY and
            full_context.get("source_sha256") == pilot_context.get("source_sha256") == freeze["source_sha256"] and
            full_context.get("model_sha256") == pilot_context.get("model_sha256") and
            full_context.get("manifest_sha256") == pilot_context.get("manifest_sha256") == digest(DATA_MANIFEST) and
            full_context.get("qa_sha256") == pilot_context.get("qa_sha256") == digest(QA_PLAN),
            "Full/pilot source, model, input or ROI policy drift")
    core_hash = freeze["source_sha256"][str(HERE / MODEL_CORE)]
    require(eval_summary["source_sha256_start"][MODEL_CORE] == core_hash ==
            eval_summary["source_sha256_end"][MODEL_CORE] ==
            eval_context["source_sha256_start"][MODEL_CORE] ==
            eval_context["source_sha256_end"][MODEL_CORE],
            "Evaluation core hash differs from chosen full/pilot core")
    source_snapshot_check(run, full_context)
    source_snapshot_check(pilot, pilot_context)

    rows, qa = read_selection()
    comparison = {"fixed_qa_images": 38, "files_per_qa_image": 19,
                  "files_compared": 0, "byte_identical": True, "mismatches": []}
    expected_per_image: set[Path] = set()
    case_files: list[dict] = []
    for row in rows:
        selected = row["output_file"] in qa
        paths = case_paths(run, row["output_file"], selected)
        expected_per_image.update(paths)
        entries = [file_entry(run, path) for path in paths]
        case_files.append({"output_file": row["output_file"], "camera_role": row["camera_role"],
                           "qa_selected": selected, "files": entries})
        if selected:
            for entry in entries:
                other = pilot / entry["relative_path"]
                ordinary_file(other, "pilot comparison file")
                comparison["files_compared"] += 1
                if other.stat().st_size != entry["bytes"] or digest(other) != entry["sha256"]:
                    comparison["mismatches"].append(entry["relative_path"])
    require(len(expected_per_image) == 400 * 18 + 38 and comparison["files_compared"] == 38 * 19,
            "Fixed 400/38 output contract count changed")
    comparison["byte_identical"] = not comparison["mismatches"]
    require(comparison["byte_identical"],
            f"Pilot/full QA output bytes differ ({len(comparison['mismatches'])} mismatches)")

    all_files = safe_walk(run)
    require(output_manifest_path not in all_files, "Refuse existing output manifest")
    require(expected_per_image.issubset(set(all_files)), "Full tree is missing per-image output")
    other_files = [file_entry(run, path) for path in all_files if path not in expected_per_image]
    require({"summary.json", "run_context.json", "run_log.jsonl", "independent_audit.json",
             "review.html"}.issubset({item["relative_path"] for item in other_files}),
            "Full run metadata or review page missing")

    pages = [(ordinary_file(HERE / "结果说明.html", "root result report"), None),
             (ordinary_file(run / "review.html", "full review"), 400),
             (ordinary_file(pilot / "review.html", "pilot review"), 38)]
    future = {output_manifest_path, delivery_path}
    link_audit = check_links(pages, future)
    baseline = load(ordinary_file(BASELINE_MANIFEST, "C3 baseline manifest"))
    require(baseline.get("id") == "RB-C3.0.0", "Wrong parent baseline archive")

    output_manifest = {
        "format": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "version": "RB-C4.0.0", "source_selection": str(DATA_MANIFEST),
        "source_selection_sha256": digest(DATA_MANIFEST), "qa_selection_sha256": digest(QA_PLAN),
        "case_count": len(case_files), "per_image_file_count": len(expected_per_image),
        "other_run_file_count": len(other_files), "all_existing_file_count": len(all_files),
        "cases": case_files, "other_run_files": other_files,
        "self_excluded": "output_manifest.json is created after hashing and is included by hash in root delivery_manifest.json",
    }
    write_new(output_manifest_path, output_manifest)
    report_files = [HERE / name for name in (
        "run_full_surface.py", "full_surface_core.py", "protocol.json", "protocol_revision_002.json",
        "evaluation_protocol.json", "evaluation_full_surface.py", "independent_audit.py",
        "build_review.py", "freeze_selection.py", "freeze_all_001.json",
        "revision_notes.json", "finalize_delivery.py", "make_report.py",
        "update_indexes.py", "index_change_log.json", "full_visual_review_001.json",
        "结果说明.html")]
    root_files = [{"path": str(ordinary_file(path, "root delivery file")),
                   "bytes": path.stat().st_size, "sha256": digest(path)} for path in report_files]
    evidence_files = [run / name for name in (
        "summary.json", "run_context.json", "independent_audit.json", "review.html", "output_manifest.json")]
    evidence_files += [pilot / name for name in ("summary.json", "run_context.json", "independent_audit.json", "review.html")]
    evidence_files += [evaluation / name for name in ("summary.json", "run_context.json")]
    evidence_files += [BASELINE_MANIFEST]
    evidence = [{"path": str(ordinary_file(path, "delivery evidence")),
                 "bytes": path.stat().st_size, "sha256": digest(path)} for path in evidence_files]
    delivery = {
        "format": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": "RB-C4.0.0", "parent_baseline": "RB-C3.0.0",
        "run": str(run), "selected_pilot": str(pilot), "selected_evaluation": str(evaluation),
        "mask_roi_policy": ROI_POLICY,
        "sam_semantic_limit": "images/left_camera/ruler_left_05.png: SAM2 alone omitted an accepted bright ROI portion; the uniform SAM∪B1 working ROI prevents that crop, but does not repair the SAM2 semantic mask or prove full dark-surface coverage.",
        "real_tangency_or_cutpoint_coverage": "not_evaluated_no_ground_truth",
        "evaluation_scope": "fixed38_open_development_proxy_not_dark_or_cutpoint_truth",
        "full_run": {"planned": 400, "completed": 400, "failed": 0,
                      "independent_audit_passed": 400, "independent_audit_failed": 0},
        "fixed_qa_pilot_full_consistency": comparison,
        "output_inventory": {"per_image_files": len(expected_per_image),
                             "other_run_files_before_manifest": len(other_files),
                             "all_existing_files_before_manifest": len(all_files),
                             "manifest_path": str(output_manifest_path),
                             "manifest_sha256": digest(output_manifest_path)},
        "html_local_link_audit_before_manifest_creation": link_audit,
        "root_files": root_files, "evidence_files": evidence,
        "equipment_access": False, "reserved_inputs": False,
        "no_git_initialized": True,
    }
    write_new(delivery_path, delivery)
    final_links = check_links(pages, set())
    require(final_links["planned_future_links"] == 0, "Local report links still missing")
    print(json.dumps({"delivery_manifest": str(delivery_path),
                      "output_manifest": str(output_manifest_path),
                      "per_image_files": len(expected_per_image),
                      "pilot_full_compared": comparison["files_compared"],
                      "pilot_full_mismatches": len(comparison["mismatches"]),
                      "html_links_checked": final_links["links_checked"],
                      "full_audit_passed": full_audit["passed"], "exit_code": 0},
                     ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"exit_code": 1, "error": type(error).__name__,
                          "detail": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
