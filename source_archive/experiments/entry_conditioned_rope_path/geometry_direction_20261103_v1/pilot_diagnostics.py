"""Read-only development-pilot geometry overlays; no evaluation or heldout use.

For each pilot source, save a two-panel original-image diagnostic with G0/G1
candidate centers, seed normal-width bars, high-confidence background directions,
stop points, and the saved seed normal profile. This does not alter predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DEFAULT_PILOT = Path(r"C:\precise_arrangement_results\geometry_direction_20261103_v1\pilot_002")
COLORS = [(30, 230, 80), (230, 65, 225), (40, 210, 245)]  # BGR


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_color(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode development image: " + str(path))
    return image


def font(size: int):
    return ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", size)


def _line_segments(panel: np.ndarray, points: np.ndarray, states: list[str], color: tuple,
                   thickness: int) -> None:
    if len(points) != len(states):
        raise ValueError("candidate point/status mismatch")
    p = np.rint(points).astype(np.int32)
    for k in range(1, len(p)):
        if states[k] in ("input_entry", "entry_condition"):
            continue
        supported = states[k] in ("image_supported", "visible_supported")
        if supported or k % 5 < 3:
            cv2.line(panel, tuple(p[k - 1]), tuple(p[k]), (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.line(panel, tuple(p[k - 1]), tuple(p[k]), color if supported else (0, 175, 255),
                     thickness, cv2.LINE_AA)


def _draw_candidates(panel: np.ndarray, record: dict) -> None:
    selected = record.get("selected_candidate_id")
    for i, candidate in enumerate(record.get("candidates", [])[:3]):
        points = np.asarray(candidate.get("points_xy", []), float).reshape(-1, 2)
        if not len(points):
            continue
        color = COLORS[i]
        _line_segments(panel, points, candidate.get("point_status", []), color,
                       4 if candidate.get("id") == selected else 2)
        tip = tuple(np.rint(points[-1]).astype(int))
        cv2.circle(panel, tip, 11, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.circle(panel, tip, 11, (0, 40, 255), 3, cv2.LINE_AA)
        cv2.putText(panel, f"{candidate['id']} STOP", (tip[0] + 14, tip[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(panel, f"{candidate['id']} STOP", (tip[0] + 14, tip[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 40, 255), 2, cv2.LINE_AA)


def _seed_normals(panel: np.ndarray, seed: np.ndarray, width: float) -> None:
    if len(seed) < 2 or not np.isfinite(seed).all() or width <= 0:
        return
    vec = seed[-1] - seed[0]
    norm = float(np.linalg.norm(vec))
    if norm < 1:
        return
    tangent = vec / norm
    normal = np.array([-tangent[1], tangent[0]])
    for t in (0.2, 0.5, 0.8):
        center = seed[0] + t * vec
        a = tuple(np.rint(center - 1.25 * width * normal).astype(int))
        b = tuple(np.rint(center + 1.25 * width * normal).astype(int))
        l = tuple(np.rint(center - 0.5 * width * normal).astype(int))
        r = tuple(np.rint(center + 0.5 * width * normal).astype(int))
        cv2.line(panel, a, b, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(panel, a, b, (60, 230, 255), 2, cv2.LINE_AA)
        cv2.circle(panel, l, 5, (255, 100, 20), -1, cv2.LINE_AA)
        cv2.circle(panel, r, 5, (255, 100, 20), -1, cv2.LINE_AA)
    p = np.rint(seed).astype(np.int32)
    cv2.polylines(panel, [p], False, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.polylines(panel, [p], False, (255, 255, 20), 3, cv2.LINE_AA)


def _background_lines(panel: np.ndarray, fields: dict, stride: int = 28,
                      minimum_confidence: float = 0.5) -> int:
    conf = np.asarray(fields["direction_confidence"], np.float32)
    cos2 = np.asarray(fields["direction_cos2"], np.float32)
    sin2 = np.asarray(fields["direction_sin2"], np.float32)
    scale = float(fields["field_scale"])
    if conf.shape != cos2.shape or conf.shape != sin2.shape:
        raise ValueError("direction field shape mismatch")
    overlay = panel.copy()
    count = 0
    for iy in range(stride // 2, conf.shape[0], stride):
        for ix in range(stride // 2, conf.shape[1], stride):
            if not np.isfinite(conf[iy, ix]) or conf[iy, ix] < minimum_confidence:
                continue
            theta = 0.5 * np.arctan2(float(sin2[iy, ix]), float(cos2[iy, ix]))
            center = np.array([(ix + 0.5) / scale - 0.5, (iy + 0.5) / scale - 0.5])
            direction = np.array([np.cos(theta), np.sin(theta)])
            half = 38.0
            a = tuple(np.rint(center - half * direction).astype(int))
            b = tuple(np.rint(center + half * direction).astype(int))
            cv2.line(overlay, a, b, (255, 240, 35), 5, cv2.LINE_AA)
            count += 1
    cv2.addWeighted(overlay, 0.62, panel, 0.38, 0, dst=panel)
    return count


def _profile_strip(profile: list[float], width: float, target_width: int) -> Image.Image:
    height = 260
    board = Image.new("RGB", (target_width, height), (249, 249, 249))
    draw = ImageDraw.Draw(board)
    draw.text((24, 12), f"入口法向平均灰度剖面 | 光学成像宽度 {width:.1f} px（非物理绳径）",
              font=font(25), fill=(20, 30, 40))
    if not profile:
        draw.text((24, 86), "入口剖面不可用", font=font(26), fill=(150, 30, 30))
        return board
    values = np.asarray(profile, float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite saved entry profile")
    left, right, top, bottom = 75, target_width - 80, 66, 208
    lo = min(0.0, float(values.min()) - 0.03)
    hi = max(1.0, float(values.max()) + 0.03)
    x = np.linspace(left, right, len(values))
    y = bottom - (values - lo) / (hi - lo) * (bottom - top)
    draw.line([(left, bottom), (right, bottom)], fill=(120, 130, 140), width=2)
    draw.line([(left, top), (left, bottom)], fill=(120, 130, 140), width=2)
    for frac in (-0.5, 0.5):
        at = left + (frac + 1.25) / 2.5 * (right - left)
        draw.line([(at, top), (at, bottom)], fill=(200, 210, 220), width=2)
    draw.line(list(zip(x.tolist(), y.tolist())), fill=(25, 95, 200), width=4, joint="curve")
    draw.text((left, 218), "-1.25w", font=font(20), fill=(55, 65, 80))
    draw.text((right - 70, 218), "+1.25w", font=font(20), fill=(55, 65, 80))
    return board


def render_one(row: dict, pilot: Path, target: Path) -> dict:
    source = Path(row["source_path"])
    if sha(source) != row["source_sha256"]:
        raise ValueError("source hash changed: " + row["source_id"])
    image = read_color(source)
    if list(image.shape[:2]) != row["shape_hw"]:
        raise ValueError("source shape changed: " + row["source_id"])
    name = row["source_id"]
    records = {arm: read_json(pilot / "predictions" / arm / (name + ".json"))
               for arm in ("G0", "G1")}
    evidence_path = pilot / "evidence_float32" / (name + ".npz")
    with np.load(evidence_path, allow_pickle=False) as z:
        fields = {key: z[key] for key in ("direction_cos2", "direction_sin2", "direction_confidence", "field_scale")}
    g0, g1 = image.copy(), image.copy()
    _draw_candidates(g0, records["G0"])
    count = _background_lines(g1, fields)
    _draw_candidates(g1, records["G1"])
    seed = np.asarray(row["e1"].get("xy", []), float).reshape(-1, 2)
    width = float(records["G1"].get("seed_measurement", {}).get("optical_width_px", 0))
    _seed_normals(g0, seed, width)
    _seed_normals(g1, seed, width)
    # Diagnostic scale is fixed and separate from the original-pixel main PNG.
    h, w = image.shape[:2]
    size = (w // 2, h // 2)
    left = cv2.cvtColor(cv2.resize(g0, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.resize(g1, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    gap = 28
    header = 116
    profile = _profile_strip(records["G1"].get("seed_measurement", {}).get("profile_mean", []),
                             width, size[0] * 2 + gap)
    board = Image.new("RGB", (size[0] * 2 + gap, size[1] + header + profile.height), (247, 248, 250))
    board.paste(Image.fromarray(left), (0, header))
    board.paste(Image.fromarray(right), (size[0] + gap, header))
    board.paste(profile, (0, size[1] + header))
    draw = ImageDraw.Draw(board)
    draw.text((14, 8), "G0 原图几何候选（诊断）", font=font(31), fill=(0, 0, 0))
    draw.text((size[0] + gap + 14, 8), "G1 几何 + 周围卷面方向（诊断）", font=font(31), fill=(0, 0, 0))
    for i, arm in enumerate(("G0", "G1")):
        x = i * (size[0] + gap) + 14
        r = records[arm]
        label = f"{name} | {r.get('status')} | {r.get('stop_reason')} | 选中 {r.get('selected_candidate_id') or '无'}"
        draw.text((x, 55), label[:90], font=font(22), fill=(45, 55, 70))
    draw.text((size[0] + gap + 14, 87), f"青短线：背景无向轴，高置信采样 {count} 处；并非同根标签",
              font=font(19), fill=(35, 70, 95))
    if target.exists():
        raise FileExistsError("Will not overwrite diagnostic: " + str(target))
    board.save(target, format="PNG", compress_level=3)
    return {"source_id": name, "source_sha256": row["source_sha256"],
            "G0_prediction_sha256": sha(pilot / "predictions/G0" / (name + ".json")),
            "G1_prediction_sha256": sha(pilot / "predictions/G1" / (name + ".json")),
            "evidence_sha256": sha(evidence_path), "direction_lines": count,
            "output": str(target), "output_sha256": sha(target)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    args = ap.parse_args()
    pilot = args.pilot_dir.resolve()
    if not pilot.name.startswith("pilot_"):
        raise ValueError("Only development pilot directories are allowed")
    receipt = read_json(pilot / "run_receipt.json")
    if receipt.get("phase") != "pilot" or receipt.get("exit_code") != 0:
        raise ValueError("Completed successful development pilot receipt required")
    index = read_json(HERE / "inputs.json")
    allowed = set(index["development_evaluation_policy"]["development_source_groups"])
    rows = {r["source_id"]: r for r in index["rows"] if r["source_group"] in allowed}
    ids = [r["source_id"] for r in receipt["rows"]]
    if len(ids) != len(set(ids)) or not ids or not set(ids).issubset(rows):
        raise ValueError("Pilot receipt includes nondevelopment or duplicate source")
    dest = pilot / "diagnostics"
    dest.mkdir(exist_ok=False)
    outputs = [render_one(rows[name], pilot, dest / (name + "__diagnostic.png")) for name in ids]
    note = {"purpose": "development-only original-image geometry diagnosis, not evaluation",
            "pilot_receipt_sha256": sha(pilot / "run_receipt.json"),
            "input_index_sha256": sha(HERE / "inputs.json"), "script_sha256": sha(Path(__file__)),
            "source_count": len(outputs), "outputs": outputs,
            "source_images_not_modified": True, "old_predictions_not_modified": True,
            "candidate_lines_are_not_confirmed_material_identity": True}
    (dest / "diagnostic_receipt.json").write_text(json.dumps(note, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(outputs), "directory": str(dest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
