"""Screen propagated pseudo-labels and create a compact first/anchor/last contact sheet."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
COLORS = {
    'wound_rope_surface': (35, 220, 100),
    'flange_inner_face': (0, 205, 255),
    'incoming_rope_free_span': (255, 145, 0),
}


def load_mask(folder: Path, class_name: str) -> np.ndarray:
    with Image.open(folder / f'{class_name}.png') as image:
        return np.asarray(image.convert('L')) > 127


def connected_fraction(mask: np.ndarray) -> float:
    count, _labels, stats, _centers = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    if count <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float(areas.max() / max(1, areas.sum()))


def line_angle(mask: np.ndarray) -> float | None:
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return None
    vx, vy, _x0, _y0 = cv2.fitLine(
        np.column_stack([xs, ys]).astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
    return math.degrees(math.atan2(float(vy), float(vx)))


def angle_delta(a: float, b: float) -> float:
    return abs((a - b + 90) % 180 - 90)


def overlay(frame_path: Path, mask_folder: Path, header: str, flags: list[str]) -> Image.Image:
    with Image.open(frame_path) as source:
        image = source.convert('RGBA')
    for name, color in COLORS.items():
        mask = load_mask(mask_folder, name)
        alpha = Image.fromarray(mask.astype(np.uint8) * 95)
        layer = Image.new('RGBA', image.size, (*color, 0)); layer.putalpha(alpha)
        image = Image.alpha_composite(image, layer)
    width = 640
    height = round(image.height * width / image.width)
    image = image.resize((width, height), Image.Resampling.LANCZOS).convert('RGB')
    title = Image.new('RGB', (width, 58), '#17232f')
    draw = ImageDraw.Draw(title)
    draw.text((10, 7), header, fill='white')
    draw.text((10, 31), ', '.join(flags) if flags else 'screening: OK',
              fill='#ffcc80' if flags else '#a5d6a7')
    tile = Image.new('RGB', (width, title.height + image.height), 'black')
    tile.paste(title, (0, 0)); tile.paste(image, (0, title.height))
    return tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', type=Path, default=ROOT / 'task_v1')
    parser.add_argument('--mask-subdir', default='pseudo_labels')
    parser.add_argument('--output-name', default='propagation_qa')
    args = parser.parse_args()
    task_dir = args.task_dir.resolve()
    window_root = task_dir / 'propagation_windows'
    output = task_dir / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((window_root / 'manifest.json').read_text(encoding='utf-8'))
    screened = []
    tiles = []
    per_window = []
    for window in source_manifest['windows']:
        window_dir = window_root / window['window_id']
        report = json.loads((window_dir / 'propagation_report.json').read_text(encoding='utf-8'))
        anchor = report['frames'][report['anchor_index']]
        anchor_mask_folder = window_dir / args.mask_subdir / f"{report['anchor_index']:05d}"
        anchor_areas = {name: float(load_mask(anchor_mask_folder, name).mean())
                        for name in COLORS}
        anchor_angle = line_angle(load_mask(anchor_mask_folder, 'incoming_rope_free_span'))
        accepted = 0
        frame_results = []
        for row in report['frames']:
            index = int(row['index'])
            mask_folder = window_dir / args.mask_subdir / f'{index:05d}'
            flags = []
            ratios = {}
            for name in COLORS:
                mask_path = mask_folder / f'{name}.png'
                if not mask_path.is_file():
                    flags.append(f'{name}:missing')
                    continue
                mask = load_mask(mask_folder, name)
                ratio = float(mask.mean()) / max(anchor_areas[name], 1e-9)
                ratios[name] = ratio
                low, high = ((.90, 1.10) if name != 'incoming_rope_free_span' else (.70, 1.35))
                if not low <= ratio <= high:
                    flags.append(f'{name}:area_ratio_{ratio:.2f}')
                component = connected_fraction(mask)
                # A foreground guide/actuator can legitimately split the
                # visible flange into two pieces.  Rope and wound surface stay
                # strict because their connectivity is part of the geometry.
                minimum_component = .80 if name == 'flange_inner_face' else .85
                if component < minimum_component:
                    flags.append(f'{name}:fragmented_{component:.2f}')
            incoming_angle = line_angle(load_mask(mask_folder, 'incoming_rope_free_span'))
            if incoming_angle is None or anchor_angle is None:
                flags.append('incoming:angle_missing')
                delta = None
            else:
                delta = angle_delta(incoming_angle, anchor_angle)
                if delta > 3.0:
                    flags.append(f'incoming:angle_delta_{delta:.1f}')
            candidate = not flags
            accepted += int(candidate)
            result = {
                'window_id': window['window_id'], 'index': index,
                'source_frame': row['source_frame'], 'time_s': row['time_s'],
                'is_anchor': row['is_anchor'], 'candidate': candidate,
                'area_ratios': ratios, 'incoming_angle_deg': incoming_angle,
                'incoming_angle_delta_deg': delta, 'flags': flags,
                'frame': str((Path(window['frame_dir']) / f'{index:05d}.jpg').resolve()),
                'mask_dir': str(mask_folder.resolve()),
            }
            screened.append(result); frame_results.append(result)
        per_window.append({
            'window_id': window['window_id'], 'frames': len(frame_results),
            'candidates': accepted, 'flagged': len(frame_results)-accepted,
        })
        selected = sorted(set([0, report['anchor_index'], len(report['frames'])-1]))
        by_index = {row['index']: row for row in frame_results}
        for index in selected:
            row = by_index[index]
            tiles.append(overlay(
                Path(row['frame']), Path(row['mask_dir']),
                f"{window['window_id'][:18]}  idx={index}  t={row['time_s']:.2f}s",
                row['flags']))

    columns = 3
    gap = 10
    if tiles:
        rows = math.ceil(len(tiles) / columns)
        width = columns * tiles[0].width + (columns + 1) * gap
        height = rows * tiles[0].height + (rows + 1) * gap
        sheet = Image.new('RGB', (width, height), '#273746')
        for i, tile in enumerate(tiles):
            x = gap + (i % columns) * (tile.width + gap)
            y = gap + (i // columns) * (tile.height + gap)
            sheet.paste(tile, (x, y))
        sheet.save(output / 'propagation_contact_sheet.jpg', quality=91)

    payload = {'schema_version': 1, 'windows': per_window, 'frames': screened}
    (output / 'screened_manifest.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    total_candidates = sum(row['candidates'] for row in per_window)
    total_frames = sum(row['frames'] for row in per_window)
    lines = [
        '# 第四阶段SAM 2视频传播检查', '',
        f'- 短窗口：{len(per_window)}个，每个锚点前后约3秒、5 fps。',
        f'- 已传播帧：{total_frames}张。',
        f'- 通过面积、连通性和活动绳角度初筛：{total_candidates}/{total_frames}张。',
        '- “通过初筛”仍是候选伪标签，不等同于人工真值；训练前保留窗口抽样人工检查。', '',
        '| 窗口 | 帧数 | 候选 | 自动剔除 |', '|---|---:|---:|---:|',
    ]
    for row in per_window:
        lines.append(f"| {row['window_id']} | {row['frames']} | {row['candidates']} | {row['flagged']} |")
    (output / 'propagation_quality_report.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print(f'frames={total_frames} candidates={total_candidates} flagged={total_frames-total_candidates}')
    print(output / 'propagation_contact_sheet.jpg')


if __name__ == '__main__':
    main()
