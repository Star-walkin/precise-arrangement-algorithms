"""Create quantitative and visual QA for a SAM 2 anchor-label task."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
COLORS = {
    'wound_rope_surface': (35, 220, 100),
    'flange_inner_face': (0, 205, 255),
    'incoming_rope_free_span': (255, 145, 0),
}


def distance_to_mask(mask: np.ndarray, point: list[float]) -> float | None:
    if not mask.any():
        return None
    inverse = (~mask).astype(np.uint8)
    distance = cv2.distanceTransform(inverse, cv2.DIST_L2, 5)
    x = min(mask.shape[1] - 1, max(0, round(point[0])))
    y = min(mask.shape[0] - 1, max(0, round(point[1])))
    return float(distance[y, x])


def mask_metrics(mask: np.ndarray) -> dict:
    area = int(mask.sum())
    ys, xs = np.where(mask)
    if not area:
        return {'pixels': 0, 'area_fraction': 0.0, 'bbox': None,
                'components': 0, 'largest_component_fraction': 0.0}
    count, _labels, stats, _centers = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    return {
        'pixels': area,
        'area_fraction': area / mask.size,
        'bbox': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        'components': int(count - 1),
        'largest_component_fraction': float(component_areas.max() / area),
    }


def overlay_tile(task: dict, masks: dict, tangent: list[float], metrics: dict,
                 task_count: int) -> Image.Image:
    with Image.open(task['image']) as source:
        image = source.convert('RGBA')
    for name, mask in masks.items():
        alpha = Image.fromarray(mask.astype(np.uint8) * 95)
        layer = Image.new('RGBA', image.size, (*COLORS[name], 0))
        layer.putalpha(alpha)
        image = Image.alpha_composite(image, layer)
    draw = ImageDraw.Draw(image)
    x, y = tangent
    draw.ellipse((x-18, y-18, x+18, y+18), outline=(255, 0, 255, 255), width=9)
    target_width = 760
    target_height = round(image.height * target_width / image.width)
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS).convert('RGB')
    header = Image.new('RGB', (target_width, 62), '#17232f')
    header_draw = ImageDraw.Draw(header)
    summary = (f"{task['order']}/{task_count}  {task['clip']}  t={task['time_s']:.3f}s  "
               f"count={metrics.get('remaining_rope_count', '--')}  flags={len(metrics['flags'])}")
    header_draw.text((12, 8), summary, fill='white')
    header_draw.text((12, 34), ', '.join(metrics['flags']) if metrics['flags'] else 'automatic checks: OK',
                     fill='#ffcc80' if metrics['flags'] else '#a5d6a7')
    tile = Image.new('RGB', (target_width, header.height + image.height), 'black')
    tile.paste(header, (0, 0)); tile.paste(image, (0, header.height))
    return tile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', type=Path, default=ROOT / 'task_v1')
    parser.add_argument('--report', type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    task_dir = args.task_dir.resolve()
    labels = task_dir / 'labels'
    qa = task_dir / 'qa'
    manifest = json.loads((task_dir / 'manifest.json').read_text(encoding='utf-8'))
    report_md = args.report or (qa / 'anchor_quality_report.md')
    qa.mkdir(parents=True, exist_ok=True)
    results = []
    tiles = []
    for task in manifest['tasks']:
        folder = labels / Path(task['image_name']).stem
        annotation_path = folder / 'annotation.json'
        if not annotation_path.is_file():
            results.append({'image_name': task['image_name'], 'complete': False,
                            'flags': ['missing_annotation']})
            continue
        annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
        visibility = annotation.get('incoming_visibility', 'visible')
        masks = {}
        class_metrics = {}
        flags = []
        for name in COLORS:
            mask_path = folder / f'{name}.png'
            if not mask_path.is_file():
                if name != 'incoming_rope_free_span' or visibility == 'visible':
                    flags.append(f'{name}:missing')
                continue
            if name == 'incoming_rope_free_span' and visibility != 'visible':
                flags.append('incoming:mask_conflicts_with_visibility')
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert('L')) > 127
            masks[name] = mask
            class_metrics[name] = mask_metrics(mask)
            if class_metrics[name]['largest_component_fraction'] < 0.85:
                flags.append(f'{name}:fragmented')
        tangent = annotation.get('tangent_point')
        if tangent is None and visibility == 'visible':
            flags.append('missing_tangent')
        if tangent is not None and visibility != 'visible':
            flags.append('incoming:tangent_conflicts_with_visibility')
        if 'incoming_rope_free_span' in masks and tangent is not None:
            distance = distance_to_mask(masks['incoming_rope_free_span'], tangent)
            class_metrics['incoming_rope_free_span']['tangent_distance_px'] = distance
            if distance is not None and distance > 35:
                flags.append(f'incoming:tangent_far_{distance:.0f}px')
        if 'wound_rope_surface' in masks and tangent is not None:
            distance = distance_to_mask(masks['wound_rope_surface'], tangent)
            class_metrics['wound_rope_surface']['tangent_distance_px'] = distance
            if distance is not None and distance > 35:
                flags.append(f'wound:tangent_far_{distance:.0f}px')
        incoming_fraction = class_metrics.get('incoming_rope_free_span', {}).get('area_fraction', 0)
        if incoming_fraction > 0.02:
            flags.append(f'incoming:area_too_large_{incoming_fraction:.1%}')
        remaining = annotation.get('remaining_rope_count', task.get('remaining_rope_count'))
        row = {
            'image_name': task['image_name'],
            'clip': task['clip'], 'time_s': task['time_s'],
            'remaining_rope_count': remaining,
            'incoming_visibility': visibility,
            'complete': annotation.get('completed', False),
            'classes': class_metrics, 'tangent_point': tangent, 'flags': flags,
        }
        results.append(row)
        if len(masks) == len(COLORS) and tangent is not None:
            tiles.append(overlay_tile(task, masks, tangent, row, len(manifest['tasks'])))

    if tiles:
        gap = 12
        width = tiles[0].width * 2 + gap * 3
        rows = math.ceil(len(tiles) / 2)
        height = tiles[0].height * rows + gap * (rows + 1)
        sheet = Image.new('RGB', (width, height), '#273746')
        for index, tile in enumerate(tiles):
            x = gap + (index % 2) * (tile.width + gap)
            y = gap + (index // 2) * (tile.height + gap)
            sheet.paste(tile, (x, y))
        sheet.save(qa / 'anchor_contact_sheet.jpg', quality=92)

    payload = {'schema_version': 1, 'anchors': results}
    (qa / 'qa_report.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    complete = sum(row.get('complete', False) for row in results)
    flagged = sum(bool(row.get('flags')) for row in results)
    lines = [
        '# SAM 2锚点质量检查', '',
        f'- 任务目录：`{task_dir}`',
        f'- 完整标注：{complete}/{len(manifest["tasks"])}',
        f'- 自动几何检查有提示：{flagged}/{len(manifest["tasks"])}',
        '- 自动检查只能发现面积、碎片和切点距离异常；最终结论需结合叠加图人工复核。', '',
        '| 时间 | 本相机活动绳 | 人工根数 | 活动绳面积 | 活动绳到切点 | 卷绕面到切点 | 自动提示 |',
        '|---:|---|---:|---:|---:|---:|---|',
    ]
    for row in results:
        incoming = row.get('classes', {}).get('incoming_rope_free_span', {})
        wound = row.get('classes', {}).get('wound_rope_surface', {})
        lines.append(
            f"| {row.get('time_s', 0):.3f}s | {row.get('incoming_visibility', '--')} | "
            f"{row.get('remaining_rope_count', '--')} | "
            f"{incoming.get('area_fraction', 0):.3%} | "
            f"{incoming.get('tangent_distance_px', float('nan')):.1f}px | "
            f"{wound.get('tangent_distance_px', float('nan')):.1f}px | "
            f"{', '.join(row.get('flags', [])) or '无'} |")
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'anchors={len(results)} complete={complete} flagged={flagged}')
    print(qa / 'anchor_contact_sheet.jpg')


if __name__ == '__main__':
    main()
