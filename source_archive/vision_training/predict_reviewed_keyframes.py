"""Run the distilled model on the 23 operator-reviewed keyframes."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from evaluate_tiny_segmentation import load_record, overlay
from tiny_segmentation import TinyMultiLabelUNet


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MANIFEST_PATH = ROOT / 'task_v1' / 'distillation_dataset.json'
MODEL_DIR = ROOT / 'task_v1' / 'models' / 'right_camera_poc_v1'
REVIEW_DIR = PROJECT / 'vision_prototype' / 'dataset_bootstrap' / 'batch_v1'
REVIEW_PATH = REVIEW_DIR / 'operator_review.json'
FRAME_DIR = REVIEW_DIR / 'frames'


def rope_angle(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    points = np.column_stack([xs, ys]).astype(np.float64)
    points -= points.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(points, full_matrices=False)
    vx, vy = axes[0]
    if vx < 0:
        vx, vy = -vx, -vy
    return float(math.degrees(math.atan2(vy, vx)))


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    review = json.loads(REVIEW_PATH.read_text(encoding='utf-8'))
    checkpoint = torch.load(MODEL_DIR / 'best.pt', map_location='cpu', weights_only=False)
    model = TinyMultiLabelUNet(base=checkpoint['base'])
    model.load_state_dict(checkpoint['model_state']); model.eval()
    rows = []
    tiles = []
    with torch.inference_mode():
        for filename, user in review['annotations'].items():
            row = {'image': str((FRAME_DIR / filename).resolve())}
            image, tensor, _ = load_record(row, manifest, include_truth=False)
            probabilities = torch.sigmoid(model(tensor))[0].numpy()
            masks = probabilities >= .5
            rows.append({
                'filename': filename,
                'time_s': user['time_s'],
                'user_original_algorithm_review': user,
                'area_fraction': {
                    name: float(masks[index].mean())
                    for index, name in enumerate(checkpoint['classes'])
                },
                'incoming_angle_deg': rope_angle(masks[2]),
                'incoming_mean_probability': float(probabilities[2][masks[2]].mean())
                if masks[2].any() else 0.0,
            })
            visual = overlay(image, masks).resize((384, 192), Image.Resampling.BILINEAR)
            tile = Image.new('RGB', (384, 216), 'white'); tile.paste(visual, (0, 24))
            title = f"{user['clip'][:6]} t={user['time_s']:.2f}s"
            ImageDraw.Draw(tile).text((5, 5), title, fill='black', font=ImageFont.load_default())
            tiles.append(tile)

    columns = 4; rows_count = math.ceil(len(tiles) / columns)
    sheet = Image.new('RGB', (384 * columns, 216 * rows_count), (230, 230, 230))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 384, (index // columns) * 216))
    sheet_path = MODEL_DIR / 'reviewed_keyframes_prediction_sheet.jpg'
    sheet.save(sheet_path, quality=92)
    report = {
        'schema_version': 1,
        'status': 'visual_screening_only_no_segmentation_ground_truth',
        'frame_count': len(rows),
        'sheet': str(sheet_path.resolve()),
        'frames': rows,
    }
    (MODEL_DIR / 'reviewed_keyframes_prediction_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'frames={len(rows)} sheet={sheet_path}')


if __name__ == '__main__':
    main()
