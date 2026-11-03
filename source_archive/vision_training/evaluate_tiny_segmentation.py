"""Evaluate and visually audit the first distilled segmentation model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from tiny_segmentation import TinyMultiLabelUNet


ROOT = Path(__file__).resolve().parent
CLASSES = ['wound_rope_surface', 'flange_inner_face', 'incoming_rope_free_span']
COLORS = np.array([[40, 220, 70], [30, 210, 240], [255, 145, 20]], dtype=np.uint8)


def load_record(row, manifest, include_truth=True):
    with Image.open(row['image']) as source:
        image = source.convert('RGB')
    width, height = image.size
    crop_norm = manifest['crop_xyxy_normalized']
    crop = (round(crop_norm[0] * width), round(crop_norm[1] * height),
            round(crop_norm[2] * width), round(crop_norm[3] * height))
    target_size = tuple(manifest['input_size_wh'])
    image = image.crop(crop).resize(target_size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    truth = None
    if include_truth:
        masks = []
        for class_name in CLASSES:
            with Image.open(Path(row['mask_dir']) / f'{class_name}.png') as mask_source:
                mask = mask_source.convert('L').crop(crop).resize(
                    target_size, Image.Resampling.NEAREST)
            masks.append(np.asarray(mask) >= 128)
        truth = np.stack(masks)
    return image, torch.from_numpy(array).unsqueeze(0), truth


def accumulate(metrics, prediction, truth):
    for class_index, class_name in enumerate(CLASSES):
        predicted = prediction[class_index]
        actual = truth[class_index]
        metrics[class_name]['intersection'] += int(np.logical_and(predicted, actual).sum())
        metrics[class_name]['union'] += int(np.logical_or(predicted, actual).sum())
        metrics[class_name]['predicted'] += int(predicted.sum())
        metrics[class_name]['truth'] += int(actual.sum())


def summarize(metrics):
    result = {}
    for class_name, values in metrics.items():
        intersection = values['intersection']
        union = max(values['union'], 1)
        denominator = max(values['predicted'] + values['truth'], 1)
        result[class_name] = {
            'iou': intersection / union,
            'dice': 2 * intersection / denominator,
        }
    result['macro_iou'] = sum(result[name]['iou'] for name in CLASSES) / len(CLASSES)
    return result


def blank_metrics():
    return {name: {'intersection': 0, 'union': 0, 'predicted': 0, 'truth': 0}
            for name in CLASSES}


def overlay(image, masks):
    base = np.asarray(image).copy()
    for index, color in enumerate(COLORS):
        selected = masks[index]
        base[selected] = (0.45 * base[selected] + 0.55 * color).astype(np.uint8)
    return Image.fromarray(base)


def labelled(image, title):
    canvas = Image.new('RGB', (image.width, image.height + 24), 'white')
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((6, 5), title, fill='black', font=ImageFont.load_default())
    return canvas


def selected_rows(rows):
    chosen = []
    for window_id in sorted({row['window_id'] for row in rows}):
        window = [row for row in rows if row['window_id'] == window_id]
        anchor = next(row for row in window if row['is_anchor'])
        for row in (window[0], anchor, window[-1]):
            if row not in chosen:
                chosen.append(row)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'task_v1' / 'distillation_dataset.json')
    parser.add_argument('--model-dir', type=Path,
                        default=ROOT / 'task_v1' / 'models' / 'right_camera_poc_v1')
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    manifest = json.loads(args.manifest.resolve().read_text(encoding='utf-8'))
    checkpoint = torch.load(model_dir / 'best.pt', map_location='cpu', weights_only=False)
    model = TinyMultiLabelUNet(base=checkpoint['base'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    validation_rows = [row for row in manifest['records'] if row['split'] == 'validation']
    all_metrics = blank_metrics()
    anchor_metrics = blank_metrics()
    predictions = {}
    end_to_end_times = []
    with torch.inference_mode():
        for row in validation_rows:
            started = time.perf_counter()
            image, tensor, truth = load_record(row, manifest)
            prediction = (torch.sigmoid(model(tensor))[0].numpy() >= .5)
            end_to_end_times.append((time.perf_counter() - started) * 1000)
            accumulate(all_metrics, prediction, truth)
            if row['is_anchor']:
                accumulate(anchor_metrics, prediction, truth)
            predictions[(row['window_id'], row['index'])] = (image, truth, prediction)

    # Measure model-forward latency separately from image decoding/preprocessing.
    _, latency_tensor, _ = load_record(validation_rows[0], manifest, include_truth=False)
    with torch.inference_mode():
        for _ in range(5):
            model(latency_tensor)
        samples = []
        for _ in range(30):
            started = time.perf_counter(); model(latency_tensor)
            samples.append((time.perf_counter() - started) * 1000)

    rows_for_sheet = selected_rows(validation_rows)
    tile_width, tile_height = manifest['input_size_wh'][0], manifest['input_size_wh'][1] + 24
    sheet = Image.new('RGB', (tile_width * 3, tile_height * len(rows_for_sheet)), (235, 235, 235))
    for row_index, row in enumerate(rows_for_sheet):
        image, truth, prediction = predictions[(row['window_id'], row['index'])]
        suffix = 'ANCHOR' if row['is_anchor'] else f"t={row['time_s']:.2f}s"
        tiles = [
            labelled(image, f'{row["window_id"][:9]} {suffix} SOURCE'),
            labelled(overlay(image, truth), 'PSEUDO / HUMAN-ANCHOR TRUTH'),
            labelled(overlay(image, prediction), 'TINY MODEL PREDICTION'),
        ]
        for column, tile in enumerate(tiles):
            sheet.paste(tile, (column * tile_width, row_index * tile_height))
    sheet_path = model_dir / 'validation_contact_sheet.jpg'
    sheet.save(sheet_path, quality=92)

    report = {
        'schema_version': 1,
        'status': 'proof_of_concept_not_for_device_control',
        'checkpoint_epoch': checkpoint['epoch'],
        'validation_frames': len(validation_rows),
        'human_anchor_frames_in_validation': sum(row['is_anchor'] for row in validation_rows),
        'validation_metrics_against_propagated_labels': summarize(all_metrics),
        'human_anchor_metrics': summarize(anchor_metrics),
        'latency_cpu_ms_model_only': {
            'mean': float(np.mean(samples)), 'median': float(np.median(samples)),
            'p95': float(np.percentile(samples, 95)),
        },
        'latency_cpu_ms_cached_end_to_end': {
            'mean': float(np.mean(end_to_end_times)), 'median': float(np.median(end_to_end_times)),
            'p95': float(np.percentile(end_to_end_times, 95)),
        },
        'contact_sheet': str(sheet_path.resolve()),
        'limitations': [
            'validation windows use the same camera and capture session as training',
            'only two validation frames are directly human-anchored',
            'model predicts coarse regions; original-resolution geometry must refine the tangent',
            'not independently validated for new lighting, occlusion, other winding thicknesses, or device control',
        ],
    }
    report_path = model_dir / 'evaluation_report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
