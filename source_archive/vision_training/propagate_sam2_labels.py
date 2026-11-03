"""Propagate the three verified anchor masks through short sampled windows."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from sam2.build_sam import build_sam2_video_predictor


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'checkpoints' / 'sam2.1_hiera_tiny.pt'
CONFIG = 'configs/sam2.1/sam2.1_hiera_t.yaml'
CLASSES = {
    1: 'wound_rope_surface',
    2: 'flange_inner_face',
    3: 'incoming_rope_free_span',
}


def simple_metrics(mask: np.ndarray) -> dict:
    ys, xs = np.where(mask)
    if not len(xs):
        return {'pixels': 0, 'area_fraction': 0.0, 'bbox': None}
    return {
        'pixels': int(len(xs)),
        'area_fraction': float(len(xs) / mask.size),
        'bbox': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', type=Path, default=ROOT / 'task_v1')
    parser.add_argument('--complete-rope-to-anchor-x', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    task_dir = args.task_dir.resolve()
    window_root = task_dir / 'propagation_windows'
    manifest = json.loads((window_root / 'manifest.json').read_text(encoding='utf-8'))
    complete_rope_span = None
    if args.complete_rope_to_anchor_x:
        from rope_span_completion import complete_rope_span
    predictor = build_sam2_video_predictor(
        CONFIG, str(CHECKPOINT), device='cpu', apply_postprocessing=False)
    overall_start = time.perf_counter()
    for number, window in enumerate(manifest['windows'], 1):
        started = time.perf_counter()
        window_dir = window_root / window['window_id']
        report_path = window_dir / 'propagation_report.json'
        if args.resume and report_path.is_file():
            print(f'[{number}/{len(manifest["windows"])}] {window["window_id"]}: already complete')
            continue
        output_dir = window_dir / 'pseudo_labels'
        output_dir.mkdir(exist_ok=True)
        state = predictor.init_state(
            video_path=window['frame_dir'], offload_video_to_cpu=True,
            offload_state_to_cpu=True, async_loading_frames=False)
        anchor = int(window['anchor_index'])
        anchor_dir = Path(window['anchor_label_dir'])
        annotation = json.loads((anchor_dir / 'annotation.json').read_text(encoding='utf-8'))
        anchor_tangent_x = float(annotation['tangent_point'][0])
        with torch.inference_mode():
            for obj_id, class_name in CLASSES.items():
                with Image.open(anchor_dir / f'{class_name}.png') as mask_image:
                    mask = np.asarray(mask_image.convert('L')) > 127
                predictor.add_new_mask(state, frame_idx=anchor, obj_id=obj_id, mask=mask)

        outputs: dict[int, dict[int, np.ndarray]] = {}

        def consume(reverse: bool, maximum: int):
            if maximum <= 0:
                return
            with torch.inference_mode():
                context = nullcontext()
                with context:
                    iterator = predictor.propagate_in_video(
                        state, start_frame_idx=anchor,
                        max_frame_num_to_track=maximum, reverse=reverse)
                    for frame_idx, object_ids, mask_logits in iterator:
                        frame_output = outputs.setdefault(int(frame_idx), {})
                        for position, obj_id in enumerate(object_ids):
                            frame_output[int(obj_id)] = (
                                mask_logits[position, 0].detach().cpu().numpy() > 0)

        consume(reverse=True, maximum=anchor + 1)
        consume(reverse=False, maximum=int(window['frame_count']) - anchor)

        frame_rows = []
        for frame in window['frames']:
            frame_idx = int(frame['index'])
            frame_folder = output_dir / f'{frame_idx:05d}'
            frame_folder.mkdir(exist_ok=True)
            classes = {}
            frame_masks = outputs.get(frame_idx, {})
            completion = None
            if args.complete_rope_to_anchor_x and 1 in frame_masks and 3 in frame_masks:
                completed, completion = complete_rope_span(
                    frame_masks[3], frame_masks[1], (anchor_tangent_x, float('nan')))
                frame_masks = dict(frame_masks)
                frame_masks[3] = completed
            for obj_id, class_name in CLASSES.items():
                mask = frame_masks.get(obj_id)
                if mask is None:
                    classes[class_name] = {'missing': True}
                    continue
                Image.fromarray(mask.astype(np.uint8) * 255).save(
                    frame_folder / f'{class_name}.png', optimize=True)
                classes[class_name] = simple_metrics(mask)
            frame_rows.append({**frame, 'classes': classes,
                               'rope_completion': (completion.__dict__ if completion else None)})
        elapsed = time.perf_counter() - started
        result = {
            'schema_version': 1,
            'window_id': window['window_id'],
            'anchor_index': anchor,
            'elapsed_s': elapsed,
            'frames': frame_rows,
        }
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        predictor.reset_state(state)
        print(f'[{number}/{len(manifest["windows"])}] {window["window_id"]}: '
              f'{len(frame_rows)} frames in {elapsed:.1f}s')
    print(f'All windows finished in {time.perf_counter()-overall_start:.1f}s')


if __name__ == '__main__':
    main()
