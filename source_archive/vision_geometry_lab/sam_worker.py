"""Optional SAM2 worker: separate torch environment; no OpenCV dependency."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding='utf-8'))
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'vision_training/vendor/sam2'))
    import numpy as np
    from PIL import Image
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(min(4, torch.get_num_threads()))
    start = time.perf_counter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',
                      str(root/'vision_training/checkpoints/sam2.1_hiera_tiny.pt'),
                      device=device, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(model)
    points = np.asarray(request['points'], np.float32)
    labels = np.asarray(request.get('labels', [1]*len(points)), np.int32)
    if len(points) == 0 or not np.any(labels == 1):
        raise ValueError('At least one positive rope prompt is required.')
    with Image.open(request['image']) as source:
        rgb = np.asarray(source.convert('RGB'))
    # Crop to the requested free-span region if supplied. This gives thin ropes
    # more encoder pixels; no invented pixels or masks beyond this region.
    x0, y0, x1, y1 = 0, 0, rgb.shape[1], rgb.shape[0]
    if request.get('roi') is not None:
        r = request['roi']
        x0, y0 = max(0, int(r[0])), max(0, int(r[1]))
        x1, y1 = min(rgb.shape[1], int(r[2])), min(rgb.shape[0], int(r[3]))
    if x1-x0 < 8 or y1-y0 < 8:
        raise ValueError('SAM2 ROI is too small.')
    if np.any((points[:,0] < x0) | (points[:,0] >= x1) | (points[:,1] < y0) | (points[:,1] >= y1)):
        raise ValueError('All SAM2 prompts must lie inside the rope ROI.')
    points = points - [x0, y0]
    with torch.inference_mode():
        predictor.set_image(rgb[y0:y1, x0:x1].copy())
        masks, scores, _ = predictor.predict(point_coords=points,
                                             point_labels=labels,
                                             multimask_output=True)
    best = int(np.argmax(scores))
    full = np.zeros(rgb.shape[:2], np.uint8)
    full[y0:y1, x0:x1] = masks[best].astype(np.uint8)*255
    Image.fromarray(full).save(request['mask_output'])
    Path(request['metadata_output']).write_text(json.dumps(dict(
        device=device, score=float(scores[best]), scores=[float(v) for v in scores],
        elapsed_s=time.perf_counter()-start, use='proposal_only_not_precision_truth',
        roi=[x0,y0,x1,y1])), encoding='utf-8')


if __name__ == '__main__':
    main()
