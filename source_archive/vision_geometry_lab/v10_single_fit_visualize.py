"""Post-hoc display only: same response scale/crop, no changed decoding."""
from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'vision_geometry_lab/outputs/v10_single_fit'


def render(output_dir):
    output_dir = Path(output_dir)
    target = output_dir / 'single_train_fit_comparison.png'
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads((output_dir / 'manifest.json').read_text('utf-8'))
    sample = manifest['samples'][0]
    evaluation = json.loads((output_dir / 'evaluation.json').read_text('utf-8'))
    label = json.loads(Path(sample['annotation_path']).read_text('utf-8-sig'))
    image = np.asarray(Image.open(sample['image_path']).convert('RGB'))
    crop = sample['source_roi_xyxy']  # Display only, never an inference ROI.
    x0, y0, x1, y1 = crop
    tile_w, tile_h = 840, round(840 * (y1-y0)/(x1-x0))
    canvas = Image.new('RGB', (tile_w*2, (tile_h+65)*2+42), '#111827')
    draw = ImageDraw.Draw(canvas)
    sx, sy = tile_w/(x1-x0), tile_h/(y1-y0)
    def points(line):
        return [((p[0]-x0)*sx, (p[1]-y0)*sy) for p in line]
    def panel(index, pixels, title, subtitle):
        px, py = (index % 2)*tile_w, (index // 2)*(tile_h+65)
        patch = Image.fromarray(pixels[y0:y1, x0:x1]).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas.paste(patch, (px, py+60))
        draw.text((px+10, py+8), title, fill='white')
        draw.text((px+10, py+30), subtitle, fill='#cbd5e1')
        return px, py+60
    panel(0, image, 'ONE TRAIN IMAGE: f246 / 24.6 s (not holdout)',
          'Original RGB; crop is display-only, inference used fixed seed-defined windows.')
    for index, arm in enumerate(('masked_bce', 'balanced_bce_dice'), 1):
        with np.load(output_dir / arm / 'final_prediction.npz', allow_pickle=False) as data:
            center = cv2.resize(data['center_probability'], (image.shape[1], image.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(center, 0, 1)*255),
                                             cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        pixels = np.uint8(.4*image.astype(float) + .6*heat.astype(float))
        panel(index, pixels, arm + ': center response at fixed step 1200',
              'Identical color scale 0..1; response is NOT calibrated probability or a decoded path.')
    px, py = panel(3, image, 'FIXED V5 DECODE / V6 5px scoring',
                   'Green: trusted reference | Cyan: BCE path | Magenta: balanced loss path')
    def path(line, color, width):
        if len(line) >= 2:
            draw.line([(px+x, py+y) for x, y in points(line)], fill=color, width=width)
    reference = evaluation['arms']['masked_bce']['score']['metrics']['score_support']['downstream_reference_segments']
    for segment in reference:
        path(segment, '#28ec63', 3)
    path(label['polylines']['free_seed'], '#ffdc40', 2)
    for arm, color in (('masked_bce', '#00ddff'), ('balanced_bce_dice', '#f52beb')):
        for segment in evaluation['arms'][arm]['decoded']['segments']:
            path(segment, color, 2)
    draw.text((10, canvas.height-31), 'Training fit diagnostic only. No generalization, physical contact, new-layer or force-time validation.',
              fill='white')
    canvas.save(target)
    print(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    render(parser.parse_args().output_dir)
