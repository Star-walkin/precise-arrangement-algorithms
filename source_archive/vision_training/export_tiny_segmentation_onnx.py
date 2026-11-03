"""Export the distilled right-camera model for OpenCV-DNN shadow inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch

from tiny_segmentation import TinyMultiLabelUNet


ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path,
                        default=ROOT / 'task_v1' / 'models' / 'right_camera_poc_v1')
    parser.add_argument('--onnx-name', default='right_camera_poc_v1.onnx')
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    source = model_dir / 'best.pt'
    destination = model_dir / args.onnx_name
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    model = TinyMultiLabelUNet(base=checkpoint['base'])
    model.load_state_dict(checkpoint['model_state']); model.eval()
    width, height = checkpoint['input_size_wh']
    example = torch.zeros(1, 3, height, width, dtype=torch.float32)
    torch.onnx.export(
        model, example, destination, input_names=['image_rgb_0_1'],
        output_names=['mask_logits'], opset_version=17, dynamo=False,
        do_constant_folding=True,
    )
    graph = onnx.load(destination)
    onnx.checker.check_model(graph)
    metadata = {
        'schema_version': 1,
        'status': 'shadow_mode_only_not_for_device_control',
        'source_checkpoint': str(source.resolve()),
        'onnx_file': str(destination.resolve()),
        'sha256': sha256(destination),
        'input_name': 'image_rgb_0_1',
        'input_layout': 'NCHW',
        'input_size_wh': checkpoint['input_size_wh'],
        'input_color': 'RGB',
        'input_range': [0.0, 1.0],
        'output_name': 'mask_logits',
        'classes': checkpoint['classes'],
        'threshold': 0.5,
        'crop_xyxy_normalized': checkpoint['crop_xyxy_normalized'],
    }
    (model_dir / 'onnx_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
