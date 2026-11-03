"""Torch-free final-step evaluation for the V10 single train-image fit check."""
from pathlib import Path
import argparse
import hashlib
import json

import cv2
import numpy as np

from .centerline_benchmark_v5 import decode_heatmaps
from .centerline_quality_v5 import load_label_quality
from .centerline_review_v5 import validate_annotation, read_clean_image
from .evaluate_centerline_multiscale_v9 import independent_width, score_decoded
from .centerline_path_metrics_v6 import protocol_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'vision_geometry_lab/outputs/v10_single_fit'
SAMPLE_ID = 'dual_20260924_111500__left__f00246'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def evaluate(output_dir):
    output_dir = Path(output_dir).resolve()
    if (output_dir / 'evaluation.json').exists():
        raise FileExistsError('immutable final evaluation already exists')
    protocol_path = output_dir / 'protocol.json'
    protocol = json.loads(protocol_path.read_text('utf-8'))
    for path, expected in protocol['dependencies_sha256'].items():
        if file_hash(path) != expected:
            raise ValueError('frozen implementation changed')
    if file_hash(protocol['manifest']) != protocol['manifest_sha256']:
        raise ValueError('single-sample manifest changed')
    manifest = json.loads(Path(protocol['manifest']).read_text('utf-8'))
    if len(manifest['samples']) != 1:
        raise ValueError('single train sample required')
    sample = manifest['samples'][0]
    if sample['sample_id'] != SAMPLE_ID or sample['split'] != 'train':
        raise ValueError('forbidden evaluation sample')
    for key in ('image', 'annotation'):
        if file_hash(sample[key + '_path']) != protocol[key + '_sha256']:
            raise ValueError('source hash changed')
    label = validate_annotation(json.loads(Path(sample['annotation_path']).read_text('utf-8-sig')), sample)
    quality = load_label_quality(sample)
    if quality['quality_sha256'] != protocol['quality_sha256']:
        raise ValueError('quality hash changed')
    image = cv2.cvtColor(np.asarray(read_clean_image(sample)), cv2.COLOR_RGB2BGR)
    width = independent_width(image, label['polylines']['free_seed'])
    records, reports = {}, []
    for arm in protocol['arms']:
        report = json.loads((output_dir / arm / 'report.json').read_text('utf-8'))
        if report['protocol_sha256'] != file_hash(protocol_path) or report['steps_completed'] != 1200:
            raise ValueError('fixed final training report mismatch')
        path = Path(report['final_prediction'])
        if file_hash(path) != report['final_prediction_sha256']:
            raise ValueError('prediction hash changed')
        with np.load(path, allow_pickle=False) as data:
            body, center, coverage = (data[key] for key in ('body_probability', 'center_probability', 'coverage_mask'))
        if body.shape != (1500, 2048) or center.shape != body.shape or coverage.shape != body.shape:
            raise ValueError('prediction geometry mismatch')
        if not np.isin(coverage, (0, 1)).all() or np.any(body[coverage == 0]) or np.any(center[coverage == 0]):
            raise ValueError('uncovered grid must remain zero')
        decoded = decode_heatmaps(body, center, sample['roi_xyxy'], label['polylines']['free_seed'], width['width_px'])
        scored = score_decoded(label, quality, decoded, coverage, sample['image_shape_hw'])
        records[arm] = dict(decoded=decoded, score=scored,
            center_max=float(center.max()), body_max=float(body.max()),
            checkpoint_sha256=file_hash(report['final_checkpoint']))
        reports.append(report)
    paired_identical = {key: reports[0][key] == reports[1][key] for key in
        ('initial_state_sha256', 'sampler_sequence_sha256', 'input_contract', 'supervision_metadata')}
    if not all(paired_identical.values()):
        raise ValueError('paired training invariant failed')
    result = dict(version=1, sample_id=SAMPLE_ID, training_image_only=True,
        protocol_sha256=file_hash(protocol_path), v6_path_protocol_sha256=protocol_sha256(),
        independent_width=width, paired_identical=paired_identical, arms=records,
        conclusion_scope='optimization fit only; not generalization or physical-event accuracy',
        hardware_access=False)
    write_json(output_dir / 'evaluation.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    evaluate(parser.parse_args().output_dir)
