"""V8 trusted-tail cutoff integration without changing source annotations."""
from pathlib import Path
import os
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = ROOT / "vision_training/.venv/Scripts/python.exe"


def _torch_available():
    if not TRAIN_PY.is_file():
        return False
    try:
        result = subprocess.run(
            [str(TRAIN_PY), "-c", "import torch"], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _run_training_python(code):
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    result = subprocess.run(
        [str(TRAIN_PY), "-c", code], cwd=ROOT, env=environment,
        text=True, capture_output=True, timeout=120,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise AssertionError(
            f"command failed ({result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


@unittest.skipUnless(_torch_available(), "vision_training Python with torch is unavailable")
class CenterlineCutoffRasterIntegrationV8Tests(unittest.TestCase):
    def test_preclipped_supervision_is_tail_invariant_and_legacy_fallback_is_exact(self):
        code = r'''
import copy
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from vision_geometry_lab.centerline_context_v6 import rasterize_letterbox_record
from vision_geometry_lab.centerline_model_v5 import masked_two_head_loss, rasterize_manifest_record


height, width = 100, 200
trusted = {
    'centerline': np.asarray([[20., 50.], [80., 50.], [100., 50.]]),
    'edge_flange': np.asarray([[20., 45.], [80., 45.], [100., 45.]]),
    'edge_surface': np.asarray([[20., 55.], [80., 55.], [100., 55.]]),
}
cutoff = np.asarray([[100., 0.], [199., 0.], [199., 99.], [100., 99.]])


def record(image_path, tail_xy, *, trusted_targets=True, quality=True):
    tail_x, tail_y = map(float, tail_xy)
    raw = {
        'centerline': np.vstack([trusted['centerline'], [tail_x, tail_y]]),
        'edge_flange': np.vstack([trusted['edge_flange'], [tail_x, tail_y - 5.]]),
        'edge_surface': np.vstack([trusted['edge_surface'], [tail_x, tail_y + 5.]]),
        'free_seed': np.asarray([[20., 50.], [40., 50.]]),
    }
    result = {
        'sample_id': 'cutoff', 'group_id': 'fixture',
        'image_path': str(image_path), 'roi_xyxy': [0, 0, width, height],
        'label': {'status': 'partial'}, 'parsed_polylines': raw,
        'parsed_ignore_polygons': [],
        'parsed_quality_exclude_polygons': [cutoff] if quality else [],
    }
    if trusted_targets:
        # The compatibility field deliberately has only the three target curves.
        # RGB and the human free seed must continue to come from parsed_polylines.
        result['parsed_supervision_polylines'] = {
            key: value.copy() for key, value in trusted.items()
        }
    return result


with tempfile.TemporaryDirectory() as directory:
    image_path = Path(directory) / 'image.png'
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack([(xx + 20) % 255, (yy * 2 + 30) % 255,
                    np.full_like(xx, 100)], axis=2).astype(np.uint8)
    Image.fromarray(rgb, 'RGB').save(image_path)

    # Only the untrusted raw C8 direction and raw edge endpoints differ.
    first_record = record(image_path, [180., 50.])
    second_record = record(image_path, [180., 10.])
    first = rasterize_letterbox_record(first_record, [64, 128])
    second = rasterize_letterbox_record(second_record, [64, 128])
    keys = ('input', 'body_target', 'center_target',
            'body_valid_mask', 'center_valid_mask')
    for key in keys:
        assert torch.equal(first[key], second[key]), key

    # Quality and all target curves are forbidden model inputs.
    no_quality = rasterize_letterbox_record(
        record(image_path, [180., 50.], quality=False), [64, 128])
    assert torch.equal(first['input'], no_quality['input'])
    assert first['body_valid_mask'].sum() < no_quality['body_valid_mask'].sum()
    assert first['center_valid_mask'].sum() < no_quality['center_valid_mask'].sum()

    # A fixed prediction must see exactly the same loss/gradient after changing
    # only an excluded raw tail, and every invalid pixel must have zero gradient.
    torch.manual_seed(9)
    body_base = torch.randn(1, 1, 64, 128)
    center_base = torch.randn(1, 1, 64, 128)
    results = []
    for sample in (first, second):
        body_logits = body_base.clone().requires_grad_()
        center_logits = center_base.clone().requires_grad_()
        losses = masked_two_head_loss(
            {'body_logits': body_logits, 'center_logits': center_logits},
            sample['body_target'][None], sample['center_target'][None],
            body_valid_mask=sample['body_valid_mask'][None],
            center_valid_mask=sample['center_valid_mask'][None],
        )
        losses['loss'].backward()
        body_valid = sample['body_valid_mask'][None] > 0
        center_valid = sample['center_valid_mask'][None] > 0
        assert body_logits.grad[~body_valid].abs().sum().item() == 0
        assert center_logits.grad[~center_valid].abs().sum().item() == 0
        assert body_logits.grad[body_valid].abs().sum().item() > 0
        assert center_logits.grad[center_valid].abs().sum().item() > 0
        results.append((losses['loss'].detach(), body_logits.grad, center_logits.grad))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    assert torch.equal(results[0][2], results[1][2])

    # The original V5 square-resize path had the same pre-mask leakage risk and
    # must consume exactly the same pre-clipped three target curves.
    first_square = rasterize_manifest_record(first_record, 64)
    second_square = rasterize_manifest_record(second_record, 64)
    no_quality_square = rasterize_manifest_record(
        record(image_path, [180., 50.], quality=False), 64)
    for key in keys:
        assert torch.equal(first_square[key], second_square[key]), ('V5', key)
    assert torch.equal(first_square['input'], no_quality_square['input'])
    assert first_square['body_valid_mask'].sum() < no_quality_square['body_valid_mask'].sum()
    assert first_square['center_valid_mask'].sum() < no_quality_square['center_valid_mask'].sum()
    torch.manual_seed(10)
    body_base = torch.randn_like(first_square['body_target'][None])
    center_base = torch.randn_like(first_square['center_target'][None])
    square_results = []
    for sample in (first_square, second_square):
        body_logits = body_base.clone().requires_grad_()
        center_logits = center_base.clone().requires_grad_()
        losses = masked_two_head_loss(
            {'body_logits': body_logits, 'center_logits': center_logits},
            sample['body_target'][None], sample['center_target'][None],
            body_valid_mask=sample['body_valid_mask'][None],
            center_valid_mask=sample['center_valid_mask'][None],
        )
        losses['loss'].backward()
        body_valid = sample['body_valid_mask'][None] > 0
        center_valid = sample['center_valid_mask'][None] > 0
        assert body_logits.grad[~body_valid].abs().sum().item() == 0
        assert center_logits.grad[~center_valid].abs().sum().item() == 0
        square_results.append((losses['loss'].detach(), body_logits.grad, center_logits.grad))
    for index in range(3):
        assert torch.equal(square_results[0][index], square_results[1][index]), ('V5 loss', index)

    # Records without the new field retain byte-for-byte raster behaviour: an
    # explicit supervision copy equal to raw targets is exactly the legacy path.
    legacy_record = record(image_path, [180., 50.], trusted_targets=False)
    explicit_raw_record = copy.deepcopy(legacy_record)
    explicit_raw_record['parsed_supervision_polylines'] = {
        key: explicit_raw_record['parsed_polylines'][key].copy()
        for key in ('centerline', 'edge_flange', 'edge_surface')
    }
    legacy = rasterize_letterbox_record(legacy_record, [64, 128])
    explicit_raw = rasterize_letterbox_record(explicit_raw_record, [64, 128])
    for key in keys:
        assert torch.equal(legacy[key], explicit_raw[key]), key
    legacy_square = rasterize_manifest_record(legacy_record, 64)
    explicit_raw_square = rasterize_manifest_record(explicit_raw_record, 64)
    for key in keys:
        assert torch.equal(legacy_square[key], explicit_raw_square[key]), ('V5 legacy', key)
'''
        _run_training_python(code)

    def test_real_c07_c08_manifest_loads_exact_three_target_cutoffs_without_label_edits(self):
        code = r'''
import hashlib
import json
from pathlib import Path

import numpy as np

from vision_geometry_lab.centerline_model_v5 import inspect_manifest


manifest_path = Path('vision_geometry_lab/outputs/centerline_v7/review_batch01/manifest.json')
audit_path = Path('vision_geometry_lab/outputs/centerline_v8_intake_audit/audit.json')
inspection = inspect_manifest(manifest_path)
records = {record['sample_id']: record for record in inspection['usable']['train']}
audit = json.loads(audit_path.read_text('utf-8-sig'))
audit_rows = {row['sample_id']: row for row in audit['samples']}
selected = [identifier for identifier in records if identifier.startswith(('C07__', 'C08__'))]
assert len(selected) == 2, selected
for identifier in selected:
    record = records[identifier]
    targets = record['parsed_supervision_polylines']
    assert set(targets) == {'centerline', 'edge_flange', 'edge_surface'}
    assert len(targets['centerline']) == 7
    assert len(record['parsed_polylines']['centerline']) == 8
    assert 'free_seed' not in targets

    annotation_path = Path(record['annotation_path'])
    raw_annotation_hash = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    assert raw_annotation_hash == audit_rows[identifier]['annotation_sha256']
    assert raw_annotation_hash == record['annotation_sha256']

    matching = [region for region in record['quality_regions']
                if region['region_id'] == 'human_confirmed_after_center_c7']
    assert len(matching) == 1
    region = matching[0]
    origin = np.asarray(region['cutoff_origin_xy'], float)
    tangent = np.asarray(region['cutoff_unit_tangent_xy'], float)
    normal = np.asarray([-tangent[1], tangent[0]])
    assert np.allclose(targets['centerline'][-1], origin, rtol=0., atol=1e-7)
    endpoints = [np.asarray(targets[name][-1], float)
                 for name in ('edge_flange', 'edge_surface')]
    signed = [float((point-origin) @ tangent) for point in endpoints]
    offsets = [float((point-origin) @ normal) for point in endpoints]
    assert max(abs(value) for value in signed) <= 1e-7, signed
    assert offsets[0] * offsets[1] < 0., offsets
'''
        _run_training_python(code)


class CenterlineCutoffScoringIntegrationV8Tests(unittest.TestCase):
    def test_quality_halfplane_removes_unknown_tail_from_primary_scores(self):
        from vision_geometry_lab.centerline_path_metrics_v6 import evaluate_paths

        reference = [[20., 50.], [80., 50.], [100., 50.], [180., 10.]]
        predictions = [
            [[20., 50.], [80., 50.], [100., 50.], [180., 90.]],
            [[120., 20.], [180., 80.]],  # Entirely unknown; not a false positive.
        ]
        seed = [[20., 50.], [40., 50.]]
        cutoff = [[[100., 0.], [199., 0.], [199., 99.], [100., 99.]]]
        result = evaluate_paths(reference, predictions, seed, ignore_polygons=cutoff)

        expected_lengths = {'full_path': 80., 'downstream': 60.}
        for scope, expected_length in expected_lengths.items():
            metric = result[scope]
            self.assertAlmostEqual(metric['reference_length_px'], expected_length)
            self.assertAlmostEqual(metric['predicted_length_px'], expected_length)
            self.assertEqual(metric['covered_fraction'], 1.)
            self.assertEqual(metric['prediction_precision_at_5px'], 1.)
            self.assertEqual(metric['extra_predicted_fraction'], 0.)

        self.assertEqual(result['exclusion_polygon_count'], 1)
        self.assertFalse(result['quality_used_for_prediction_input'])
        self.assertEqual(result['score_support']['full_reference_segments'],
                         [[[20., 50.], [80., 50.], [100., 50.]]])
        self.assertEqual(result['score_support']['full_prediction_segments'],
                         [[[20., 50.], [80., 50.], [100., 50.]]])


if __name__ == '__main__':
    unittest.main()
