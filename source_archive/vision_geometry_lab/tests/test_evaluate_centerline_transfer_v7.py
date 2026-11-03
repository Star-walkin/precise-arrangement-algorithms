import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vision_geometry_lab.evaluate_centerline_transfer_v7 import (
    FACTORY_SHA256, validate_transfer_provenance,
)


def _fixture():
    sample = dict(sample_id='fixture', image_sha256='image', roi_xyxy=[0, 0, 64, 32])
    provenance = dict(backbone_trainable=False, input_semantics='rgb_only_no_seed_no_target',
                      strict_zero_features=True, source_checkpoint=None,
                      source_checkpoint_sha256=None, cache_npz=None, cache_sha256=None)
    meta = dict(experiment_version=7, feature_mode='zero_control', feature_provenance=provenance,
                checkpoint='head.pt', trained_checkpoint='head.pt', checkpoint_sha256='head',
                trained_checkpoint_sha256='head', model_input_hw=[32, 64], sam_embedding={'fixture': True})
    return meta, sample


def _raises(meta, sample):
    try:
        validate_transfer_provenance(meta, sample)
    except (ValueError, KeyError):
        return
    raise AssertionError('Invalid feature provenance was accepted')


def test_zero_control_valid_and_not_misrepresented_as_pretrained():
    meta, sample = _fixture()
    assert validate_transfer_provenance(meta, sample)['strict_zero_features']
    bad = copy.deepcopy(meta)
    bad['feature_provenance']['cache_npz'] = 'unexpected.npz'
    _raises(bad, sample)


def test_v7_guard_rejects_wrong_version_unfrozen_or_target_conditioned_features():
    meta, sample = _fixture()
    bad = copy.deepcopy(meta); bad['experiment_version'] = 6; _raises(bad, sample)
    bad = copy.deepcopy(meta); bad['feature_provenance']['backbone_trainable'] = True; _raises(bad, sample)
    bad = copy.deepcopy(meta); bad['feature_provenance']['input_semantics'] = 'rgb_plus_gt'; _raises(bad, sample)
    bad = copy.deepcopy(meta); bad['trained_checkpoint_sha256'] = 'different'; _raises(bad, sample)


def test_pretrained_guard_checks_cache_source_shapes_and_finite_values():
    meta, sample = _fixture()
    meta['feature_mode'] = 'sam2_frozen'
    provenance = meta['feature_provenance']
    provenance.update(strict_zero_features=False, source_checkpoint='factory.pt',
                      source_checkpoint_sha256=FACTORY_SHA256, extractor_code_sha256='code',
                      cache_sha256='cache', cache_key='key')
    cached = dict(sample_id=sample['sample_id'], image_sha256=sample['image_sha256'],
                  roi_xyxy=sample['roi_xyxy'], model_input_hw=[32, 64], cache_key='key',
                  source_checkpoint_sha256=FACTORY_SHA256, extractor_code_sha256='code',
                  input_semantics='rgb_only_no_seed_no_target', ground_truth_used=False,
                  free_seed_used=False, sam_embedding=meta['sam_embedding'])
    features = {f'feature_{i}': np.ones((c, 32//s, 64//s), np.float32)
                for i, (c, s) in enumerate(((32, 4), (64, 8), (256, 16)))}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'cache.npz'; provenance['cache_npz'] = str(path)
        def write():
            np.savez(path, metadata_json=np.array(json.dumps(cached)), **features)
        def fake_hash(filename):
            return FACTORY_SHA256 if str(filename) == 'factory.pt' else 'cache'
        with patch('vision_geometry_lab.evaluate_centerline_transfer_v7.file_hash', fake_hash):
            write()
            assert validate_transfer_provenance(meta, sample)['mode'] == 'sam2_frozen'
            cached['free_seed_used'] = True; write(); _raises(meta, sample)
            cached['free_seed_used'] = False; cached['image_sha256'] = 'wrong'; write(); _raises(meta, sample)
            cached['image_sha256'] = sample['image_sha256']; features['feature_0'][0, 0, 0] = np.nan
            write(); _raises(meta, sample)


def test_pretrained_guard_rejects_unapproved_backbone_before_opening_any_file():
    meta, sample = _fixture()
    meta['feature_mode'] = 'sam2_frozen'
    meta['feature_provenance'].update(strict_zero_features=False, source_checkpoint_sha256='legacy_field_model')
    _raises(meta, sample)
