import json
from pathlib import Path

import numpy as np
import pytest

from vision_geometry_lab.v25_v14_holdout import select_entries, load_input, safe, sha, write


def inventory():
    samples = [dict(sample_id=f's{i}', split='holdout') for i in range(8)]
    entries = {}
    for sample in samples:
        for variant in ('manual', 'auto'):
            key = sample['sample_id']+'::'+variant
            entries[key] = dict(key=key, **sample, variant=variant, available=variant == 'manual')
    return dict(entries=entries, evaluation_keys=list(entries)), samples


def test_all_eight_and_unavailable_kept():
    index, samples = inventory()
    rows = select_entries(index, samples)
    assert len(rows) == 16
    assert sum(not r['available'] for r in rows) == 8


def test_missing_variant_rejected():
    index, samples = inventory()
    index['evaluation_keys'].pop()
    with pytest.raises(ValueError):
        select_entries(index, samples)


def test_duplicate_variant_rejected():
    index, samples = inventory()
    index['evaluation_keys'].append(index['evaluation_keys'][0])
    with pytest.raises(ValueError):
        select_entries(index, samples)


def test_training_substitution_rejected():
    index, samples = inventory()
    index['entries']['s0::manual']['split'] = 'train'
    with pytest.raises(ValueError):
        select_entries(index, samples)


def test_absent_input_cannot_hide_artifact():
    assert load_input(dict(available=False, input_path=None, input_sha256=None)) is None
    with pytest.raises(ValueError):
        load_input(dict(available=False, input_path='hidden.npz'))


def test_supervision_array_rejected(tmp_path):
    path = tmp_path/'input.npz'
    np.savez(path, input=np.zeros((8,256,512),np.float32), valid_mask=np.ones((256,512)), gt=np.ones(1))
    with pytest.raises(ValueError, match='exclude supervision'):
        load_input(dict(available=True, input_path=str(path), input_sha256=sha(path)))


def test_hash_change_rejected(tmp_path):
    path = tmp_path/'input.npz'
    np.savez(path, input=np.zeros(2))
    with pytest.raises(ValueError, match='hash'):
        load_input(dict(available=True, input_path=str(path), input_sha256='0'*64))


def test_no_overwrite_or_blind_path(tmp_path):
    path = tmp_path/'x.json'
    write(path, dict(a=1))
    with pytest.raises(FileExistsError):
        write(path, dict(a=2))
    assert json.loads(path.read_text()) == dict(a=1)
    with pytest.raises(ValueError):
        safe('D:/video/dual_20260925_01/frame.png')
