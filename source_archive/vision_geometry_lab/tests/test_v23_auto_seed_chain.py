import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from vision_geometry_lab import v23_auto_seed_chain as m


def test_writes_do_not_overwrite():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'record.json'
        m.write(p, {'first': True})
        with pytest.raises(FileExistsError): m.write(p, {'second': True})
        assert m.read(p)=={'first': True}


def test_verify_detects_changed_source():
    with tempfile.TemporaryDirectory() as td:
        base=Path(td)
        p=base/'source.json';m.write(p, {'a': 1})
        m.write(base/'protocol.json', {'source_hashes': {str(p): m.sha(p)}})
        assert m.verify(base)
        # Change through a temporary fixture, not production files.
        p.write_text(json.dumps({'a': 2}), encoding='utf-8')
        with pytest.raises(ValueError, match='Frozen file changed'): m.verify(base)


def test_evaluation_requires_saved_predictions_before_labels():
    with tempfile.TemporaryDirectory() as td:
        base=Path(td);m.write(base/'protocol.json', {'source_hashes': {}})
        with patch('vision_geometry_lab.evaluate_centerline_multiscale_v9._dataset') as labels:
            with pytest.raises(FileNotFoundError): m.evaluate(base)
            labels.assert_not_called()


def test_prepare_has_no_annotation_open_contract():
    import inspect
    source=inspect.getsource(m.prepare)
    for forbidden in ('annotation_path', '_dataset', 'free_seed\']', 'seed_points='):
        assert forbidden not in source
    assert 'automatic_seed(rope, arc)' in source


def test_old_reports_and_detector_remain_read_only():
    import inspect
    source=inspect.getsource(m)
    assert "save(path, input=built['input']" in source
    assert "write(OLD/" not in source
    assert "write(FREE/" not in source
