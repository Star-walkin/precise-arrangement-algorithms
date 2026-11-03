from pathlib import Path
import tempfile
from unittest.mock import patch

from vision_geometry_lab.v11_path_experiment import write_json, sha, require_gate, subset_manifest


def test_failed_or_mismatched_gate_blocks_expansion():
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)
        write_json(path/'protocol.json',{'fixture':True})
        write_json(path/'gate.json',{'allow_development_run':False,'protocol_sha256':sha(path/'protocol.json')})
        try: require_gate(path)
        except ValueError: pass
        else: raise AssertionError('failed gate bypassed')
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)
        write_json(path/'protocol.json',{'fixture':True})
        write_json(path/'gate.json',{'allow_development_run':True,'protocol_sha256':'wrong'})
        try: require_gate(path)
        except ValueError: pass
        else: raise AssertionError('foreign gate bypassed')


def test_outputs_cannot_overwrite_history():
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'result.json'
        write_json(path,{'first':True})
        old=path.read_bytes()
        try: write_json(path,{'second':True})
        except FileExistsError: pass
        else: raise AssertionError('overwrite allowed')
        assert path.read_bytes()==old


def test_single_scope_does_not_include_holdout_records():
    from vision_geometry_lab.v11_path_experiment import SAMPLE,read
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder); manifest=path/'source.json'
        write_json(manifest,{'samples':[{'sample_id':SAMPLE,'split':'train'},
            {'sample_id':'unopened_holdout','split':'holdout','image_path':'nonexistent'}]})
        with patch('vision_geometry_lab.v11_path_experiment.MANIFEST',manifest):
            subset=subset_manifest(path,'single')
        assert read(subset)['samples']==[{'sample_id':SAMPLE,'split':'train'}]
