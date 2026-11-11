import json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_checked as entry

class CacheRecoveryTests(unittest.TestCase):
    def test_bad_cache_is_preserved_and_recomputed_then_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'cache').mkdir();(root/'frames').mkdir();(root/'out').mkdir()
            (root/'frames/000002.json').write_text(json.dumps(dict(source_pixel_sha256='same-pixel')),encoding='utf8')
            target=root/'cache/000002.npz';target.write_bytes(b'broken original')
            legacy=root/'legacy';data=dict(points=np.arange(12).reshape(3,4))
            with patch.object(entry,'ROOT',root),patch.object(entry,'BASE',root),\
                 patch.object(entry.inputs,'LEGACY_CACHE',legacy),\
                 patch.object(entry.inputs,'extract',return_value=data) as extract,\
                 patch.object(entry,'RAW_WORKER',side_effect=[zipfile.BadZipFile('bad'),{'frame':2}]) as worker:
                result=entry.checked_worker(2,np.zeros((2,2,3),np.uint8),root/'out')
                self.assertEqual(result,{'frame':2});self.assertEqual(worker.call_count,2)
                self.assertEqual(extract.call_count,1);self.assertEqual(entry.inputs.LEGACY_CACHE,legacy)
            self.assertEqual(next((root/'cache').glob('*.damaged_*.npz')).read_bytes(),b'broken original')
            with np.load(target,allow_pickle=False) as f:self.assertTrue(np.array_equal(f['points'],data['points']))
            self.assertTrue((root/'out/cache_recovery_000002.json').exists())

    def test_failed_fresh_extraction_restores_lookup_and_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'frames').mkdir()
            (root/'frames/000002.json').write_text('{}',encoding='utf8');legacy=root/'legacy'
            with patch.object(entry,'ROOT',root),patch.object(entry,'BASE',root),\
                 patch.object(entry.inputs,'LEGACY_CACHE',legacy),\
                 patch.object(entry.inputs,'extract',side_effect=AssertionError('pixel mismatch')),\
                 patch.object(entry,'RAW_WORKER',side_effect=zipfile.BadZipFile('bad')):
                with self.assertRaisesRegex(AssertionError,'pixel mismatch'):entry.checked_worker(2,None,root)
                self.assertEqual(entry.inputs.LEGACY_CACHE,legacy)

if __name__=='__main__':unittest.main()
