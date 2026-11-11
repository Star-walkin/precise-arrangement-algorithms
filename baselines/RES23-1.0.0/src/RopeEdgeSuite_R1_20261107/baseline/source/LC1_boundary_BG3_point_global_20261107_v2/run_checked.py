"""Recommended entry point: recover corrupt evidence from pixels, never use bad caches."""
import zipfile,json,time
import numpy as np
import inputs,run_bg3
from inputs import ROOT,BASE,dump
RAW_WORKER=run_bg3.worker

def checked_worker(i,image,out):
    try:return RAW_WORKER(i,image,out)
    except (zipfile.BadZipFile,EOFError) as error:
        # Recompute with the unchanged 01 extractor; it verifies the source
        # pixel hash, mask hash and exact raw-point coordinates before saving.
        legacy=inputs.LEGACY_CACHE
        inputs.LEGACY_CACHE=ROOT/'__force_fresh_extraction__'
        assert not inputs.LEGACY_CACHE.exists()
        try:
            old=json.loads((BASE/'frames'/f'{i:06d}.json').read_text(encoding='utf8'))
            data=inputs.extract(image,old)
        finally:inputs.LEGACY_CACHE=legacy
        assert data is not None
        target=ROOT/'cache'/f'{i:06d}.npz'
        if target.exists():target.rename(target.with_name(target.stem+f'.damaged_{time.time_ns()}.npz'))
        temp=target.with_suffix('.repair.npz');np.savez_compressed(temp,**data);temp.replace(target)
        result=RAW_WORKER(i,image,out)
        from pathlib import Path
        dump(Path(out)/f'cache_recovery_{i:06d}.json',dict(error=repr(error),
            source_pixel_sha256=old['source_pixel_sha256'],same_raw_points_verified=True,legacy_cache_modified=False))
        return result

if __name__=='__main__':
    run_bg3.worker=checked_worker
    run_bg3.main()
