"""Repeat small deterministic contracts; write a reviewable local receipt."""
from pathlib import Path
import ast
import hashlib
import importlib.metadata
import json
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    started=time.perf_counter()
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    suite=unittest.defaultTestLoader.discover(str(ROOT),pattern='test_lc1*.py')
    with (ROOT/'verification_tests.txt').open('w',encoding='utf-8') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    packages={name:importlib.metadata.version(name) for name in
              ['numpy','opencv-python-headless','scipy','Pillow','imageio-ffmpeg','torch']}
    (ROOT/'requirements-tested.txt').write_text('\n'.join(f'{name}=={version}' for name,version in packages.items())+'\n',encoding='utf-8')
    receipt=dict(success=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),
                 errors=len(result.errors),skipped=len(result.skipped),python=sys.version,
                 packages=packages,wall_seconds=time.perf_counter()-started,
                 scope='geometry/state machine, live routing, source/input contracts and independent straight-evidence adapters; not physical accuracy or realtime validation',
                 test_files={p.name:sha(p) for p in ROOT.glob('test_lc1_*.py')},
                 source_files={p.name:sha(p) for p in ROOT.glob('lc1_*.py')})
    (ROOT/'verification_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(receipt,indent=2,ensure_ascii=False))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
