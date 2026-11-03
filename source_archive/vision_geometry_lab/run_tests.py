"""Run independent offline tests without altering legacy test infrastructure."""
from pathlib import Path
import runpy
import sys
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    passed, failed = 0, []
    for path in sorted((Path(__file__).parent/'tests').glob('test_*.py')):
        try:
            namespace = runpy.run_path(str(path))
            for value in namespace.values():
                if isinstance(value,type) and issubclass(value,unittest.TestCase) and value is not unittest.TestCase:
                    suite = unittest.defaultTestLoader.loadTestsFromTestCase(value)
                    result = unittest.TextTestRunner(verbosity=1).run(suite)
                    passed += result.testsRun-len(result.errors)-len(result.failures)-len(result.skipped)
                    failed.extend(str(test) for test,_ in result.errors+result.failures)
            for name, test in namespace.items():
                if name.startswith('test_') and callable(test):
                    try:
                        test()
                        passed += 1
                    except Exception:
                        failed.append(f'{path.name}::{name}')
                        traceback.print_exc()
        except Exception:
            failed.append(str(path))
            traceback.print_exc()
    print(f'PASSED {passed}; FAILED {len(failed)}', flush=True)
    for item in failed:
        print(item)
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
