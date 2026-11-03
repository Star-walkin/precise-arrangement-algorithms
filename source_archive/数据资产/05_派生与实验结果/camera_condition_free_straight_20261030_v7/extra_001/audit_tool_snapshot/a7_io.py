"""A7固定开放400图的只读入口；不加载F3图像、曲线、A6路径或标签。"""
from pathlib import Path
import csv
import hashlib
import json
import os
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
DATA = ASSETS / 'camera_condition_images_20261009_v2'
C3 = ASSETS / 'camera_condition_grouped_conic_20261026_v6/all_001'
C4 = ASSETS / 'camera_condition_full_surface_20261026_v7/all_001'
A1 = ASSETS / 'camera_condition_incoming_rope_20261026_v1/all_001'
A2 = ASSETS / 'camera_condition_incoming_rope_20261026_v2/all_001'
VERSION = 'IR-A7.0.0'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, Path): return str(item)
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=convert, allow_nan=False) + '\n', encoding='utf-8')


def read_image(path, flag=1):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if image is None:
        raise ValueError('Cannot decode: ' + str(path))
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError('Cannot encode: ' + str(path))
    encoded.tofile(str(path))


def selection(mode):
    if mode not in ('pilot', 'extra', 'all'):
        raise ValueError('Unknown fixed input selection')
    protocol = read_json(HERE / 'protocol.json')
    manifest = DATA / 'manifest.csv'
    if sha(manifest) != protocol['input_manifest_sha256']:
        raise ValueError('Original input manifest changed')
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 400 or any(sum(r['camera_role'] == role for r in rows) != 200
                               for role in ('left', 'right')):
        raise ValueError('Expected fixed left200 + right200 denominator')
    seen = set()
    for row in rows:
        relative = Path(row['output_file'])
        if (relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 3
                or relative.parts[0] != 'images'
                or relative.parts[1] != row['camera_role'] + '_camera'
                or relative.suffix.lower() != '.png' or row['output_file'] in seen):
            raise ValueError('Unsafe or duplicate original input path')
        if ('20260925' in row['source_file'] + row['output_file']
                or any(word in row['dataset_use_status'].lower() for word in ('reserved', 'heldout'))):
            raise ValueError('Reserved data are not allowed')
        seen.add(row['output_file'])
    if mode == 'all':
        return rows
    requested = protocol['pilot_inputs' if mode == 'pilot' else 'extra_review_inputs']
    if len(requested) != len(set(requested)) or not set(requested).issubset(seen):
        raise ValueError('Fixed development list is invalid')
    # Preserve the original manifest order, independently of worker scheduling.
    return [row for row in rows if row['output_file'] in requested]


def _key(path):
    return os.path.normcase(os.path.abspath(str(path)))


_HASHES = None


def checked(path):
    """Only decode the four parent manifests needed for A7's allowed inputs.

    Other frozen manifest files may be byte-hashed as historical evidence, but
    their F3/A6 predictions are never loaded into the detector's input dictionary.
    """
    global _HASHES
    if _HASHES is None:
        _HASHES = {}
        allowed = {_key(C3 / 'integrity_audit.json'), _key(C4 / 'output_manifest.json'),
                   _key(A1 / 'output_manifest.json'), _key(A2 / 'output_manifest.json')}
        protocol = read_json(HERE / 'protocol.json')
        found = set()
        for filename, expected in protocol['parent_manifest_sha256'].items():
            manifest = Path(filename)
            if sha(manifest) != expected:
                raise ValueError('Frozen parent manifest changed: ' + filename)
            if _key(manifest) not in allowed:
                continue
            found.add(_key(manifest))
            data = read_json(manifest)
            files = ([item for case in data['cases'] for item in case['files']]
                     if 'cases' in data else data.get('files', data.get('file_manifest', [])))
            for item in files:
                filename = item.get('absolute_path') or item.get('path', item.get('relative_path'))
                target = Path(filename)
                if not target.is_absolute(): target = manifest.parent / target
                old = _HASHES.get(_key(target))
                if old is not None and old != item['sha256']:
                    raise ValueError('Conflicting frozen input hash: ' + str(target))
                _HASHES[_key(target)] = item['sha256']
        if found != allowed:
            raise ValueError('Missing allowed parent manifest in protocol')
    actual = sha(path)
    if _HASHES.get(_key(path)) != actual:
        raise ValueError('Parent input is not frozen or has changed: ' + str(path))
    return actual


def input_paths(row):
    relative = Path(row['output_file'])
    role, stem = relative.parent.name, relative.stem
    return {
        'original': DATA / relative,
        'surface': C4 / 'roi/surface_masks' / role / relative.name,
        'a1_record': A1 / 'records' / role / (stem + '.json'),
        'a2_record': A2 / 'records' / role / (stem + '.json'),
        'pitch_record': C3 / 'boundaries/records' / role / (stem + '.json'),
    }


def inputs(row):
    paths = input_paths(row)
    dependencies = {str(path): checked(path) for key, path in paths.items() if key != 'original'}
    original_hash = sha(paths['original'])
    if original_hash != row['output_sha256']:
        raise ValueError('Original image changed')
    dependencies[str(paths['original'])] = original_hash
    original = read_image(paths['original'])
    surface = read_image(paths['surface'], 0)
    if original.dtype != np.uint8 or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError('Expected uint8 BGR original')
    if surface.shape != original.shape[:2] or surface.dtype != np.uint8:
        raise ValueError('Surface and original dimensions differ')
    # C3's full curve list is intentionally not passed to the algorithm.
    pitch = float(read_json(paths['pitch_record'])['record']['pitch_px'])
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError('Invalid saved pixel pitch')
    return {'original': original, 'surface': surface,
            'a1_record': read_json(paths['a1_record']),
            'a2_record': read_json(paths['a2_record']), 'pitch': pitch,
            'input_dependency_sha256': dependencies}
