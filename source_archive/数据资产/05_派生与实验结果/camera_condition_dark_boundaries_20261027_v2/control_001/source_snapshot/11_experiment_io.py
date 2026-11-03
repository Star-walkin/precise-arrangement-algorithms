"""活动绳离线实验的文件入口；只从已授权的固定400图清单取图。"""
from pathlib import Path
import csv
import hashlib
import json
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
ROOT = ASSETS.parents[1]
DATA = ASSETS / 'camera_condition_images_20261009_v2'
C3 = ASSETS / 'camera_condition_grouped_conic_20261026_v6/all_001'
C4 = ASSETS / 'camera_condition_full_surface_20261026_v7/all_001'
MANIFEST = DATA / 'manifest.csv'
QA = ASSETS / 'camera_condition_bright_rope_boundaries_20261009_v1/qa_plan.csv'
BASELINE = ROOT / '工程导航/05_实验版本档案/卷绳边界_整链基准_C3_20261026/baseline_manifest.json'
_INPUT_HASHES = None

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def dump(path, value):
    def convert(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, Path): return str(v)
        raise TypeError(type(v).__name__)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert,
                            allow_nan=False) + '\n', encoding='utf-8')

def read_image(path, flag=cv2.IMREAD_COLOR):
    a = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if a is None: raise ValueError('Image decode failed: ' + str(path))
    return a

def write_image(path, image):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(p.suffix, image,
                           [cv2.IMWRITE_JPEG_QUALITY, 94] if p.suffix == '.jpg' else [])
    if not ok: raise ValueError('Image encode failed: ' + str(path))
    data.tofile(str(p))

def select(mode):
    protocol = read_json(HERE / 'protocol.json')
    for p, key in [(MANIFEST, 'input_manifest_sha256'), (QA, 'qa_plan_sha256'),
                   (BASELINE, 'baseline_manifest_sha256')]:
        if sha(p) != protocol[key]: raise ValueError('Frozen input changed: ' + str(p))
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    with QA.open(encoding='utf-8-sig', newline='') as f:
        qa = {r['output_file'] for r in csv.DictReader(f)}
    assert len(rows) == 400 and len(qa) == 38
    assert sum(r['camera_role'] == 'left' for r in rows) == 200
    for row in rows:
        if ('20260925' in row['source_file'] + row['output_file'] or
            any(s in row['dataset_use_status'].lower() for s in ['reserved', 'heldout'])):
            raise ValueError('Reserved input is forbidden')
    return ([r for r in rows if r['output_file'] in qa] if mode == 'pilot' else rows), qa

def dependency_hash(path):
    global _INPUT_HASHES
    if _INPUT_HASHES is None:
        p = read_json(HERE / 'protocol.json')
        c3_manifest = C3 / 'integrity_audit.json'
        c4_manifest = C4 / 'output_manifest.json'
        if sha(c3_manifest) != p['c3_integrity_sha256'] or sha(c4_manifest) != p['c4_output_manifest_sha256']:
            raise ValueError('Parent output inventory changed')
        _INPUT_HASHES = {r['path']: r['sha256'] for r in read_json(c3_manifest)['file_manifest']}
        for case in read_json(c4_manifest)['cases']:
            for item in case['files']:
                _INPUT_HASHES[item['absolute_path']] = item['sha256']
    actual = sha(path)
    if _INPUT_HASHES.get(str(path)) != actual:
        raise ValueError('Parent prediction changed: ' + str(path))
    return actual

def inputs(row):
    rel = Path(row['output_file']); role = rel.parent.name; stem = rel.stem
    original = DATA / rel
    if sha(original) != row['output_sha256']: raise ValueError('Original PNG changed')
    surface_path = C4 / 'roi/surface_masks' / role / rel.name
    c3_path = C3 / 'boundaries/records' / role / (stem + '.json')
    c4_path = C4 / 'boundaries/records' / role / (stem + '.json')
    deps = {str(p): dependency_hash(p) for p in [surface_path, c3_path, c4_path]}
    image = read_image(original)
    surface = read_image(surface_path, 0)
    if image.shape != (3000, 4096, 3) or surface.shape != image.shape[:2]:
        raise ValueError('Unexpected source dimensions')
    return image, surface, read_json(c3_path), read_json(c4_path), deps

def source_inventory():
    # Explicit list: imports are pure image analysis, never device/UI entry points.
    source = [HERE / n for n in ['experiment_io.py', 'run_experiment.py',
              'render_results.py', 'free_segment.py', 'connection_core.py', 'protocol.json']]
    source += [ROOT / 'vision_geometry_lab/rope_detector.py',
               ROOT / 'vision_prototype/semantic_locator.py']
    models = []
    for folder, name in [('task_left_v1', 'left_camera_thick_v1'), ('task_v1', 'right_camera_poc_v1')]:
        directory = ROOT / 'vision_training' / folder / 'models' / name
        models += [directory / (name + '.onnx'), directory / 'onnx_metadata.json']
    return {str(p): sha(p) for p in source}, {str(p): sha(p) for p in models}
