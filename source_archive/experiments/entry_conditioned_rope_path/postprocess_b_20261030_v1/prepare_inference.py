"""Reproduce the frozen B network inputs and preserve numeric center responses.

This is an offline inference-only helper. It never reads target annotations,
trains a model, imports device code, or changes earlier experiment outputs.
Use the OpenCV environment for ``prepare`` and the PyTorch one for ``run``.
The 292 source images and 299 separate entry conditions are inherited exactly
from the user's A/B gallery. The remaining 108 open400 images had no saved
entry and are retained as upstream failures, not classified as absent rope.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parents[1]
UPSTREAM = CODE / 'batch_ab_20261030_001'
DATA = ROOT / '数据资产/05_派生与实验结果/camera_condition_images_20261009_v2'
VERSION = 'IR-H1.0.0'
# Chosen before new inference: six images per camera, the requested tool/multi
# entry case, straight f15, rod-occluded Rrun55, and major acquisition groups.
PILOT_IDS = [
    'L_run_20260923_01__f000015__left',
    'L_run_20260923_01__f000142__left',
    'L_run_stop_run_20260922_01__f001719__left',
    'L_static_20260922_01__f000100__left',
    'dual_20260924_111500__pair0015__left',
    'ruler_left_05',
    'dual_20260923_101517__pair0075__right',
    'R_run_20260923_01__f000055__right',
    'R_run_stop_run_20260922_01__f002521__right',
    'dual_20260923_101643__pair0375__right',
    'dual_20260923_113305__pair0555__right',
    'dual_20260924_111143__pair0075__right',
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def verify_hashes(rows):
    for row in rows:
        assert sha(row['path']) == row['sha256'], row['path']


def initialize_protocol():
    """Freeze the explicit whitelist before decoding any new input image."""
    dest = HERE / 'protocol_inputs.json'
    if dest.exists():
        protocol = read_json(dest)
        verify_hashes(protocol['frozen_sources'])
        assert sha(__file__) == protocol['inference_helper_sha256']
        return protocol
    old = read_json(UPSTREAM / 'jobs.json')
    receipt = read_json(UPSTREAM / 'receipt.json')
    rows = list(csv.DictReader((DATA / 'manifest.csv').open(encoding='utf-8-sig')))
    assert len(rows) == old['source_count'] == 400
    assert sha(DATA / 'manifest.csv') == old['source_manifest_sha256']
    assert len(old['jobs']) == receipt['count'] == 292
    assert len(old['skipped']) == 108
    assert sum(len(j['seeds']) for j in old['jobs']) == 299
    assert sha(CODE / 'batch_ab_preview.py') == old['source_sha256'] == receipt['source_sha256']
    assert old['pair']['main_comparison_step'] == 750
    model = old['pair']['main_checkpoints']['B']
    assert sha(model['path']) == model['sha256']
    lookup = {Path(row['output_file']).stem: row for row in rows}
    jobs = []
    for item in old['jobs']:
        row = lookup[item['id']]
        assert '20260925' not in row['source_file'] + row['output_file']
        assert not any(x in row['dataset_use_status'].lower() for x in ('reserved', 'heldout'))
        original = (DATA / row['output_file']).resolve()
        assert original == Path(item['image_path']).resolve()
        assert item['image_sha256'] == row['output_sha256']
        assert item['bbox'] == [0, 1050, 4096, 3000] and item['shape'] == [3000, 4096]
        seed_sources = [{'path': seed['source'], 'sha256': sha(seed['source'])} for seed in item['seeds']]
        jobs.append({k: item[k] for k in ('id', 'image_path', 'image_sha256', 'bbox', 'shape', 'seeds')} |
                    {'camera_role': row['camera_role'], 'source_group': row['source_group'],
                     'source_time_s': row['source_time_s'], 'dataset_use_status': row['dataset_use_status'],
                     'seed_source_hashes': seed_sources})
    assert len(set(PILOT_IDS)) == 12 and set(PILOT_IDS) <= {j['id'] for j in jobs}
    assert sum(j['camera_role'] == 'left' for j in jobs if j['id'] in PILOT_IDS) == 6
    frozen_paths = [UPSTREAM / 'jobs.json', UPSTREAM / 'receipt.json', DATA / 'manifest.csv',
                    Path(old['entry_selection_source']), Path(model['path'])] + [
                        CODE / name for name in ('batch_ab_preview.py', 'batch_preview.py', 'model.py', 'train.py', 'prepare.py')]
    protocol = {
        'name': '热图接续 H1', 'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'same user A/B gallery; 292 open development images and 299 automatically generated entry conditions',
        'source_count': 400, 'eligible_count': 292, 'entry_condition_count': 299,
        'upstream_skipped_count': 108, 'upstream_skipped': old['skipped'],
        'skipped_is_not_absent_rope': True, 'pilot_ids': PILOT_IDS,
        'pilot_selection': 'fixed before H1 inference, six per camera, major groups, tools, occlusion and multi-entry',
        'model_B': model, 'checkpoint_step': 750,
        'input_shape_hw': [975, 2048], 'original_shape_hw': [3000, 4096],
        'bbox_xyxy': [0, 1050, 4096, 3000], 'scale': 0.5,
        'seed_length_original_px': 120, 'seed_sigma_original_px': 4.0,
        'preprocessing': 'exact batch_ab_preview: cvtColor BGR2GRAY/255 float32; INTER_AREA x0.5; Gaussian distance seed; tensor_sample no augmentation',
        'response_semantics': 'sigmoid center response intensity, not calibrated probability or rope body mask',
        'response_coordinates': 'model crop coordinates; upsample bilinear align_corners=False to bbox; no response exists outside bbox',
        'response_files': 'responses/<image_id>/entry_00.npz, key=response, float32 (975,2048); entries never merged',
        'input_cache_files': 'input_cache/<image_id>.npz, keys=image and seeds; same spatial shape',
        'no_training': True, 'target_labels_read': False, 'S1_labels_read': False,
        'reserved_data_decoded': False, 'devices_connected': False, 'independent_test_claim': False,
        'model_training_overlap_note': 'the model uses historical open development groups; these292 are not an independent test set',
        'frozen_sources': [{'path': str(p), 'sha256': sha(p)} for p in frozen_paths],
        'inference_helper_sha256': sha(__file__), 'jobs': jobs,
    }
    write_new_json(dest, protocol)
    snapshot = HERE / 'inference_source_snapshot'
    snapshot.mkdir(exist_ok=False)
    for p in [Path(__file__)] + [CODE / n for n in ('batch_ab_preview.py', 'batch_preview.py', 'model.py', 'train.py', 'prepare.py')]:
        shutil.copyfile(p, snapshot / p.name)
    return protocol


def subset_jobs(protocol, subset):
    return [j for j in protocol['jobs'] if subset == 'all' or ((j['id'] in PILOT_IDS) == (subset == 'pilot'))]


def seed_response(cv2, seed_xy, shape):
    # Exact copy of prepare.response_from_lines's numerical operations; the
    # implementation is local to avoid importing annotation-preparation code.
    mask = np.zeros(shape, np.uint8)
    points = (np.asarray(seed_xy, float) - np.asarray([0, 1050])) * 0.5
    cv2.polylines(mask, [np.rint(points).astype(np.int32)], False, 1, 1, cv2.LINE_8)
    assert mask.any()
    distance = cv2.distanceTransform(1-mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.exp(-0.5*(distance/(4.0*0.5))**2).astype(np.float32)


def prepare(protocol, jobs, subset):
    import cv2
    cv2.setNumThreads(2)
    cache_dir = HERE / 'input_cache'
    cache_dir.mkdir(exist_ok=True)
    output = []
    for i, job in enumerate(jobs, 1):
        assert sha(job['image_path']) == job['image_sha256']
        verify_hashes(job['seed_source_hashes'])
        dest = cache_dir / (job['id'] + '.npz')
        if dest.exists():
            with np.load(dest, allow_pickle=False) as z:
                assert z['image'].shape == (975, 2048) and z['seeds'].shape == (len(job['seeds']), 975, 2048)
            reused = True
        else:
            bgr = cv2.imdecode(np.fromfile(job['image_path'], np.uint8), cv2.IMREAD_COLOR)
            assert bgr.shape == (3000, 4096, 3)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            image = cv2.resize(gray[1050:], (2048, 975), interpolation=cv2.INTER_AREA)
            seeds = np.stack([seed_response(cv2, s['xy'], image.shape) for s in job['seeds']])
            np.savez_compressed(dest, image=image, seeds=seeds)
            reused = False
        output.append({'id': job['id'], 'cache_path': str(dest), 'cache_sha256': sha(dest),
                       'entry_count': len(job['seeds']), 'reused': reused})
        if i == 1 or i % 30 == 0:
            print(json.dumps({'prepared': i, 'total': len(jobs), 'subset': subset}), flush=True)
    return {'opencv': cv2.__version__, 'items': output, 'count': len(output)}


def run(protocol, jobs, subset):
    import torch
    sys.path.insert(0, str(CODE))
    from train import load_model, tensor_sample
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    cp = protocol['model_B']
    assert sha(cp['path']) == cp['sha256']
    checkpoint = torch.load(cp['path'], map_location='cpu', weights_only=False)
    assert checkpoint['step'] == 750
    cfg = checkpoint['config']
    net = load_model(cp['path'])
    output = []
    dest_root = HERE / 'responses'
    dest_root.mkdir(exist_ok=True)
    prep_receipt = read_json(HERE / ('prepare_' + subset + '_receipt.json'))
    assert prep_receipt['exit_code'] == 0
    caches = {r['id']: r for r in prep_receipt['result']['items']}
    with torch.inference_mode():
        for i, job in enumerate(jobs, 1):
            assert sha(job['image_path']) == job['image_sha256']
            cache = caches[job['id']]
            assert sha(cache['cache_path']) == cache['cache_sha256']
            with np.load(cache['cache_path'], allow_pickle=False) as z:
                image, seeds = z['image'], z['seeds']
            assert image.shape == (975, 2048)
            dest = dest_root / job['id']
            dest.mkdir(exist_ok=False)
            entries = []
            for k, seed in enumerate(seeds):
                zeros = np.zeros_like(image)
                arrays = {'image': image, 'seed': seed, 'target': zeros,
                          'valid': zeros.astype(bool), 'stage': zeros.astype(np.uint8),
                          'roi': np.ones_like(image, bool)}
                x, _, _, _, normalization = tensor_sample(arrays, cfg, augment=False)
                response = net(x)[0, 0, :975, :2048].numpy().copy()
                assert response.dtype == np.float32 and response.shape == (975, 2048)
                assert np.isfinite(response).all() and 0 <= response.min() <= response.max() <= 1
                path = dest / f'entry_{k:02d}.npz'
                np.savez_compressed(path, response=response)
                entries.append({'entry_index': k, 'seed': job['seeds'][k], 'response_path': str(path),
                                'response_sha256': sha(path), 'range': [float(response.min()), float(response.max())],
                                'normalization': normalization})
            record = {'id': job['id'], 'image_sha256': job['image_sha256'], 'bbox': job['bbox'],
                      'shape_hw': [975, 2048], 'model_sha256': cp['sha256'], 'entry_conditions': entries,
                      'input_cache_sha256': cache['cache_sha256'], 'source_sha256': sha(__file__),
                      'protocol_sha256': sha(HERE/'protocol_inputs.json'), 'entry_conditions_merged': False,
                      'target_labels_read': False, 'original_response_not_minmax_normalized': True}
            write_new_json(dest / 'inference.json', record)
            output.append({'id': job['id'], 'entries': len(entries), 'inference_path': str(dest/'inference.json')})
            if i == 1 or i % 20 == 0:
                print(json.dumps({'inferred': i, 'total': len(jobs), 'subset': subset}), flush=True)
    return {'torch': torch.__version__, 'device': 'cpu', 'threads': 8,
            'count': len(output), 'network_forward_calls': sum(r['entries'] for r in output), 'items': output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--subset', choices=['pilot', 'remaining', 'all'], required=True)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    receipt = HERE / f'{args.mode}_{args.subset}_receipt.json'
    assert not receipt.exists(), 'refuse to overwrite prior execution receipt'
    tick = time.perf_counter()
    result = None
    failure = None
    try:
        protocol = initialize_protocol()
        jobs = subset_jobs(protocol, args.subset)
        result = (prepare if args.mode == 'prepare' else run)(protocol, jobs, args.subset)
        code = 0
    except Exception:
        failure = traceback.format_exc()
        print(failure, file=sys.stderr)
        code = 1
    write_new_json(receipt, {'version': VERSION, 'mode': args.mode, 'subset': args.subset,
                            'exit_code': code, 'failure': failure, 'result': result,
                            'seconds': time.perf_counter()-tick, 'command': sys.argv,
                            'python': sys.version, 'executable': sys.executable,
                            'platform': platform.platform(), 'numpy': np.__version__,
                            'source_sha256': sha(__file__), 'created_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'mode': args.mode, 'subset': args.subset, 'exit_code': code,
                      'seconds': round(time.perf_counter()-tick, 2)}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
