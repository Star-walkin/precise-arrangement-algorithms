"""One allowed current frame: independent serial/worker F3, no model calls."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

import run_full_fixed_d_anchored as runner


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparable(value):
    # Timing is deliberately process-specific; geometry and decisions are not.
    if isinstance(value, dict):
        return {key: comparable(item) for key, item in value.items()
                if not key.endswith('_seconds') and not key.endswith('_ms')}
    if isinstance(value, list):
        return [comparable(item) for item in value]
    return value


def main():
    output = ROOT/'audit_F3_parallel_equivalence.json'
    if output.exists():
        raise FileExistsError('Preserve completed audit')
    frame_id = 0
    record_path = ROOT/'analysis/frames/frame_000000.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    source = runner.SOURCE
    capture = cv2.VideoCapture(str(source))
    ok, image = capture.read()
    position = capture.get(cv2.CAP_PROP_POS_FRAMES)
    capture.release()
    if not ok or abs(position-1) > .1:
        raise RuntimeError('Cannot decode permitted source f0')
    pixel = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    if pixel != record['source_pixel_sha256']:
        raise RuntimeError('Source pixels do not match current-frame record')
    mask, pitch, scene = runner.calibration(source.parent.name, 'left')
    y = record['intersection_y_px']
    if y is None:
        raise RuntimeError('Audit needs the available current S1 intersection')
    started = time.perf_counter()
    cv2.setNumThreads(1)
    serial = runner.SurfaceObserver(mask, pitch).step(image, d=2238.7, active_y=y)
    with ProcessPoolExecutor(max_workers=1, initializer=runner.init_surface_worker,
                             initargs=(mask, pitch)) as executor:
        work = executor.submit(runner.current_surface, image, y, frame_id).result()
    if (work['frame_index'] != frame_id or work['source_pixel_sha256'] != pixel
            or work['intersection_y_px'] != y):
        raise RuntimeError('Worker current-frame binding failed')
    parallel = work['surface']
    serial_comparable = comparable(serial)
    parallel_comparable = comparable(parallel)
    differing_keys = [key for key in serial_comparable
                      if serial_comparable[key] != parallel_comparable.get(key)]
    equal = serial_comparable == parallel_comparable
    result = dict(status='passed' if equal else 'failed',
        source_video=str(source), source_frame=frame_id,
        source_original_pixel_sha256=pixel,
        current_record_path=str(record_path), current_record_sha256=digest(record_path),
        fixed_d_original_px=2238.7, current_intersection_y_px=y,
        mask_sha256=scene['mask_sha256'], pitch_px=pitch,
        actual_F3_calls=2, actual_model_calls=0, actual_S1_calls=0,
        worker_source_pixel_y_and_frame_binding_passed=True,
        all_non_timing_surface_fields_exactly_equal=equal,
        differing_top_level_non_timing_fields=differing_keys,
        serial_summary={key:serial.get(key) for key in (
            'count_valid','continuous_pitch_distance','edge_y','local_pitch_px',
            'measurement_x','count_observed_grooves','reasons','crossings_y')},
        worker_summary={key:parallel.get(key) for key in (
            'count_valid','continuous_pitch_distance','edge_y','local_pitch_px',
            'measurement_x','count_observed_grooves','reasons','crossings_y')},
        timing_fields_excluded_from_exact_comparison=True,
        parallel_processes_used_for_this_equivalence_test=1,
        scope='One-frame serial/worker transport and geometry equivalence, not all-video accuracy validation.',
        source_hashes=dict(audit_script=digest(Path(__file__)),
            worker_runner=digest(Path(runner.__file__)),
            original_F3=digest(runner.BASE/'lc1_surface.py')),
        wall_seconds=time.perf_counter()-started)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key:result[key] for key in (
        'status','all_non_timing_surface_fields_exactly_equal',
        'differing_top_level_non_timing_fields','wall_seconds')}, ensure_ascii=False))
    if not equal:
        raise RuntimeError('F3 equivalence audit did not pass')


if __name__ == '__main__':
    main()
