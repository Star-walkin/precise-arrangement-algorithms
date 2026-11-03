"""Fixed-pilot free-stage probe; auxiliary output only, no source/parameter edits."""
from collections import Counter
from pathlib import Path
import json
import time

import cv2
import numpy as np

from whole_io import HERE, inputs, selection, dump, sha, write_image
from free_rope import refine_and_extend, dependencies


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if not (isinstance(v, np.ndarray) and v.ndim >= 2)}
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    return value


def paint(image, mask, color):
    result = image.copy(); use = mask != 0
    result[use] = np.clip(.6 * result[use] + .4 * np.asarray(color), 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 1)
    return result


def frame(shape, center, size):
    h, w = shape[:2]; cw, ch = size
    x = int(np.clip(round(center[0] - cw / 2), 0, max(0, w - cw)))
    y = int(np.clip(round(center[1] - ch / 2), 0, max(0, h - ch)))
    return [x, y, min(cw, w), min(ch, h)]


def panels(images, labels, rectangle, output):
    x, y, w, h = rectangle; pieces = []
    for image, label in zip(images, labels):
        head = np.full((42, w, 3), 18, np.uint8)
        cv2.putText(head, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, .42, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(head, f'x={x},y={y}; original pixels', (6, 34), cv2.FONT_HERSHEY_SIMPLEX, .32, (185, 185, 185), 1, cv2.LINE_AA)
        pieces.append(np.vstack((head, image[y:y+h, x:x+w])))
    write_image(output, np.hstack(pieces))


def main():
    out = HERE / 'free_probe_002'
    if out.exists(): raise ValueError('probe output already exists')
    rows = selection('pilot'); assert len(rows) == 16
    source_paths = [HERE / 'free_rope.py', HERE / 'whole_io.py', Path(__file__), *map(Path, dependencies())]
    source = {str(p): sha(p) for p in source_paths}
    dump(out / 'run_context.json', {'selection': [r['output_file'] for r in rows], 'source_sha256': source,
                                  'parameters_unchanged': True, 'scope': 'free-stage probe only'})
    records = []
    for number, row in enumerate(rows, 1):
        begin = time.perf_counter(); data = inputs(row); result = refine_and_extend(data)
        role = Path(row['output_file']).parent.name; stem = Path(row['output_file']).stem
        write_image(out / 'masks/free'/role/(stem+'.png'), result['free_mask'])
        write_image(out / 'masks/straight'/role/(stem+'.png'), result['straight_observed_mask'])
        raw = data['original']; free_image = paint(raw, result['free_mask'], (30, 255, 30))
        straight_image = paint(free_image, result['straight_observed_mask'], (255, 255, 0))
        straight_image = paint(straight_image, result['straight_inferred_mask'], (255, 0, 220))
        seed = data['a2_record'].get('free_segment', data['a1_record'].get('free_segment', {}))
        seed_ends = np.asarray(seed.get('observed_segment_xy') or [[raw.shape[1]/2,raw.shape[0]/2]]*2, float)
        center = seed_ends.mean(axis=0)
        surface_dir = seed.get('toward_surface_direction_xy')
        near = seed_ends[int(np.argmax(seed_ends @ np.asarray(surface_dir)))] if surface_dir is not None else seed_ends[-1]
        candidates = result['straight_candidates']; points = list(seed_ends)
        for candidate in candidates:
            end = np.asarray(candidate['visible_end_xy']); points.append(end)
            cv2.drawMarker(straight_image, tuple(np.rint(end).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        if len(candidates) == 1: near = np.asarray(candidates[0]['visible_end_xy'])
        rects = {'free_detail': frame(raw.shape, center, (600, 340)), 'stop_detail': frame(raw.shape, near, (600, 440))}
        preview_paths = {}
        for kind, rectangle in rects.items():
            path = out/'previews'/role/(stem+'_'+kind+'.jpg')
            panels([raw, free_image, straight_image], ['Original RGB', 'Revalidated free (green)', 'Straight support (cyan); stop red'], rectangle, path)
            preview_paths[kind] = str(path)
        combined = np.hstack([cv2.resize(raw, (819, 600)), cv2.resize(straight_image, (819, 600))])
        overview = out/'previews'/role/(stem+'_overview.jpg');write_image(overview, combined)
        record = {'output_file': row['output_file'], 'status': result['status'], 'reason': result['diagnostics'].get('reason'),
                  'outside': result['diagnostics'].get('outside_remeasurement'), 'free_pixels': int(np.count_nonzero(result['free_mask'])),
                  'straight_pixels': int(np.count_nonzero(result['straight_observed_mask'])),
                  'inferred_pixels': int(np.count_nonzero(result['straight_inferred_mask'])),
                  'wide': result['diagnostics'].get('wide_remeasurement'),
                  'outward_end': result['outward_extension']['visible_end_xy'] if result['outward_extension'] else None,
                  'width_px': result['free_segment'].get('width_px'), 'seed_width_px': seed.get('width_px'),
                  'candidate_count': len(candidates), 'stops': [{'reason': c['stop_reason'], 'visible_end_xy': c['visible_end_xy'],
                   'observed_profiles': c['observed_profile_count'], 'overlap_observed_profiles': sum(s['state']=='observed' and s['in_surface'] for s in c['support_samples'])} for c in candidates],
                  'previews': preview_paths, 'overview': str(overview), 'rectangles': rects,
                  'input_dependency_sha256': data['input_dependency_sha256'], 'elapsed_s': time.perf_counter() - begin}
        dump(out/'records'/role/(stem+'.json'), {'summary': record, 'free_result': clean(result)})
        records.append(record)
        print(json.dumps({k:record[k] for k in ['output_file','status','reason','free_pixels','straight_pixels','stops']}, ensure_ascii=False), flush=True)
    assert source == {str(p): sha(p) for p in source_paths}
    dump(out/'index.json', {'cases': records, 'source_unchanged': True, 'counts': dict(Counter(r['status'] for r in records)),
                            'accuracy': None, 'equipment_access': False, 'reserved_data_access': False})


if __name__ == '__main__': main()
