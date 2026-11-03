"""Prepare a small, group-separated HUMAN review pilot from existing videos.

No camera/control imports. Clean frames only; automatic geometry is used to
suggest a viewing ROI, never to generate human truth. Re-running cannot erase
annotations. The historical V4 protocol/reference is intentionally immutable.
"""
from pathlib import Path
import argparse
import hashlib
import json

import cv2
import numpy as np

from .geometry import analyze_frame, read_image, write_image, dump_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'vision_geometry_lab/outputs/centerline_v5/pilot'
VIDEO_ROOT = ROOT / '现场视频/训练采集/双相机同步'
CASES = (
    ('dual_20260924_111500', 'left', 'train', (24.0, 24.6, 27.3)),
    ('dual_20260924_111143', 'right', 'train', (11.4, 13.0, 19.0)),
    ('dual_20260923_113745', 'left', 'holdout', (45.0, 55.0, 58.0)),
    ('dual_20260923_113305', 'right', 'holdout', (25.0, 50.0, 55.0)),
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def suggested_roi(shape, geometry):
    h, w = shape[:2]
    points = []
    for item in (geometry.get('arc', {}).get('points'),
                 geometry.get('rope', {}).get('observed_segment')):
        if item is not None:
            p = np.asarray(item, float)
            if p.ndim == 2 and p.shape[1] == 2 and np.isfinite(p).all():
                points.extend(p.tolist())
    if len(points) < 4:
        return [0, 0, w, h]
    p = np.asarray(points)
    lower = np.maximum([0, 0], np.floor(p.min(axis=0) - [220, 180])).astype(int)
    upper = np.minimum([w, h], np.ceil(p.max(axis=0) + [220, 300])).astype(int)
    return [int(lower[0]), int(lower[1]), int(upper[0]), int(upper[1])]


def check_groups(samples):
    groups = {}
    ids = set()
    for sample in samples:
        if sample['sample_id'] in ids:
            raise ValueError('Duplicate sample_id')
        ids.add(sample['sample_id'])
        group, split = sample['group_id'], sample['split']
        if split not in ('train', 'holdout'):
            raise ValueError('Unknown split')
        if group in groups and groups[group] != split:
            raise ValueError('A recording group cannot cross splits')
        groups[group] = split
    return groups


def validate_prepared(manifest):
    check_groups(manifest['samples'])
    for sample in manifest['samples']:
        if file_hash(sample['image_path']) != sample['image_sha256']:
            raise ValueError('Prepared clean image changed: ' + sample['sample_id'])
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        existing = validate_prepared(json.loads(manifest_path.read_text('utf-8')))
        print(f"Reused {len(existing['samples'])} verified clean frames; labels unchanged.")
        return 0
    cv2.setNumThreads(2)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(schema_version=1, cases=[dict(group_id=g, camera=c, split=s, times=list(ts))
                    for g, c, s, ts in CASES],
                    selection='fixed small pilot from existing known sessions; not a random population sample',
                    split_basis='entire recording session, both cameras kept in same group',
                    labels_generated=False, hardware_access=False)
    protocol_path = output / 'preparation_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text('utf-8')) != protocol:
        raise ValueError('Preparation protocol changed; choose a new output directory')
    dump_json(protocol_path, protocol)
    samples = []
    thumbnails = []
    for group, camera, split, times in CASES:
        matches = list((VIDEO_ROOT / group).glob(camera + '_*.avi'))
        if len(matches) != 1:
            raise ValueError(f'Expected one {camera} AVI in {group}')
        source = matches[0]
        cap = cv2.VideoCapture(str(source))
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps > 0:
            raise ValueError('Video has no nominal frame rate: ' + str(source))
        try:
            for seconds in times:
                index = int(round(seconds * fps))
                if not 0 <= index < count:
                    raise ValueError(f'Frame outside source: {source} {index}')
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Cannot read {source} frame {index}')
                name = f'{group}__{camera}__f{index:05d}'
                path = output / 'images' / (name + '.png')
                digest = hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
                if path.exists():
                    cached = read_image(path)
                    if hashlib.sha256(np.ascontiguousarray(cached).tobytes()).hexdigest() != digest:
                        raise ValueError('Existing image differs; will not overwrite: ' + str(path))
                else:
                    write_image(path, frame)
                result = analyze_frame(frame)
                roi = suggested_roi(frame.shape, result)
                sample = dict(sample_id=name, group_id=group, split=split, image_path=str(path),
                              image_sha256=file_hash(path), image_shape_hw=list(frame.shape[:2]),
                              roi_xyxy=roi, roi_source='automatic viewing proposal; not ground truth',
                              annotation_path=str(output / 'labels' / (name + '.json')),
                              frame_bgr_sha256=digest,
                              source=dict(video_path=str(source), frame_index=index, ui_frame=index+1,
                                          time_s=index/fps, nominal_fps=fps, camera=camera,
                                          time_basis='AVI frame index / nominal fps, not exposure-calibrated'))
                samples.append(sample)
                dump_json(output / 'diagnostics_not_labels' / (name + '.json'), result)
                x0,y0,x1,y1 = roi
                thumb = cv2.resize(frame[y0:y1,x0:x1], (560,300))
                cv2.rectangle(thumb, (0,0),(560,38),(20,20,20),-1)
                cv2.putText(thumb, f'{len(samples):02d} {split} {camera} {index/fps:.1f}s',
                            (8,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
                thumbnails.append(thumb)
                print(name, 'prepared', flush=True)
        finally:
            cap.release()
    check_groups(samples)
    manifest = dict(schema_version=1, task='seed_conditioned_new_winding_centerline',
                    samples=samples, physical_control=False,
                    label_policy='manual only; legacy SAM masks and rough reference dots are not truth',
                    initial_human_labels=0,
                    user_clarification=dict(ui_frame=247, time_s=24.6,
                        intent='six existing dots were placed as close as possible to rope-body center',
                        precision='approximate manual center references, not pixel-exact truth',
                        source='user clarification in this conversation; historical V4 artifacts unchanged'))
    dump_json(manifest_path, manifest)
    contact = np.vstack([np.hstack(thumbnails[i:i+3]) for i in range(0,len(thumbnails),3)])
    write_image(output / 'contact_sheet.jpg', contact)
    print(f'Created {len(samples)} clean samples, zero new labels: {manifest_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
