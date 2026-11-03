"""Extract short 5 fps windows around the six verified SAM 2 anchors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def source_frame_from_name(name: str) -> int:
    match = re.search(r'__f(\d+)(?:__[^.]*)?\.jpg$', name)
    if not match:
        raise ValueError(f'Cannot parse source frame from {name}')
    return int(match.group(1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', type=Path, default=ROOT / 'task_v1')
    parser.add_argument('--video-root', type=Path, default=None)
    parser.add_argument('--sample-fps', type=float, default=5.0)
    parser.add_argument('--radius-seconds', type=float, default=3.0)
    parser.add_argument('--mirror-horizontal', action='store_true')
    return parser.parse_args()


def find_video(name: str, video_root: Path | None) -> Path:
    if video_root is not None:
        matches = list(video_root.rglob(name))
    else:
        matches = list((PROJECT / '现场视频').rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one source video named {name}, got {matches}')
    return matches[0]


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {'file': str(path.resolve()), 'fps': fps, 'frame_count': frame_count,
            'duration_s': frame_count / fps}


def main():
    args = parse_args()
    task_dir = args.task_dir.resolve()
    manifest = json.loads((task_dir / 'manifest.json').read_text(encoding='utf-8'))
    clips = {name: probe_video(find_video(name, args.video_root))
             for name in {row['clip'] for row in manifest['tasks']}}
    output_root = task_dir / 'propagation_windows'
    windows = []
    for task in manifest['tasks']:
        clip = clips[task['clip']]
        fps = float(clip['fps'])
        start_time = max(0.0, task['time_s'] - args.radius_seconds)
        end_time = min(float(clip['duration_s']), task['time_s'] + args.radius_seconds)
        start_frame = max(0, round(start_time * fps))
        end_frame = min(int(clip['frame_count']) - 1, round(end_time * fps))
        step = max(1, round(fps / args.sample_fps))
        source_frames = set(range(start_frame, end_frame + 1, step))
        anchor_source_frame = source_frame_from_name(task['image_name'])
        source_frames.add(anchor_source_frame)
        source_frames = sorted(source_frames)

        window_id = f"anchor_{task['order']:02d}_{Path(task['image_name']).stem}"
        frame_dir = output_root / window_id / 'frames'
        frame_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(clip['file'])
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {clip['file']}")
        records = []
        for output_index, source_frame in enumerate(source_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode {task['clip']} frame {source_frame}")
            if args.mirror_horizontal:
                frame = cv2.flip(frame, 1)
            output_path = frame_dir / f'{output_index:05d}.jpg'
            if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f'Cannot write {output_path}')
            records.append({
                'index': output_index,
                'source_frame': source_frame,
                'time_s': source_frame / fps,
                'is_anchor': source_frame == anchor_source_frame,
            })
        cap.release()
        anchor_indices = [row['index'] for row in records if row['is_anchor']]
        if len(anchor_indices) != 1:
            raise RuntimeError(f'Expected one anchor in {window_id}, got {anchor_indices}')
        window = {
            'window_id': window_id,
            'clip': task['clip'],
            'source_video': clip['file'],
            'source_fps': fps,
            'sample_fps': args.sample_fps,
            'mirror_horizontal': args.mirror_horizontal,
            'frame_dir': str(frame_dir.resolve()),
            'frame_count': len(records),
            'anchor_index': anchor_indices[0],
            'anchor_image_name': task['image_name'],
            'anchor_label_dir': str((task_dir / 'labels' / Path(task['image_name']).stem).resolve()),
            'frames': records,
        }
        (output_root / window_id / 'window.json').write_text(
            json.dumps(window, ensure_ascii=False, indent=2), encoding='utf-8')
        windows.append(window)
        print(f"{window_id}: {len(records)} frames, anchor={anchor_indices[0]}")
    (output_root / 'manifest.json').write_text(
        json.dumps({'schema_version': 1, 'windows': windows}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(f'Prepared {len(windows)} windows and {sum(w["frame_count"] for w in windows)} frames.')


if __name__ == '__main__':
    main()
