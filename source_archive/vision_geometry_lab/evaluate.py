"""Reproducible raw-frame test, no labels fed to automatic detectors.

Visibility labels are used ONLY for reporting. SAM masks are rough references,
not subpixel truth. This script reports availability, not an accuracy pass.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import time

import cv2
import json
import numpy as np
from PIL import Image, ImageDraw

from .geometry import analyze_frame, draw_overlay, read_image, write_image, dump_json

ROOT = Path(__file__).resolve().parents[1]


def image_cases(include_train=False):
    for task_name in ['task_dual_midlayer_v1', 'task_dual_thicklayer_v1']:
        for split in (['train','holdout'] if include_train else ['holdout']):
            manifest = ROOT/'vision_training'/task_name/split/'manifest.json'
            if not manifest.is_file():
                continue
            data = json.loads(manifest.read_text(encoding='utf-8'))
            for row in data['tasks']:
                path = Path(row['image'])
                label_path = manifest.parent/'labels'/path.stem/'annotation.json'
                label = json.loads(label_path.read_text(encoding='utf-8')) if label_path.is_file() else {}
                yield dict(name=path.stem, image=read_image(path), source=str(path),
                           reference_visibility=label.get('incoming_visibility','unknown'),
                           provenance=task_name+'/'+split+'_inspected_during_development')
    folder = ROOT/'现场图片/0924测试'
    for path in sorted(folder.glob('*.bmp')):
        yield dict(name=path.stem, image=read_image(path), source=str(path),
                   reference_visibility='unknown', provenance='field_still')


def video_cases():
    base = ROOT/'现场视频/训练采集/双相机同步'
    cases = [('dual_20260924_111500', 'left_DA9627065.avi',
              [22,24,24.3,25,26,26.5,27,27.3,28,29,30,31,32]),
             ('dual_20260924_111143', 'right_DA9626975.avi',
              [10,11.4,12,13,13.8,14.4,15,16,16.8,17.1,18.6,20])]
    for session, filename, timestamps in cases:
        path = base/session/filename
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError('Cannot open video '+str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        for seconds in timestamps:
            index = round(seconds*fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode {path} frame {index}')
            yield dict(name=f'{session}_{Path(filename).stem}_t{index/fps:.3f}',
                       image=frame, source=str(path), time_s=index/fps,
                       reference_visibility='unknown', provenance='reversal_raw_video')
        cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(Path(__file__).parent/'outputs/regression_v3'))
    parser.add_argument('--videos', action='store_true')
    parser.add_argument('--include-train', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    measurements, thumbnails = [], []
    def all_cases():
        yield from image_cases(args.include_train)
        if args.videos:
            yield from video_cases()
    for i, case in enumerate(all_cases()):
        if args.limit and i >= args.limit:
            break
        frame = case.pop('image')
        result = analyze_frame(frame)
        result.update(case)
        dump_json(folder/(case['name']+'.json'), result)
        overlay = draw_overlay(frame, result)
        # Keep original-resolution diagnostic image for checking actual pixels.
        write_image(folder/(case['name']+'.jpg'), overlay)
        thumb = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        thumb.thumbnail((560,400))
        tile = Image.new('RGB',(580,460),'#111827')
        tile.paste(thumb,((580-thumb.width)//2,45))
        draw = ImageDraw.Draw(tile)
        draw.text((8,5),case['name'],fill='white')
        draw.text((8,22),f"arc={result['arc']['valid']} ellipse={result.get('conic',{}).get('valid')} rope={result['rope']['valid']} {result['timing_ms']['total']:.0f}ms",fill='white')
        thumbnails.append(tile)
        measurements.append(result)
        print(i+1, case['name'], result['arc']['valid'], result['rope']['valid'],
              result['rope'].get('reason'), f"{result['timing_ms']['total']:.0f}ms", flush=True)
    summary = dict(cases=len(measurements),
        arc_available=sum(r['arc']['valid'] for r in measurements),
        local_boundary_available=sum(r['arc'].get('local_boundary_valid',False) for r in measurements),
        wound_curve_statuses=dict(Counter(r.get('wound_curve',{}).get('status','not_run') for r in measurements)),
        wound_curve_identity_confirmed=sum(r.get('wound_curve',{}).get('identity_confirmed',False) for r in measurements),
        conic_available=sum(r.get('conic',{}).get('valid',False) for r in measurements),
        conic_reasons=dict(Counter(r.get('conic',{}).get('reason','') for r in measurements)),
        relation_statuses=dict(Counter(r.get('relation',{}).get('status','') for r in measurements)),
        rope_available=sum(r['rope']['valid'] for r in measurements),
        visibility_breakdown={label:dict(n=sum(r['reference_visibility']==label for r in measurements),
            rope_available=sum(r['reference_visibility']==label and r['rope']['valid'] for r in measurements))
            for label in sorted({r['reference_visibility'] for r in measurements})},
        rope_reasons=dict(Counter(r['rope'].get('reason','') for r in measurements)),
        median_ms=float(np.median([r['timing_ms']['total'] for r in measurements])) if measurements else None,
        p95_ms=float(np.percentile([r['timing_ms']['total'] for r in measurements],95)) if measurements else None,
        accuracy_status='NOT_VALIDATED; availability is not accuracy; no precision arc/line ground truth',
        labels_fed_to_detector=False, cases_manifest=[{k:r.get(k) for k in ('name','source','time_s','provenance')} for r in measurements])
    dump_json(folder/'summary.json', summary)
    for start in range(0,len(thumbnails),12):
        batch = thumbnails[start:start+12]
        sheet = Image.new('RGB',(580*3,460*((len(batch)+2)//3)),'#111827')
        for j,tile in enumerate(batch):
            sheet.paste(tile,((j%3)*580,(j//3)*460))
        sheet.save(folder/f'contact_sheet_{start//12+1}.jpg')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
