"""Frozen raw-frame temporal-boundary ablation. No training or device access.

Consumes independently measured current-frame V16 geometry and center-only paths.
Only frame-to-frame registration is newly decoded from raw video. The control is
the identical direct-boundary grid with all historical propagation disabled.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime
from pathlib import Path
import time
import cv2
import numpy as np

from .v16_video_measurement import BASE as PREVIOUS, read, write, sha, verify as verify_previous
from .v11_path_contract import array_sha256
from .v17_scene_guard import ScenePoseGuard, DEFAULT_PARAMETERS
from .v17_temporal_boundary import TemporalBoundary, PARAMETERS, boundary_path_features

LAB = Path(__file__).resolve().parent
BASE = LAB/'outputs/v17_temporal_boundary'
MODES = ('direct_grid','temporal_guarded')
SOURCES = (LAB/'v17_boundary_trial.py', LAB/'v17_temporal_boundary.py', LAB/'v17_scene_guard.py')


def validate_inputs(extraction, measurement, expected, fps, protocol_sha, extraction_sha):
    if measurement['protocol_sha256'] != protocol_sha or extraction['protocol_sha256'] != protocol_sha:
        raise ValueError('input protocol mismatch')
    if measurement['extraction_index_sha256'] != extraction_sha:
        raise ValueError('input extraction provenance mismatch')
    for payload in (extraction, measurement):
        records = payload['records']
        if [r['frame_index'] for r in records] != expected:
            raise ValueError('input sequence missing, duplicate, or reordered')
        for r in records:
            if not np.isfinite(r['time_s']) or abs(r['time_s']-r['frame_index']/fps) > 1e-8:
                raise ValueError('input timestamp mismatch')


def validate_metadata(meta, index, fps):
    if meta['frame_index'] != index or abs(meta['fps']-fps) > 1e-8:
        raise ValueError('metadata frame/fps mismatch')
    if not np.isfinite(meta['time_s']) or abs(meta['time_s']-index/fps) > 1e-8:
        raise ValueError('metadata timestamp mismatch')


def freeze(output=BASE):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Use a new experiment directory; never overwrite a frozen run')
    old=verify_previous(PREVIOUS)
    files={}
    for session in ('left','right'):
        for path in (PREVIOUS/'extract'/session/'index.json',PREVIOUS/'measure'/session/'center_only.json'):
            files[str(path)]=sha(path)
    protocol=dict(schema='v17_temporal_boundary_1',frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input_protocol=dict(path=str(PREVIOUS/'protocol.json'),sha256=sha(PREVIOUS/'protocol.json')),
        input_files=files,source_snapshot={str(f):sha(f) for f in SOURCES},
        boundary_parameters=asdict(PARAMETERS),scene_parameters=asdict(DEFAULT_PARAMETERS),
        modes=list(MODES),sessions=['left','right'],videos=old['videos'],model='frozen V14 center_only via V16',
        hardware_access=False,labels_used_for_measurement=False,deployment_allowed=False,
        raw_path_visibility_classification='not_implemented; path probability is not visibility evidence',
        uncertainty='uncalibrated heuristic uncertainty envelope; not guaranteed bounds or statistical confidence intervals',
        stop_rule='No timing deployment if first full-video candidates miss or timing is too late; no threshold tuning on these results')
    write(output/'protocol.json',protocol)
    print('Frozen V17 boundary protocol',sha(output/'protocol.json'),flush=True)


def verify(output=BASE,verify_videos=False):
    output=Path(output);p=read(output/'protocol.json')
    if p['schema']!='v17_temporal_boundary_1' or p['boundary_parameters']!=asdict(PARAMETERS) or p['scene_parameters']!=asdict(DEFAULT_PARAMETERS):
        raise ValueError('V17 protocol mismatch')
    if p['modes']!=list(MODES):raise ValueError('mode mismatch')
    for path,digest in {**p['source_snapshot'],**p['input_files']}.items():
        if sha(path)!=digest:raise ValueError('Frozen file changed '+path)
    if sha(p['input_protocol']['path'])!=p['input_protocol']['sha256']:raise ValueError('V16 protocol changed')
    verify_previous(PREVIOUS,verify_videos=verify_videos)
    return p


def run(output,session):
    output=Path(output);p=verify(output,True);target=output/'measure'/session
    if target.exists():raise FileExistsError('V17 run already exists')
    target.mkdir(parents=True)
    video=p['videos'][session]
    extraction_path=PREVIOUS/'extract'/session/'index.json'
    extraction=read(extraction_path)
    rows=read(PREVIOUS/'measure'/session/'center_only.json')
    expected=video['selected_indices']
    validate_inputs(extraction,rows,expected,video['fps'],p['input_protocol']['sha256'],sha(extraction_path))
    lookup={r['frame_index']:r for r in rows['records']}; ex={r['frame_index']:r for r in extraction['records']}
    scene_guard=ScenePoseGuard();boundaries={mode:TemporalBoundary() for mode in MODES}
    cv2.setNumThreads(2)
    cap=cv2.VideoCapture(video['path']);records={m:[] for m in MODES};started=time.perf_counter()
    if not cap.isOpened():raise ValueError('video unavailable')
    try:
        for index in range(max(expected)+1):
            ok,frame=cap.read()
            if not ok:raise ValueError('video ended before frame '+str(index))
            if index not in lookup:continue
            decode_index=int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1
            if decode_index!=index:raise ValueError('decoded frame index mismatch')
            info=ex[index]
            if sha(info['metadata_path'])!=info['metadata_sha256']:raise ValueError('metadata changed')
            meta=read(info['metadata_path']);r=lookup[index]
            validate_metadata(meta,index,video['fps'])
            if array_sha256(frame)!=meta['frame_sha256']:raise ValueError('original frame pixels mismatch')
            now=index/video['fps']
            scene=scene_guard.update(frame,now)
            geometry=meta.get('geometry');width=geometry['width_px'] if geometry else float('nan')
            for mode in MODES:
                before=time.perf_counter()
                boundary=boundaries[mode].update(now,frame.shape,meta['arc'].get('local_boundary_segments',[]),
                    width,scene,temporal=mode=='temporal_guarded')
                features=boundary_path_features(r['decoded'],boundary,geometry)
                elapsed=(time.perf_counter()-before)*1000
                records[mode].append(dict(frame_index=index,time_s=now,features=features,boundary=boundary,
                    scene=scene,decoded_path_ambiguous=r['decoded'].get('diagnostics',{}).get('ambiguous'),
                    path_identity_confirmed=False,path_visibility_assessed=False,
                    decoded_available=r['decoded'].get('valid',False),
                    additional_processing_ms=elapsed+scene['elapsed_ms'],
                    approximate_serial_processing_ms=r['approximate_serial_processing_ms']+elapsed+scene['elapsed_ms'],
                    combined_timing_is_live_measurement=False))
            if len(records[MODES[0]])%25==0:
                print({'session':session,'frames':len(records[MODES[0]]),'elapsed_s':round(time.perf_counter()-started,2)},flush=True)
    finally:cap.release()
    verify(output)
    for mode,items in records.items():
        if [r['frame_index'] for r in items]!=expected:raise ValueError('not complete')
        write(target/(mode+'.json'),dict(schema='v17_boundary_measurements_1',session=session,mode=mode,
            protocol_sha256=sha(output/'protocol.json'),records=items,video=video,hardware_access=False,
            ground_truth_used=False,deployment_allowed=False,model_identity_verified=False))
    print('Completed V17',session,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('freeze','verify','run'))
    p.add_argument('--output',type=Path,default=BASE)
    p.add_argument('--session',choices=('left','right'),default='left')
    args=p.parse_args()
    if args.stage=='freeze':freeze(args.output)
    elif args.stage=='verify':verify(args.output,True);print('V17 frozen sources and inputs verified')
    else:run(args.output,args.session)


if __name__=='__main__':main()
