"""Read-only V16 source/seed/path visual audit. Does not change measurements.

Fixed chronological sample times are chosen independently of event annotations.
Images are diagnostics, not new labels or calibrated geometric ground truth.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .v16_video_measurement import BASE, ARMS, read, sha, write
from .v11_path_contract import array_sha256

DEFAULT_TIMES = (0., 10., 20., 30., 40., 50., 60.)
COLORS = dict(seed='#ff9900', arc='#50e87b', path='#ff59e8')


def _font(size=17):
    for name in ('C:/Windows/Fonts/consola.ttf', 'C:/Windows/Fonts/arial.ttf'):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _points(segments):
    arrays = [np.asarray(p, float) for p in segments if len(p)]
    return np.concatenate(arrays) if arrays else np.empty((0, 2))


def focus_box(meta, extra_segments=()):
    """Context crop from image-domain observations only, never event truth."""
    h, w = meta['frame_shape'][:2]
    points = _points([meta.get('free_seed') or []] + meta['arc'].get('local_boundary_segments', []) + list(extra_segments))
    if not len(points):
        return (0, 0, w, h)
    x0, y0 = np.maximum(0, points.min(axis=0)-100)
    x1, y1 = np.minimum([w, h], points.max(axis=0)+[100, 500])
    return tuple(map(int, (x0, y0, x1, y1)))


def panel(frame_rgb, meta, decoded=None, features=None, title='raw', show_overlay=True, size=(850, 500), crop=None):
    crop = focus_box(meta) if crop is None else crop
    source = Image.fromarray(frame_rgb).crop(crop)
    available_h = size[1]-68
    scale = min(size[0]/source.width, available_h/source.height)
    width, height = round(source.width*scale), round(source.height*scale)
    source = source.resize((width, height), Image.Resampling.LANCZOS)
    tile = Image.new('RGB', size, '#17202a'); ox = (size[0]-width)//2; oy = 62+(available_h-height)//2
    tile.paste(source, (ox, oy)); draw = ImageDraw.Draw(tile)
    def mapped(points):
        return [(ox+(x-crop[0])*scale, oy+(y-crop[1])*scale) for x,y in points]
    if show_overlay:
        for segment in meta['arc'].get('local_boundary_segments', []):
            if len(segment)>1:
                draw.line(mapped(segment), fill=COLORS['arc'], width=2)
        seed = meta.get('free_seed')
        if seed:
            pp = mapped(seed); draw.line(pp, fill=COLORS['seed'], width=3)
            x,y = pp[-1]; draw.ellipse((x-5,y-5,x+5,y+5), outline=COLORS['seed'], width=3)
            draw.text((x+8,y-18),'auto seed END',font=_font(14), fill=COLORS['seed'])
        if decoded:
            for segment in decoded.get('observed_segments', []):
                if len(segment)>1:
                    draw.line(mapped(segment), fill=COLORS['path'], width=3)
    draw.text((10,5), title, font=_font(19), fill='white')
    draw.text((10,30), 'orange: observed free seed | green: direct boundary | magenta: unconfirmed path', font=_font(14), fill='#c8d2dc')
    if features is not None:
        value=features.get('image_gap_proxy'); text='proxy='+('unavailable' if value is None else f'{value:.3f}')
        text+=f" | direct points={features.get('supported_samples',0)} | identity NOT confirmed"
        draw.rectangle((0,size[1]-25,size[0],size[1]),fill='#17202a')
        draw.text((10,size[1]-23),text,font=_font(14),fill='#ffd084')
    return tile


def audit_seed(meta):
    seed=meta.get('free_seed'); width=meta['rope'].get('width_px')
    result=dict(frame_index=meta['frame_index'],time_s=meta['time_s'],available=meta['available'],
        optical_width_px=width,angle_deg=meta['rope'].get('angle_deg'),free_seed=seed,
        seed_endpoint_is_measured_bend=False,manual_annotations_read=False,
        direct_arc_segments=len(meta['arc'].get('local_boundary_segments',[])),
        arc_observed_points=len(meta['arc'].get('points',[])),seed_reason=meta['seed_reason'])
    arc=_points(meta['arc'].get('local_boundary_segments',[]))
    if seed:
        segment=np.asarray(seed); result['observed_seed_length_px']=float(np.linalg.norm(segment[-1]-segment[0]))
        result['entry_endpoint_to_nearest_observed_arc_vertex_px']=float(np.linalg.norm(arc-segment[-1],axis=1).min()) if len(arc) else None
    return result


def render(output, sessions=('left','right'), times=DEFAULT_TIMES, destination=None, caller_specified_times=False):
    output=Path(output); protocol=read(output/'protocol.json')
    if destination is None:
        destination=output/'diagnostics'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Use a new diagnostic directory')
    destination.mkdir(parents=True)
    records=[]; used_sources={str(output/'protocol.json'):sha(output/'protocol.json')}
    for session in sessions:
        video=protocol['videos'][session]; cap=cv2.VideoCapture(video['path'])
        if not cap.isOpened():raise ValueError('cannot open original video')
        measurements={}
        for arm in ARMS:
            path=output/'measure'/session/(arm+'.json')
            if path.is_file():
                value=read(path)
                if value['protocol_sha256']!=sha(output/'protocol.json'):raise ValueError('foreign measurements')
                measurements[arm]={r['frame_index']:r for r in value['records']}
                used_sources[str(path)]=sha(path)
        try:
            for seconds in times:
                index=round(seconds*video['fps']/2)*2
                meta_path=output/'extract'/session/'frames'/f'f{index:06d}.json'
                if not meta_path.is_file():continue
                meta=read(meta_path);used_sources[str(meta_path)]=sha(meta_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
                if not ok:raise ValueError('cannot decode original frame')
                actual_index=int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1
                actual_frame_hash=array_sha256(frame)
                if actual_index!=index or actual_frame_hash!=meta['frame_sha256']:
                    raise ValueError(f'Original-frame seek/hash mismatch: requested={index}, decoded={actual_index}')
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                all_paths=[]
                for values in measurements.values():
                    if index in values:all_paths.extend(values[index]['decoded'].get('observed_segments',[]))
                crop=focus_box(meta,all_paths)
                panels=[panel(rgb,meta,title=f'{session} {index/video["fps"]:.1f}s RAW',show_overlay=False,crop=crop)]
                if measurements:
                    for arm, values in measurements.items():
                        value=values.get(index)
                        if value is None:continue
                        panels.append(panel(rgb,meta,value['decoded'],value['features'],title=f'{arm}: current-frame automatic seed',crop=crop))
                else:
                    panels.append(panel(rgb,meta,title='automatic seed + direct boundary; model not yet measured',crop=crop))
                figure=Image.new('RGB',(panels[0].width*len(panels),panels[0].height),'black')
                for i,tile in enumerate(panels):figure.paste(tile,(i*tile.width,0))
                path=destination/f'{session}_f{index:06d}.jpg';figure.save(path,quality=94)
                records.append(dict(**audit_seed(meta),session=session,image=str(path),
                    arms_rendered=list(measurements),image_sha256=sha(path),
                    decoded_frame_index=actual_index,source_frame_array_sha256=actual_frame_hash,
                    source_frame_verified_against_extraction=True))
        finally:cap.release()
    write(destination/'index.json',dict(sample_times=list(times),records=records,
        source_snapshot=used_sources,diagnostic_script_sha256=sha(__file__),
        diagnostic_not_event_validation=True,labels_loaded=False,
        time_selection=('caller-specified post-hoc inspection; may be chosen using reference event times' if caller_specified_times
                        else 'preselected fixed 0/10/20/30/40/50/60 second diagnostic sampling'),
        reference_event_times_may_have_informed_selection=bool(caller_specified_times),
        sample_selection_used_for_inference=False,hardware_access=False))
    print(json.dumps(dict(directory=str(destination),images=len(records)),ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=BASE)
    p.add_argument('--session',choices=('both','left','right'),default='both')
    p.add_argument('--times',type=float,nargs='+',default=None)
    p.add_argument('--destination',type=Path)
    args=p.parse_args()
    render(args.output,('left','right') if args.session=='both' else (args.session,),
           DEFAULT_TIMES if args.times is None else args.times,args.destination,
           caller_specified_times=args.times is not None)


if __name__=='__main__':main()
