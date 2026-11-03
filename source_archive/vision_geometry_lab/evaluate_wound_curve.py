"""V3 boundary/first-wrapped-course audit. Manual frame labels are report-only.

UI frame numbers are one-based. Never pass contact labels/timestamps to the
image detector; no near-edge or event threshold is fitted to these four frames.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw
from .geometry import analyze_frame,draw_overlay,write_image,dump_json

ROOT=Path(__file__).resolve().parents[1]
MANUAL_LABELS={246:'user_visible_gap',247:'user_nearly_contact',248:'user_contact'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(Path(__file__).parent/'outputs/wound_curve_v3'))
    parser.add_argument('--dense',action='store_true')
    args=parser.parse_args()
    cv2.setNumThreads(2)
    folder=Path(args.output); folder.mkdir(parents=True,exist_ok=True)
    source=ROOT/'现场视频/训练采集/双相机同步/dual_20260924_111500/left_DA9627065.avi'
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError('Cannot open '+str(source))
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    indices=list(range(238,257)) if args.dense else [245,246,247,251]
    indices=sorted(set(indices+[220,243,265,273,300]))
    rows=[]; thumbnails=[]
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,frame=cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode frame {index}')
        result=analyze_frame(frame)
        name=f'ui_{index+1:04d}_t{index/fps:06.3f}'
        result.update(source=str(source),frame_index=index,ui_frame=index+1,time_s=index/fps,
                      manual_reference=MANUAL_LABELS.get(index+1),manual_reference_used_in_detection=False)
        dump_json(folder/(name+'.json'),result)
        overlay=draw_overlay(frame,result)
        write_image(folder/(name+'.jpg'),overlay)
        write_image(folder/(name+'_diagnostic.jpg'),draw_overlay(frame,result,show_raw=True,show_exclusion=True,show_conic=True))
        arc=result['arc']; conic=result['conic']; wound=result.get('wound_curve',{})
        points=np.asarray(arc.get('points',[]),float)
        if len(points):
            lo=np.maximum(0,np.floor(points.min(axis=0)-[260,300]).astype(int))
            hi=np.minimum([frame.shape[1],frame.shape[0]],np.ceil(points.max(axis=0)+[260,250]).astype(int))
            crop=overlay[lo[1]:hi[1],lo[0]:hi[0]]
            write_image(folder/(name+'_focus.jpg'),crop)
            if index in [245,246,247,251]:
                write_image(folder/(name+'_original_focus.jpg'),frame[lo[1]:hi[1],lo[0]:hi[0]])
            im=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
            im.thumbnail((850,350))
            tile=Image.new('RGB',(870,420),'#17202a')
            tile.paste(im,((870-im.width)//2,65))
            text=ImageDraw.Draw(tile)
            text.text((10,8),f'UI frame {index+1} | {index/fps:.3f}s | manual={MANUAL_LABELS.get(index+1,"unlabeled")}',fill='white')
            text.text((10,26),f'local={arc.get("local_boundary_valid")} ellipse={conic["valid"]}',fill='white')
            text.text((10,44),f'wound={wound.get("status","not_integrated")}',fill='white')
            thumbnails.append(tile)
        row=dict(ui_frame=index+1,time_s=index/fps,manual_reference=MANUAL_LABELS.get(index+1),
                 observed_points=len(arc.get('points',[])),local_boundary_valid=arc.get('local_boundary_valid'),
                 local_segments=len(arc.get('local_boundary_segments',[])),
                 conic_valid=conic['valid'],conic_reason=conic.get('reason'),
                 conic_sensitivity_px=conic.get('sensitivity_px'),
                 rope_valid=result['rope']['valid'],wound_status=wound.get('status'),
                 wound_reason=wound.get('reason'),wound_identity_confirmed=wound.get('identity_confirmed',False),
                 elapsed_ms=result['timing_ms']['total'])
        rows.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    cap.release()
    for start in range(0,len(thumbnails),6):
        tiles=thumbnails[start:start+6]
        sheet=Image.new('RGB',(1740,420*((len(tiles)+1)//2)),'#17202a')
        for j,tile in enumerate(tiles):
            sheet.paste(tile,((j%2)*870,(j//2)*420))
        sheet.save(folder/f'contact_sheet_{start//6+1}.jpg')
    summary=dict(cases=len(rows),source=str(source),fps=fps,rows=rows,
                 annotation_interpretation='UI frame number is one-based; needs user correction if intended otherwise',
                 coarse_previous_annotation={'last_turn_start_s':24.3,'new_layer_start_s':27.3,'first_turn_end_s':30.0},
                 manual_reference_used_in_detection=False,
                 conic_available=sum(r['conic_valid'] for r in rows),
                 local_available=sum(bool(r['local_boundary_valid']) for r in rows),
                 wound_statuses=dict(Counter(r['wound_status'] for r in rows)),
                 median_ms=float(np.median([r['elapsed_ms'] for r in rows])))
    dump_json(folder/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
