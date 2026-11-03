"""Raw-frame V2 contact-neighborhood audit; annotations never enter detection."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
from collections import Counter
import cv2
import numpy as np
from .geometry import analyze_frame, draw_overlay, write_image, dump_json
from .arc_detector import detect_arc

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(Path(__file__).parent/'outputs/tangency_v2'))
    parser.add_argument('--step',type=float,default=.5)
    args = parser.parse_args()
    if not np.isfinite(args.step) or args.step <= 0:
        parser.error('--step must be positive')
    cv2.setNumThreads(2)
    folder=Path(args.output)
    folder.mkdir(parents=True,exist_ok=True)
    base=ROOT/'现场视频/训练采集/双相机同步'
    cases=[('clear',base/'dual_20260924_111500/left_DA9627065.avi',22.,32.,
            {'last_turn_start_s':24.3,'new_layer_start_s':27.3,'new_layer_first_turn_end_s':30.0}),
           ('occluded',base/'dual_20260924_111143/right_DA9626975.avi',10.,20.,
            {'last_turn_start_s':11.4,'new_layer_start_s':14.4,'new_layer_first_turn_end_s':16.8,
             'occlusion_start_s':13.8,'occlusion_end_s':18.6,'manual_hit_s':17.1})]
    all_rows=[]
    for tag,path,start,end,reference in cases:
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError('Cannot open '+str(path))
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        indices=sorted(set(round(s*fps) for s in np.r_[np.arange(start,end+.0001,args.step),
                                                        list(reference.values()),[26.5] if tag=='clear' else []]))
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode {path} frame {index}')
            result=analyze_frame(frame)
            name=f'{tag}_{index/fps:06.3f}'
            result.update(source=str(path),frame_index=index,time_s=index/fps,
                          annotations_used_in_detection=False)
            if tag=='clear' and abs(index/fps-26.5)<.001:
                # Paired audit: same V2 raw-edge extractor, with/without exclusion.
                raw=detect_arc(frame)
                band=result['exclusion']
                def inside(points):
                    pts=np.asarray(points,float)
                    return int(np.sum(np.abs(pts@np.asarray(band['normal'])-band['rho_px']) <= band['half_width_px'])) if len(pts) and band.get('valid') else None
                audit=dict(unmasked_point_count=len(raw.get('points',[])),
                           masked_point_count=len(result['arc'].get('points',[])),
                           unmasked_points_in_band=inside(raw.get('points',[])),
                           masked_points_in_band=inside(result['arc'].get('points',[])),
                           comparison_kind='same_v2_extractor_with_vs_without_corridor')
                dump_json(folder/'26.5_exclusion_audit.json',audit)
                write_image(folder/'26.5_original.jpg',frame)
                write_image(folder/'26.5_diagnostics.jpg',draw_overlay(frame,result,show_raw=True,show_exclusion=True))
                points=np.asarray(result['arc'].get('points',[]),float)
                if len(points):
                    lo=np.maximum(0,np.floor(points.min(axis=0)-[300,450]).astype(int))
                    hi=np.minimum([frame.shape[1],frame.shape[0]],np.ceil(points.max(axis=0)+[300,200]).astype(int))
                    overlay=draw_overlay(frame,result)
                    write_image(folder/'26.5_focus.jpg',overlay[lo[1]:hi[1],lo[0]:hi[0]])
                unmasked_result=dict(result,arc=raw)
                unmasked_result.pop('conic',None)
                unmasked_result['relation']={}
                write_image(folder/'26.5_unmasked_points.jpg',draw_overlay(frame,unmasked_result,show_raw=True,show_exclusion=True))
            dump_json(folder/(name+'.json'),result)
            write_image(folder/(name+'.jpg'),draw_overlay(frame,result))
            r=result['relation']; c=result['conic']; rope=result['rope']
            row=dict(name=name,clip=tag,time_s=index/fps,
                     arc_valid=result['arc']['valid'],conic_valid=c['valid'],
                     conic_reason=c.get('reason'),rope_valid=rope['valid'],
                     rope_width_px=rope.get('width_px'),relation_status=r['status'],
                     signed_center_clearance_px=r.get('signed_center_clearance_px'),
                     error_budget_px=r.get('estimated_error_budget_px'),
                     last_turn_candidate=r.get('last_turn_candidate',False),
                     last_turn_confirmed=r['last_turn_confirmed'],
                     elapsed_ms=result['timing_ms']['total'])
            all_rows.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
        cap.release()
    summary=dict(cases=len(all_rows),rows=all_rows,
                 reference_annotations={tag:ref for tag,_,_,_,ref in cases},
                 annotations_used_in_detection=False,
                 scope='availability and temporal consistency only; not ground-truth pixel accuracy',
                 statuses=dict(Counter(r['relation_status'] for r in all_rows)),
                 median_ms=float(np.median([r['elapsed_ms'] for r in all_rows])))
    dump_json(folder/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
