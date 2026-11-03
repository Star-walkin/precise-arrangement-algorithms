"""Extract the user's red reference dots; analyze CLEAN original video pixels.

The annotated crop is never an algorithm input. Coordinates are recovered from
the saved pre-V3 crop recipe, verified dimensions, and non-red pixel agreement.
This is manual-guidance diagnostics, never an automatic-accuracy benchmark.
"""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from .geometry import read_image,write_image,dump_json,analyze_frame,draw_overlay

ROOT=Path(__file__).resolve().parent


def main():
    cv2.setNumThreads(2)
    previous=ROOT/'outputs/wound_before_v3'
    annotation=previous/'微信图片_20260926184325_152_12.png'
    original=read_image(previous/'ui_0247_t24.600_original_focus.jpg')
    marked=read_image(annotation)
    if original.shape!=marked.shape:
        raise ValueError('Annotated crop size changed: explicit image registration required')
    b,g,r=cv2.split(marked.astype(np.float32))
    red=((r>130)&(r-g>55)&(r-b>40)).astype(np.uint8)
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(red)
    components=[i for i in range(1,count) if 4<=stats[i,cv2.CC_STAT_AREA]<=600]
    points=centroids[components]
    if len(points)!=6:
        raise ValueError(f'Unexpected red-dot components: {len(points)}; inspect image manually')
    differences=np.mean(abs(marked.astype(float)-original.astype(float)),axis=2)
    clean=cv2.dilate(red,np.ones((15,15),np.uint8))==0
    agreement=float(np.percentile(differences[clean],95))
    if agreement>8:
        raise ValueError('Non-red image pixels changed too much; cannot assume identical crop alignment')
    v2=json.loads((previous/'ui_0247_t24.600.json').read_text('utf-8'))
    bounds=np.asarray(v2['arc']['points'],float)
    origin=np.maximum(0,np.floor(bounds.min(axis=0)-[260,300]).astype(int))
    points=points[np.argsort(points[:,0])]+origin
    source=Path(v2['source'])
    cap=cv2.VideoCapture(str(source)); cap.set(cv2.CAP_PROP_POS_FRAMES,v2['frame_index'])
    ok,frame=cap.read(); cap.release()
    if not ok:
        raise RuntimeError('Could not decode clean frame')
    end=origin+np.array([original.shape[1],original.shape[0]])
    fresh_crop=frame[origin[1]:end[1],origin[0]:end[0]]
    if fresh_crop.shape!=original.shape:
        raise ValueError('Fresh original video crop does not match reference dimensions')
    crop_p95=float(np.percentile(np.mean(abs(fresh_crop.astype(float)-original.astype(float)),axis=2),95))
    if crop_p95>8:
        raise ValueError('Fresh video crop differs from the saved reference; explicit registration required')
    result=analyze_frame(frame)
    guided_result=analyze_frame(frame,wound_seed_points=points)
    for value in (result,guided_result):
        value.update(source=str(source),frame_index=v2['frame_index'],ui_frame=247,time_s=24.6,
                     time_basis='decoded AVI frame index / nominal 10 fps; not exposure-calibrated',
                     annotation_image_fed_to_detector=False)
    output=ROOT/'outputs/wound_curve_v3/manual_247'
    reference=dict(annotation_source=str(annotation),clean_source=str(source),frame_index=v2['frame_index'],
                   ui_frame=247,time_s=24.6,crop_origin=origin.tolist(),points_original_px=points.tolist(),
                   original_frame_sha256=hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest(),
                   original_frame_shape=list(frame.shape),
                   annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
                   recovered_dot_count=len(points),fresh_video_crop_p95_mean_channel_difference=crop_p95,
                   nonred_p95_mean_channel_difference=agreement,
                   annotation_semantics='rough new-winding route points; center vs body-edge not yet independently verified',
                   manual_reference=True,annotation_image_fed_to_detector=False)
    dump_json(output/'reference.json',reference)
    dump_json(output/'automatic.json',result)
    dump_json(output/'guided.json',guided_result)
    measurements={}
    for mode,value in [('automatic',result),('guided',guided_result)]:
        paths=value['wound_curve'].get('candidates',[])
        distances=[]
        for candidate in paths:
            samples=np.asarray(candidate.get('points',[]),float)
            distances.append(np.min(np.linalg.norm(points[:,None,:]-samples[None,:,:],axis=2),axis=1).tolist() if len(samples) else [])
        measurements[mode]=dict(status=value['wound_curve']['status'],candidate_point_distance_px=distances,
                                identity_confirmed=value['wound_curve']['identity_confirmed'],
                                timing_ms=value['timing_ms'],wound_guidance=value['wound_guidance'],
                                score_kind='nearest sampled ridge distance to rough manual dots, not precision error')
        image=draw_overlay(frame,value,show_conic=False)
        color=(0,0,255) if mode=='guided' else (255,190,100)
        for i,point in enumerate(points):
            p=tuple(np.rint(point).astype(int))
            cv2.circle(image,p,10,color,3,cv2.LINE_AA)
            cv2.putText(image,str(i+1),(p[0]+12,p[1]-10),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        banner=('MANUAL-GUIDED CANDIDATE | red: input hints | IDENTITY NOT CONFIRMED' if mode=='guided'
                else 'AUTOMATIC CANDIDATE | blue: AFTER-THE-FACT reference, NOT INPUT | IDENTITY NOT CONFIRMED')
        def label(target,scale):
            cv2.rectangle(target,(0,0),(target.shape[1],round(40*scale)),(20,20,20),-1)
            cv2.putText(target,banner,(round(10*scale),round(28*scale)),cv2.FONT_HERSHEY_SIMPLEX,
                        .70*scale,(255,255,255),max(1,round(scale)),cv2.LINE_AA)
        focus=image[origin[1]:end[1],origin[0]:end[0]].copy()
        label(image,2)
        label(focus,1.4)
        write_image(output/(mode+'.jpg'),image)
        write_image(output/(mode+'_focus.jpg'),focus)
    dump_json(output/'summary.json',measurements)
    print(json.dumps(dict(reference=reference,comparisons=measurements),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
