"""Synthetic display-contract and layout check; never reads a source frame."""
from pathlib import Path
import hashlib,json,sys
import cv2
import numpy as np
from render_panels import render_frame,_segments

def main():
    here=Path(__file__).resolve().parent
    bgr=np.full((3000,4096,3),26,np.uint8)
    cv2.ellipse(bgr,(2150,2310),(850,580),0,0,360,(85,85,85),-1)
    for y in range(1800,2900,40):cv2.line(bgr,(1350,y),(2950,y),(120,120,120),4)
    p=np.array([[350,2050],[700,2140],[1100,2240],[1500,2335],[1900,2400],[2300,2430],[2700,2435]],float)
    cv2.polylines(bgr,[p.astype(np.int32)],False,(215,215,215),16,cv2.LINE_AA)
    yy,xx=np.mgrid[:975,:2048];center=np.interp(xx[0]*2,p[:,0],p[:,1])
    heat=(.07+.85*np.exp(-.5*((yy*2+1050-center[None,:])/14)**2)).astype(np.float32)
    candidate=dict(id='synthetic_00',points_xy=p.tolist(),point_status=['entry_condition','image_supported','image_supported','image_supported','heatmap_supported','raw_a7_supported','inferred_or_weak'])
    engine=dict(raw_candidates=[candidate,dict(id='synthetic_01',points_xy=(p+[0,90]).tolist())],surface_tracks=[
        dict(points_xy=[[1350,y],[1800,y-25],[2300,y-35],[2900,y]],inferred=[False,False,False,True]) for y in range(1850,2800,50)])
    temporal=dict(status='ACCEPTED',accepted=True,accepted_candidate_index=0,points_xy=p.tolist(),reference_points_xy=None,temporal_confirmed=True,temporal_history_count=5,epoch=1)
    meta=dict(video_id='SYNTHETIC / 非真实检测结果',media_time_s=12.4,detection_time_s=12.,heat_age_s=.4,detection_age_s=.4,source_frame_index=124,analysis_hz=2,playback_fps=5)
    before_bgr=bgr.copy();before_heat=heat.copy()
    accepted=render_frame(bgr,heat,engine,temporal,meta)
    assert accepted.shape==(1280,1600,3) and accepted.dtype==np.uint8
    assert np.array_equal(bgr,before_bgr) and np.array_equal(heat,before_heat)
    stale=dict(status='STALE',accepted=False,accepted_candidate_index=None,points_xy=None,reference_points_xy=p.tolist(),reference_age_s=.5,temporal_history_count=5,epoch=1)
    st=render_frame(bgr,None,engine,stale,meta)
    unknown=render_frame(bgr,None,engine,dict(status='UNKNOWN',accepted=False,epoch=1),meta)
    body=lambda a:a[694:1280,800:]
    isgreen=lambda a:(a[...,1]>200)&(a[...,0]<100)&(a[...,2]<100)
    assert isgreen(body(accepted)).sum()>50
    assert isgreen(body(st)).sum()==0 and isgreen(body(unknown)).sum()==0
    assert np.all(st[80,850]==40) and np.all(unknown[80,850]==40),'Missing heat must be neutral gray, not zero-response blue'
    expired=render_frame(bgr,None,engine,stale,meta|dict(display_status='UNKNOWN'))
    assert np.array_equal(body(expired),body(unknown)),'Caller UNKNOWN override must suppress old reference'
    invalid=False
    try:render_frame(bgr,heat,engine,temporal|dict(accepted_candidate_index=8),meta)
    except ValueError:invalid=True
    assert invalid
    dash=np.zeros((30,240,3),np.uint8)
    _segments(dash,np.array([[10.,15.],[220.,15.]]),(255,255,255),1,dashed=True)
    lit=np.any(dash[15]>80,axis=1)
    assert lit[10:221].sum()<160 and lit[10:221].sum()>60
    assert np.sum(lit[1:]!=lit[:-1])>=30,'Sparse old-reference path must have visible fixed-size dash gaps'
    files={}
    for name,array in [('synthetic_preview.png',accepted),('synthetic_states.png',np.concatenate([accepted[640:,800:],st[640:,800:],unknown[640:,800:]],axis=1))]:
        ok,data=cv2.imencode('.png',array);assert ok;(here/name).write_bytes(data.tobytes())
        files[name]=hashlib.sha256((here/name).read_bytes()).hexdigest()
    result=dict(command=sys.argv,exit_code=0,source_sha256=hashlib.sha256((here/'render_panels.py').read_bytes()).hexdigest(),
        checks=['1600x1280_BGR_uint8','inputs_unchanged','accepted_uses_actual_candidate_index','invalid_accepted_index_rejected',
                'STALE_no_green_measurement','UNKNOWN_no_green_measurement','missing_heat_neutral_gray','expired_reference_hidden_by_caller_override',
                'sparse_two_point_reference_uses_fixed_screen_pixel_dashes'],
        previews=files,synthetic_only=True,source_frames_read=0,device_access=False,
        scope='Rendering contracts only; does not validate a rope identity, path or video timing.')
    (here/'render_check_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':main()
