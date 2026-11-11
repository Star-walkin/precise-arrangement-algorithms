"""Recheck rejected frames with all currently observed flange fragments."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import cv2,copy
from common20 import *
from roi22 import masks
from groups20 import group
from geometry22 import measure as geometry
from count20 import measure

def generate():
    src=PARENT/'RopeEdgeSuite_R11_20261108/region_bg5_refined.py'
    text=src.read_text(encoding='utf8')
    old='    yellow=largest(yellow)\n'
    assert text.count(old)==1
    text=text.replace(old,'    yellow,selection=yellow_fragments(all_yellow_components,pitch,s)\n')
    text=text.replace("version='BG5_current_pixel_region_gate_v2'","version='BG5_current_fragments_R20_24'")
    text=text.replace('ytol=max(.75*stats[main,cv2.CC_STAT_HEIGHT]', 'ytol=max(1.5*stats[main,cv2.CC_STAT_HEIGHT]')
    text=text.replace("    out=dict(record=rec,yellow_mask=", "    rec['yellow_fragment_association']=selection\n    out=dict(record=rec,yellow_mask=")
    (ROOT/'region_fragments24.py').write_text(text,encoding='utf8')

def process(task):
    id,i=task;cv2.setNumThreads(1)
    item=next(x for x in plan() if x['id']==id);r=record(id,i)
    cp=ROOT/'counts_v1'/id/'frames'/f'{i:06d}.json.gz'
    with gzip.open(cp,'rt',encoding='utf8') as f:old=json.load(f)
    assert not old['measurement']['available']
    cal=read(ROOT/'calibration'/f'{id}.json');phase=next(x for x in cal['epochs'] if x['start']<=i<x['end'])
    assert phase['d'] is not None and i not in phase['far_guard_accepted_frames']
    cap=cv2.VideoCapture(item['source']);cap.set(1,i);ok,raw=cap.read();cap.release();assert ok and pixelsha(raw)==r['source_pixel_sha256']
    im=cv2.flip(raw,1) if item['camera']=='right' else raw
    bg5=load('_fragments24',ROOT/'region_fragments24.py');proposal=bg5.segment(im,d=phase['d'])['record']
    # The prior is strictly earlier accepted shapes, never the current fit or a future frame.
    sys.path.insert(0,str(R18))
    import boundary_bg9 as bg9
    from boundary_bg10 import Boundary
    bg9.D=phase['d']
    b=Boundary();b.shapes=[s for s in r['boundary'].get('shape_state',[]) if s['frame']<i]
    b.last_shape_frame=max((s['frame'] for s in b.shapes),default=None)
    if not b.shapes and i>0:
        prev=record(id,i-1)['boundary'];b.shapes=[s for s in prev.get('shape_state',[]) if s['frame']<i];b.last_shape_frame=max((s['frame'] for s in b.shapes),default=None)
    boundary=b.step(im,proposal,r['track'],i)
    from current_boundary23 import recover
    boundary,recovery=recover(im,proposal,r['track'],boundary,phase['interval'],i)
    rr=copy.deepcopy(r);rr.update(proposal=proposal,boundary=boundary)
    out=ROOT/'fragments24_trials'/id;out.mkdir(parents=True,exist_ok=True)
    surface,mask,meta,top=masks(im,rr)
    print('PRE',id,i,proposal['valid'],proposal['reasons'],boundary['fit_valid'],boundary['reasons'],'mask',np.count_nonzero(mask),flush=True)
    dump(out/f'{i:06d}_frontend.json',rr)
    if not np.count_nonzero(mask):return dict(id=id,frame=i,frontend=rr,measurement=dict(available=False,reasons=boundary['reasons']),old_count_file_sha256=sha(cp))
    ref=load('_fragments24_F3',PARENT/'RopeEdgeSuite_R12_20261109/run_full.py')
    data=ref.extract_current(im,mask,rr['track']);data['analysis_top_curve_xy']=top;data['mask_sha256']=np.asarray(pixelsha(mask))
    field=out/f'{i:06d}.npz';np.savez_compressed(field,**data)
    g=group(data,dict(boundary,top_curve_xy=top),phase['d']);a=geometry(im,rr,g);m=measure(g,data,boundary,a['axis'],phase)
    result=dict(id=id,frame=i,frontend=rr,source_pixel_sha256=pixelsha(raw),analysis_pixel_sha256=pixelsha(im),roi=meta,mask_sha256=pixelsha(mask),field=str(field),field_sha256=sha(field),points=len(data['points']),groups=g,geometry=a,measurement=m,old_count_file_sha256=sha(cp),producer_sha256=sha(__file__),segment_code_sha256=sha(ROOT/'region_fragments24.py'))
    dump(out/f'{i:06d}.json',result)
    print(id,i,'proposal',proposal['valid'],proposal['reasons'],'boundary',boundary['fit_valid'],boundary['reasons'],'count',m['available'],m.get('N'),m.get('N_interval'),m['reasons'],flush=True)
    return result

if __name__=='__main__':
    generate();process(('dual_20260923_113029__left',41))
