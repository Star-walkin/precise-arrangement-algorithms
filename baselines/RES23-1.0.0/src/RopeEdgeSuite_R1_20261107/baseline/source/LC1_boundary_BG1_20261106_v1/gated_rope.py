"""Hard BG1 gate followed by the exact supplied 01 full-surface observation.

The offline C3/F2 historical fusion is NOT replayed on unrelated video frames.
The supplied surface_observation.py and five frozen dependencies are unchanged.
All raw observations are recomputed on the CURRENT validated visible ROI.
"""
from pathlib import Path
import importlib.util,sys,time
import cv2,numpy as np
from region_gate import segment
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'LC1_locked_column_20261105_v1'
sys.path.insert(0,str(BASE))
from lc1_surface import SurfaceObserver
spec=importlib.util.spec_from_file_location('_bg1_exact_supplied_observation',ROOT/'vendor/supplied_01/surface_observation.py')
core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
PITCH=23.29412648284868

def analyze(image,s1_record,detector=None,processing_scale=.5):
    start=time.perf_counter();regions=segment(image);r=regions['record'];mask=regions['surface_mask']
    result=dict(region=r,F3_called=False,F3=None,count_valid=False,display_N=None,
                reasons=list(r['reasons']),s1=s1_record['S1_tracker'],
                intersection_y_px=s1_record['intersection_y_px'],fixed_d_original_px=2238.7)
    if not r['valid']:
        result['elapsed_seconds']=time.perf_counter()-start;return result,regions
    exclusion=np.zeros_like(mask)
    exclusion[mask==0]=255
    tr=s1_record['S1_tracker']
    # Exclude only observed S1 pixels, never an extrapolated straight line.
    pts=tr.get('line_points_xy',[])
    if len(pts)>1:cv2.polylines(exclusion,[np.rint(pts).astype(np.int32)],False,255,round(.8*PITCH))
    scale=float(processing_scale)
    if scale not in (.5,1.):raise ValueError('Only validated original or half-resolution grids supported')
    # Region validation stays on original pixels. The unchanged supplied core
    # operates on a declared grid; its pitch, masks, d and S1 y share that grid.
    im=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale!=1 else image
    mm=((cv2.resize(mask,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_AREA)==255)*255).astype(np.uint8)
    ee=((cv2.resize(exclusion,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_AREA)>0)*255).astype(np.uint8)
    det=(detector or core.detect_surface)(dict(original=im,surface=mm,pitch=PITCH*scale,
                                  c3_record=None,c3_exclusion=ee))
    result['F3_called']=True
    observer=SurfaceObserver(mm,PITCH*scale)
    observed,tracks=observer._observed_segments(det)
    crossings,cross_diag=observer._crossings(tracks,2238.7*scale,det)
    edge=dict(edge_y=r['edge_y']*scale,edge_current_frame_confirmed=True,edge_source='BG1_current_yellow_surface_interface')
    active=s1_record['intersection_y_px'];active=None if active is None else active*scale
    counts,reasons,count_diag=observer._count(2238.7*scale,active,crossings,edge,det['exclusion_mask'])
    if counts.get('local_pitch_px') is not None:counts['local_pitch_px']/=scale
    if not tr.get('valid'):
        reasons.append('S1_not_valid');counts['count_valid']=False;counts['continuous_pitch_distance']=None
    result['F3']=dict(method='supplied_01_unchanged_full_surface_observation_core',
                     processing_scale=scale,diagnostic_coordinates='processing_grid_pixels',
                     display_coordinates='original_pixels',local_pitch_coordinates='original_pixels',
                     historical_C3_F2_fusion_replayed=False,**counts,
                     crossings=crossings,diagnostics=det['diagnostics'],count_diagnostics=count_diag,
                     raw_observed_polylines_xy=[(np.asarray(p)/scale).tolist() for p in observed],
                     raw_observed_points_xy=[[p['x']/scale,p['y']/scale] for p in det['points']],
                     fitted_curves=[dict(curve_id=c.get('curve_id'),xy=(np.asarray(c.get('sampled_points_xy',[]))/scale).tolist(),
                                        point_states=c.get('point_states',[])) for c in det['fit']['curves']])
    result['count_valid']=counts['count_valid'];result['reasons']=reasons
    val=counts.get('continuous_pitch_distance')
    if counts['count_valid'] and val is not None and 0<=val<=5:result['display_N']=val
    result['elapsed_seconds']=time.perf_counter()-start
    return result,regions

if __name__=='__main__':
    import json
    cv2.setNumThreads(1)
    from inspect_regions import SOURCE,overlay
    cap=cv2.VideoCapture(str(SOURCE));i=121;cap.set(1,i);ok,a=cap.read();assert ok
    sr=json.loads((ROOT.parent/f'LC1_fixed_d2238p7_full_20261106_v1/analysis_anchored/frames/frame_{i:06d}.json').read_text(encoding='utf8'))
    out=ROOT/'pilot_full_surface';out.mkdir(exist_ok=True)
    r,m=analyze(a,sr)
    (out/f'{i:03d}.json').write_text(json.dumps(r,ensure_ascii=False),encoding='utf8')
    p=overlay(a,m)
    if r['F3']:
        for line in r['F3']['raw_observed_polylines_xy']:cv2.polylines(p,[np.rint(line).astype(np.int32)],False,(0,255,0),2)
    cv2.imencode('.jpg',cv2.resize(p,(1638,1200)))[1].tofile(str(out/f'{i:03d}.jpg'))
    print({k:r[k] for k in ['F3_called','count_valid','display_N','reasons','elapsed_seconds']},flush=True)
