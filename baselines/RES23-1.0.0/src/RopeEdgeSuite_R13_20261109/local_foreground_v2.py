"""Current-pixel foreground-width guard on source-bound BG5.1 proposals.

BG6.2 adds a current local count-strip check. BG5 remains a candidate generator. A smooth yellow/gray transition is not
accepted as coil when a coherent broad foreground highlight occupies the
immediate count neighborhood. Rejection supplies no replacement edge.
"""
from copy import deepcopy
import cv2,numpy as np
from common import D,PITCH,plain

PARAMETERS=dict(local_half_width_pitch=3.,column_step_px=8,maximum_single_bright_band_pitch=.85,
    coherent_foreground_span_pitch=1.5,local_foreground_fraction=.35,
    below_reference_band_pitch=[4.,8.],detection_band_pitch=[-.5,6.],absolute_brightness_floor=100.,
    relative_brightness_offset=30.,relative_MAD_multiplier=4.,brightness_ceiling=200.,scope='local_count_neighborhood')

def intervals(hits):
    e=np.diff(np.r_[False,hits,False].astype(np.int8))
    return list(zip(np.flatnonzero(e==1),np.flatnonzero(e==-1)))

def local_count_guard(image,region,d=D,pitch=PITCH):
    curve=np.asarray(region.get('top_curve_xy',[]),float).reshape(-1,2)
    result=dict(valid=False,reasons=[],columns=[],foreground_polygons_xy=[],parameters=PARAMETERS,
        semantics='current_broad_foreground_test_not_semantic_contact_or_full_tool_segmentation')
    if not len(curve) or curve[0,0]>d-3*pitch or curve[-1,0]<d+3*pitch:
        result['reasons']=['current_boundary_search_support_missing'];return result
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Test all currently proposed interface columns; the count gate is local.
    xs=d+np.arange(-int(3*pitch/8),int(3*pitch/8)+1)*8
    offsets=np.arange(-pitch,8*pitch,.5)
    for x in xs:
        edge=np.interp(x,curve[:,0],curve[:,1]);slope=(np.interp(x+12,curve[:,0],curve[:,1])-np.interp(x-12,curve[:,0],curve[:,1]))/24
        xx=np.arange(x-6,x+7,2);X=np.tile(xx,(len(offsets),1)).astype(np.float32)
        Y=(edge+offsets[:,None]+slope*(xx[None,:]-x)).astype(np.float32)
        profile=np.median(cv2.remap(gray,X,Y,cv2.INTER_LINEAR),axis=1)
        below=profile[offsets>=4*pitch]
        threshold=min(200.,max(100.,float(np.median(below)+4*np.median(abs(below-np.median(below)))+30)))
        hits=(profile>threshold)&(offsets<6*pitch)&(offsets>-.5*pitch)
        spans=intervals(hits);lo,hi=max(spans,key=lambda q:q[1]-q[0]) if spans else (0,0)
        width=(hi-lo)*.5;broad=width>.85*pitch
        row=dict(x=float(x),boundary_y=float(edge),bright_width_px=width,threshold=threshold,broad=broad,
            bright_start_y=None if not spans else float(edge+offsets[lo]),
            bright_end_y=None if not spans else float(edge+offsets[min(hi,len(offsets)-1)]))
        result['columns'].append(row)
    flags=np.array([r['broad'] for r in result['columns']]);groups=intervals(flags)
    local=abs(xs-d)<=3*pitch;local_flags=flags[local]
    longest=max(((hi-lo)*8 for lo,hi in intervals(local_flags)),default=0)
    rejected_ranges=[]
    for lo,hi in groups:
        if (hi-lo)*8<1.5*pitch:continue
        group=result['columns'][lo:hi]
        top=[[r['x'],r['bright_start_y']-.6*pitch] for r in group]
        bottom=[[r['x'],r['bright_end_y']+.8*pitch] for r in reversed(group)]
        result['foreground_polygons_xy'].append(top+bottom)
        rejected_ranges.append([group[0]['x']-.5*pitch,group[-1]['x']+.5*pitch])
    fraction=float(local_flags.mean()) if len(local_flags) else 1.
    bad=bool(fraction>=.35 and longest>=1.5*pitch)
    # Also reject a coherent occluder directly on the count column even when
    # it covers fewer than35% of the wide local test window.
    if any(lo-pitch<=d<=hi+pitch for lo,hi in rejected_ranges):bad=True
    result.update(valid=not bad,broad_fraction=fraction,longest_broad_span_px=float(longest),
        rejected_column_ranges_xy=rejected_ranges,
        median_bright_width_px=float(np.median([r['bright_width_px'] for r in result['columns'] if abs(r['x']-d)<=3*pitch])),
        reasons=['broad_foreground_occludes_count_interface'] if bad else [])
    return plain(result)
