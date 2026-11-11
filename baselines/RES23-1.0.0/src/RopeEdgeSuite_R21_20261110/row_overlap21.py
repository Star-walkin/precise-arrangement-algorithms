"""First visible winding-row overlap: optical evidence, NOT layer identity.

Uses actual accepted F3 crossings and the current active-rope centerline.
At least three uninterrupted inter-groove pitches are required to avoid
mistaking a missing top groove for an exposed outer winding row.
"""
import numpy as np

def observe(m,axis):
    out=dict(eligible=False,overlap=False,physical_layer_change_confirmed=False)
    c=m.get('central') or {};b=c.get('boundary') or {};p=c.get('local_pitch_px')
    if not m.get('available') or m.get('N') is None or m['N']>2.5 or not p:return out
    if not axis or not axis.get('current_coil_side_measured') or b.get('occluded') or not b.get('inside_current_anchor_hull'):return out
    if b.get('uncertainty_px',1e9)>.75*p:return out
    cross=sorted([x for x in c['crossings'] if x['y']>b['y']],key=lambda x:x['y'])
    if len(cross)<4:return out
    ys=np.array([x['y'] for x in cross[:4]]);gaps=np.diff(ys)/p
    if np.any((gaps<.7)|(gaps>1.3)):return out
    top_offset=(ys[0]-b['y'])/p
    if not .7<=top_offset<=1.65:return out
    overlap=(ys[0]-c['active_y'])/p
    out.update(eligible=True,overlap=bool(.15<=overlap<=.85),first_groove_y=float(ys[0]),active_y=c['active_y'],pitch=p,
        first_groove_from_boundary_pitch=float(top_offset),center_position_within_outermost_visible_row=float(overlap),
        boundary_local=b.get('locally_observed'),semantics='projected_active_center_overlaps_first_resolved_winding_row_not_physical_layer_confirmation')
    return out

def summarize(rows,fps):
    events=[];hist=[];active=False
    for r in rows:
        t=r['time_s'];hist=[x for x in hist if t-x['time_s']<=.5+1e-6];hist.append(r)
        hits=[x for x in hist if x['evidence']['overlap']]
        if len(hits)>=4 and hits[-1]['time_s']-hits[0]['time_s']>=.3-1e-6 and not active:
            events.append(dict(start_s=hits[0]['time_s'],confirmed_s=t,physical_layer_change_confirmed=False,
                interpretation='first_visible_row_overlap_candidate',support_frames=[x['frame'] for x in hits]));active=True
        if not hits:active=False
    return events
