"""R20 F3 point classes -> one crossing per class -> measured phase / interval.

Display and count share the same recovered curves. Model intervals are labelled
estimates; missing grooves are never invented to obtain a desired integer.
"""
import copy,sys,numpy as np
from common20 import BASE,plain,PITCH_PRIOR
sys.path.insert(0,str(BASE))
import transfer_engine as e
from geometry20 import boundary_at
from recover_points20 import recover as recover_points

def prepare(data,boundary):
    data=dict(data);top=data.get('analysis_top_curve_xy',boundary['top_curve_xy'])
    data['model_envelope']=e.model_envelope(data['surface'],np.asarray(top)*.5)
    for k in ('surface','exclusion','broad','clean','body','model_envelope'):data[k]=data[k].astype(np.float32,copy=False)
    return data

def refreshed_rejections(groups,points,d):
    # BG3 rejection y_at_d belongs to its initialization column. Re-evaluate
    # the candidate relevance filter at the NEW column from nearby real points.
    b=copy.deepcopy(groups);lookup={int(p[0]):p for p in points}
    for r in b['diagnostics']['rejections']:
        q=np.array([lookup[k] for k in r.get('point_ids',[]) if k in lookup])
        if len(q):
            near=q[np.argsort(abs(q[:,1]-d))[:min(9,len(q))]]
            r['y_at_d']=float(np.median(near[:,2]))
    return b

def at_column(groups,data,boundary,axis,d,*,recover=True):
    out=dict(d=float(d),available=False,reasons=[],crossings=[],rejected=[],curves=[],phase_N=None,geometry_N=None,local_pitch_px=None,
        count_observed_grooves=0,physical_contact_confirmed=False)
    bd=boundary_at(boundary,d);out['boundary']=bd
    if not bd['valid'] or axis is None:out['reasons'].append('current_axis_or_boundary_unavailable');return out
    abc=axis['line_abc'];active=float(-(abc[0]*d+abc[2])/abc[1]);out['active_y']=active
    if not 0<=active<2*data['surface'].shape[0]:
        out['reasons'].append('active_projection_outside_image');return out
    if recover:
        b=refreshed_rejections(groups,data['points'],d*.5)
        curves,logs=e.local.recover_classes(b,data,d*.5,bd['y']*.5,active*.5)
        curves,gap_logs=recover_points(curves,groups,data,d*.5,bd['y']*.5,active*.5)
    else:
        curves=[dict(c,xy=np.asarray(c['xy'])*.5,recovered_local=c.get('recovered_local',False)) for c in groups['curves']];logs=[];gap_logs=[]
    cross,rejected=e.local.local_crossings(curves,data['points'],data,d*.5)
    # The frozen counter's current-edge boolean means an available current
    # optical boundary; local visibility is independently enforced below.
    count,reasons,diag=e.local.SurfaceObserver(data['surface'].astype(np.uint8),float(data['pitch']))._count(
        d*.5,active*.5,cross,dict(edge_y=bd['y']*.5,edge_current_frame_confirmed=True),data['exclusion'])
    ys=np.array([c['y']*2 for c in cross]);pitch=float(count['local_pitch_px'])*2
    nearby=ys[(ys>=bd['y']-PITCH_PRIOR)&(ys<=max(active,bd['y'])+5*PITCH_PRIOR)]
    gaps=np.diff(nearby);plausible=gaps[(gaps>=.65*PITCH_PRIOR)&(gaps<=1.45*PITCH_PRIOR)]
    pitch_mad=float(np.median(abs(plausible-np.median(plausible)))/np.median(plausible)) if len(plausible) else None
    pitch_valid=len(plausible)>=3 and pitch_mad<=.12
    pitch_source='current_near_endpoint_groove_intervals'
    if not pitch_valid:
        # Occlusion of the first few rows need not hide all current pitch
        # evidence. Use the nearest visible lower rows as an explicitly
        # transferred pitch, with extra sensitivity, never as observed gaps.
        below=ys[(ys>=bd['y'])&(ys<=bd['y']+20*PITCH_PRIOR)]
        wider=np.diff(below);wider=wider[(wider>=.65*PITCH_PRIOR)&(wider<=1.45*PITCH_PRIOR)][:8]
        if len(wider)>=3:
            median=float(np.median(wider));relative=float(np.median(abs(wider-median))/median)
            if relative<=.12:
                pitch=median;pitch_mad=relative;pitch_valid=True;plausible=wider
                pitch_source='current_lower_visible_F3_rows_pitch_transfer'
    # Actual rejected classes inside the counting interval are unresolved,
    # not evidence that the accepted integer is complete.
    unresolved=[dict(c,y=c['y']*2) for c in rejected if bd['y']<2*c['y']<active and
        (not len(ys) or min(abs(ys-2*c['y']))>.35*pitch)]
    if unresolved:reasons.append('unresolved_local_class_in_count_interval')
    phase_ok=bool(count['count_valid'] and not unresolved)
    if not bd['locally_observed']:reasons.append('boundary_at_d_is_shape_estimate')
    if not axis['current_coil_side_measured']:reasons.append('active_axis_extrapolated_from_upstream')
    if not pitch_valid:reasons.append('insufficient_stable_current_F3_pitch')
    geom=(active-bd['y'])/pitch if pitch_valid else None
    width=float(axis['width_px'])/max(abs(float(abc[1])),.1)
    gapN=(active-bd['y']-.5*width)/pitch if pitch_valid else None
    unc=(axis['uncertainty_px']+bd['uncertainty_px'])/pitch if pitch_valid else None
    # Complete phase integrates observed intervals individually. Multiplying
    # its N by spacing variation would count that variation twice.
    if unc is not None and not phase_ok:unc+=abs(geom)*max(.02,float(pitch_mad))
    if unc is not None and pitch_source!='current_near_endpoint_groove_intervals':
        unc+=max(.2,.15*abs(geom));reasons.append('pitch_transferred_from_lower_visible_current_rows')
    out.update(available=pitch_valid,phase_N=float(count['continuous_pitch_distance']) if phase_ok else None,
        phase_complete=phase_ok,geometry_N=geom,contact_projected_gap_N=gapN,local_pitch_px=pitch if pitch_valid else None,
        current_pitch_intervals=len(plausible),pitch_relative_mad=pitch_mad,pitch_source=pitch_source,endpoint_sensitivity_N=unc,
        count_observed_grooves=int(count['count_observed_grooves']),reasons=list(dict.fromkeys(reasons)),
        crossings=[dict(c,y=c['y']*2) for c in cross],rejected=[dict(c,y=c['y']*2) for c in rejected],
        curves=[dict(c,xy=np.asarray(c['xy'])*2) for c in curves],recovery=logs,unowned_point_recovery=gap_logs,unresolved_local_classes=unresolved,
        count_diagnostics=diag,count_diagnostics_coordinate_scale=.5)
    return plain(out)

def measure(groups,data,boundary,axis,calibration):
    d=calibration.get('d');out=dict(available=False,status='unavailable',reasons=[],calibration_status=calibration['status'],
        d=d,d_interval=calibration.get('interval'),N=None,N_interval=None,physical_contact_confirmed=False)
    if d is None:out['reasons'].append('d_not_identified_for_this_motion_phase');return out
    if groups is None or data is None:out['reasons'].append('current_F3_observations_unavailable');return out
    if axis is None:out['reasons'].append('current_active_rope_axis_unavailable');return out
    fields=prepare(data,boundary);central=at_column(groups,fields,boundary,axis,d);out['central']=central
    # The display/count central class set stays fixed while d is perturbed;
    # no extra yellow-point class may be invented to stabilize sensitivity.
    shared=dict(groups,curves=central['curves']);samples=[]
    for dd in np.linspace(*calibration['interval'],5):
        q=at_column(shared,fields,boundary,axis,float(dd),recover=False)
        samples.append({k:q.get(k) for k in ('d','available','geometry_N','phase_N','contact_projected_gap_N','endpoint_sensitivity_N','local_pitch_px','reasons')})
    out['column_sensitivity_samples']=samples
    good=[q for q in samples if q['available'] and q.get('geometry_N') is not None]
    if not central['available'] or len(good)<4:
        out['reasons']=central['reasons']+['insufficient_F3_evidence_over_d_range'];return out
    values=[q['phase_N'] if q['phase_N'] is not None else q['geometry_N'] for q in good]
    lows=[v-q['endpoint_sensitivity_N'] for v,q in zip(values,good)];highs=[v+q['endpoint_sensitivity_N'] for v,q in zip(values,good)]
    contact_lo=min(q['contact_projected_gap_N']-q['endpoint_sensitivity_N'] for q in good)
    contact_hi=max(q['contact_projected_gap_N']+q['endpoint_sensitivity_N'] for q in good)
    n=central['phase_N'] if central['phase_N'] is not None else central['geometry_N'];interval=[float(min(lows)),float(max(highs))]
    interval=[min(interval[0],n),max(interval[1],n)]
    direct=bool(calibration['status']=='observed_bend_calibrated' and central['phase_complete'] and
        central['boundary']['locally_observed'] and axis['current_coil_side_measured'] and interval[1]-interval[0]<=1.)
    out.update(available=True,status='observed_pitch_distance' if direct else 'conditional_pitch_distance_estimate',N=float(n),
        N_interval=interval,contact_gap_interval=[float(contact_lo),float(contact_hi)],reasons=central['reasons'],
        d_sensitivity_N=float(max(q['geometry_N'] for q in good)-min(q['geometry_N'] for q in good)),
        count_definition='active_centerline_to_visible_coil_limit_in_current_F3_pitch_units',
        contact_gap_definition='projected_near_active_rope_edge_to_visible_coil_limit; not 3D contact',
        complete_F3_phase=central['phase_complete'],observed_groove_count=central['count_observed_grooves'])
    if interval[1]<0:out.update(status='projection_geometry_inconsistent',available=False,reasons=out['reasons']+['active_projection_above_visible_coil_limit'])
    return plain(out)
