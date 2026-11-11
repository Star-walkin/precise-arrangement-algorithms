"""Gate -> fresh 01 yellow points -> global classes -> continuous models/count."""
import time,hashlib
import numpy as np
from inputs import BASE,SurfaceObserver,PITCH,SCALE,sha
from global_points import group_and_fit,intersections,model_envelope

def analyze(image,old,data):
    tick=time.perf_counter()
    assert hashlib.sha256(image.tobytes()).hexdigest()==old['source_pixel_sha256']
    r={k:old[k] for k in ('region','s1','intersection_y_px','fixed_d_original_px','frame_index',
        'nominal_time_s','source_pixel_sha256','source_video_sha256','masks')}
    r.update(version='BG3-point-global-v2',BG3=None,count_valid=False,display_N=None,
        region_gate_passed=bool(old['region']['valid']),reasons=list(old['region']['reasons']),
        reused_BG1_record_sha256=sha(BASE/'frames'/f"{old['frame_index']:06d}.json"))
    if not r['region_gate_passed']:
        assert data is None;r['elapsed_model_s']=time.perf_counter()-tick;return r
    assert data is not None
    r['raw_evidence_cache_origin']=str(data['raw_evidence_cache_origin'])
    data['model_envelope']=model_envelope(data['surface'],np.asarray(old['region']['top_curve_xy'])*SCALE)
    d=2238.7*SCALE
    try:curves,diagnostics,points=group_and_fit(data,d)
    except ValueError as e:
        if str(e)!='insufficient_shared_shape_points':raise
        r['reasons'].append(str(e));r['elapsed_model_s']=time.perf_counter()-tick;return r
    crossings,rejected=intersections(curves,points,data,d)
    edge=old['region']['edge_y'];active=old['intersection_y_px']
    count,reasons,cd=SurfaceObserver(data['surface'],float(data['pitch']))._count(d,
        None if active is None else active*SCALE,crossings,
        dict(edge_y=edge*SCALE,edge_current_frame_confirmed=True),data['exclusion'])
    uncertain=[]
    if active is not None and active>=edge:
        lo=(edge-PITCH)*SCALE;hi=(active+PITCH)*SCALE
        uncertain=[q for q in rejected if lo<=q['y']<=hi]
        for q in diagnostics['rejections']:
            if q.get('domain',[1,0])[0]<=d<=q.get('domain',[1,0])[1] and lo<=q.get('y_at_d',-1)<=hi:
                uncertain.append(dict(y=q['y_at_d'],reasons=q['reasons']))
    if uncertain:reasons.append('uncertain_class_near_count_interval')
    if not old['s1']['valid']:reasons.append('S1_not_valid')
    reasons=list(dict.fromkeys(reasons));count['count_valid']=bool(count['count_valid'] and not reasons)
    if not count['count_valid']:count['continuous_pitch_distance']=None
    count['count_fitted_grooves']=count.pop('count_observed_grooves');count.pop('observed_groove_count_lower_bound',None)
    count['local_pitch_px']/=SCALE
    count['count_status']='global_class_curve_phase' if count['count_valid'] else 'unresolved_global_class_curve_phase'
    for c in curves:c['xy']/=SCALE
    for c in crossings:c['y']/=SCALE
    r.update(BG3=dict(curves=curves,crossings=crossings,**count,diagnostics=diagnostics,
        raw_points=points,raw_point_columns=['point_id','x','y','score','width','slope','local_pitch','local_track_id'],
        raw_points_and_diagnostics_coordinates='half_resolution_processing_pixels',
        curves_and_crossings_coordinates='original_pixels',count_diagnostics=cd,
        uncertain_count_evidence=uncertain,crossing_rejections=rejected,
        supported_crossings=sum(c['status']=='image_supported_fitted_curve' for c in crossings),
        inferred_crossings=sum(c['status']=='point_bracketed_model_interpolation' for c in crossings)),
        reasons=reasons,count_valid=count['count_valid'],elapsed_model_s=time.perf_counter()-tick)
    val=count['continuous_pitch_distance']
    if count['count_valid'] and val is not None and 0<=val<=5:r['display_N']=val
    return r
