"""Live current-frame B750/H6 -> conditioned S1 frontend for LC1.

Both supplied algorithms run on the ORIGINAL unmirrored image. Only compact
geometry is mirrored for the downstream common right-entering coordinate frame.
The upstream fresh F3/D4 scene adapter is documented as a new video adaptation;
this module does not claim reproduction of unavailable historical scene caches.
"""
from __future__ import annotations

import time
import numpy as np


def points_to_canonical(points, width, mirror):
    if points is None:
        return []
    a=np.asarray(points,dtype=float).reshape(-1,2).copy()
    if not np.isfinite(a).all():
        raise ValueError('Non-finite current model point')
    if mirror:
        a[:,0]=width-1-a[:,0]
    return a.tolist()


def line_to_canonical(abc,width,mirror):
    if abc is None:
        return None
    q=np.asarray(abc,dtype=float).reshape(3).copy()
    if not np.isfinite(q).all():
        return None
    if mirror:
        q[0],q[2]=-q[0],q[2]+q[0]*(width-1)
    norm=float(np.hypot(q[0],q[1]))
    if norm<1e-9:
        return None
    q/=norm
    if q[1]<0:
        q=-q
    return q.tolist()


def estimate_front_turn(front,pitch_px):
    """Use observed curve points and the independent current S1 free anchor.

    The complete-path flag remains an additional separate calibration gate.
    A fitted curve and an input seed do not turn missing spans into observations.
    """
    from lc1_real_path_probe import estimate_turn_observed_anchored
    p1=front.get('diagnostics',{}).get('P1_H6',{})
    evidence=p1.get('diagnostics',{})
    checks=evidence.get('whole_path_support_checks',{})
    eligible=bool(checks.get('genuine_H6_attached_candidate') is True and
                  checks.get('ambiguous') is False and
                  checks.get('direction_selection_allowed') is True)
    result=estimate_turn_observed_anchored(
        front.get('path_xy',[]),evidence.get('path_point_support',[]),
        evidence.get('path_stage',[]),front.get('line_abc'),
        free_line_points_xy=front.get('line_points_xy',[]),
        free_side='right',pitch_px=pitch_px,connection_eligible=eligible)
    result['estimator']='LC1.2_observed_curve_with_current_S1_anchor'
    result['whole_path_evidence_gate_separate']=True
    result['P1_whole_path_candidate_supported']=front.get('whole_path_candidate_supported') is True
    return result


class P1S1Frontend:
    def __init__(self,surface_mask_original,pitch_px,model_path,*,camera,
                 s1_source_dir=None,threads=2,surface_mode='full',s2_audit=False):
        from lc1_p1_adapter import P1H6Adapter
        from lc1_s1_adapter import S1Adapter
        self.camera=camera
        self.mirror=camera=='left'
        self.width=int(surface_mask_original.shape[1])
        self.p1=P1H6Adapter(surface_mask_original,pitch_px,model_path,threads=threads,camera=camera,
                            surface_mode=surface_mode)
        self.s1=S1Adapter(**({'source_dir':s1_source_dir} if s1_source_dir is not None else {}))
        self.s2=None
        if s2_audit:
            from lc1_s2_adapter import S2Adapter
            self.s2=S2Adapter()

    def step(self,bgr_original,t,frame_index):
        started=time.perf_counter()
        p1=self.p1.step(bgr_original)
        p1_seconds=time.perf_counter()-started
        seed=p1.get('seed_xy')
        s1_start=time.perf_counter()
        s1=self.s1.step(bgr_original,seed,seed_source='current_frame_P1_A7_auto_entry')
        s1_seconds=time.perf_counter()-s1_start
        s2=None;s2_seconds=0.
        if getattr(self,'s2',None) is not None:
            s2_started=time.perf_counter()
            s2=self.s2.step_auto(bgr_original,e1_xy=None,
                                 surface_mask=self.p1.surface,pitch_px=self.p1.pitch)
            s2_seconds=time.perf_counter()-s2_started
        line=line_to_canonical(s1.get('line_abc'),self.width,self.mirror) if s1.get('valid') is True else None
        path=points_to_canonical(p1.get('path_xy'),self.width,self.mirror)
        measured=points_to_canonical(s1.get('line_points_xy'),self.width,self.mirror) if line is not None else []
        supported=bool(p1.get('whole_path_candidate_supported') is True and line is not None and len(path)>=24)
        # Invalid/PARTIAL source paths remain available as diagnostics. Their
        # segment support governs visualization; they cannot authorize d.
        p1_diagnostics=p1.get('diagnostics',{})
        segment_support=p1_diagnostics.get('path_segment_support')
        if segment_support is None:
            segment_support=[supported]*(max(0,len(path)-1))
        if len(segment_support)!=max(0,len(path)-1):
            raise ValueError('P1 segment support does not match its path')
        return dict(
            source='LC1 live B750+H6 (fresh F3/D4 upstream) + frozen IR-GRAD-S1.0.0',
            line_abc=line,line_points_xy=measured,path_xy=path,
            whole_path_candidate_supported=supported,
            diagnostics=dict(
                frame_index=int(frame_index),time_s=float(t),
                source_coordinate_system='original_unmirrored_image_xy',
                output_coordinate_system='canonical_right_entering_xy',
                internal_left_mirror=self.mirror,
                path_segment_support=[bool(v) for v in segment_support],
                seed_xy_original=seed,
                seed_xy_canonical=points_to_canonical(seed,self.width,self.mirror),
                S1=s1,P1_H6=p1,S2=s2,
                p1_seconds=p1_seconds,s1_seconds=s1_seconds,s2_seconds=s2_seconds,
                S2_is_independent_optical_proposal_audit=s2 is not None,
                S2_used_for_C_or_whole_path_authorization=False,
                P1_whole_path_candidate_supported=p1.get('whole_path_candidate_supported') is True,
                S1_current_bilateral_line_valid=line is not None,
                P1_model_forward_count=int(p1.get('model_forward_count',0)),
                S1_core_actually_executed=s1.get('diagnostics',{}).get('s1_core_executed') is True,
                source_algorithms_actually_executed=bool(p1.get('model_forward_count',0)>0 and
                    s1.get('diagnostics',{}).get('s1_core_executed') is True),
                original_frozen_P1_historical_scene_cache_reproduced=False,
                physical_contact_is_not_ground_truth=True,
            ),
        )
