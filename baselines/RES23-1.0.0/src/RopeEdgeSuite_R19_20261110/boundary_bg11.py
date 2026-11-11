"""BG11: bounded startup exception for an already anchored, well-observed arc.

The trigger is expressed entirely in the unmodified BG10 output. If it is
false, BG10 state and numeric outputs are returned unchanged. This supports an
exact trace-invariance audit for videos that never invoke the startup repair.
"""
import copy,numpy as np
from common import PITCH
from boundary_bg10 import Boundary as BG10

def startup_repair_eligible(out):
    s=out.get('shape_observability') or {}
    return bool(out.get('reasons')==['initial_shape_evidence_accumulating'] and
        1<=len(out.get('shape_prior_frames',[]))<3 and not s.get('accepted',False) and
        .35<=s.get('excluded_fraction',1.)<=.5 and out.get('current_clean_points',0)>=120 and
        s.get('clean_span_fraction',0)>=.9 and s.get('bins',0)>=7 and
        s.get('min_curvature',0)>10 and s.get('residual_q90',1e9)<.25*PITCH and
        s.get('heldout_block_sensitivity_px',1e9)<.35*PITCH and s.get('shape_difference_px',1e9)<.3*PITCH)

class Boundary(BG10):
    def step(self,image,proposal,track,i,*,forced_exclude=None):
        before=copy.deepcopy(self.__dict__) if len(self.shapes)<3 else None
        out=super().step(image,proposal,track,i,forced_exclude=forced_exclude)
        if not startup_repair_eligible(out):return out
        rejected={tuple(p) for p in out['excluded_candidates_xy']}
        pts=[p for p in out['raw_candidates_xy'] if tuple(p) not in rejected]
        clean=dict(proposal,top_observed_xy=pts,rejected_top_observations_xy=[])
        trial=BG10();trial.__dict__.update(before)
        candidate=trial.step(image,clean,track,i,forced_exclude=forced_exclude)
        if not candidate.get('shape_observability',{}).get('accepted',False):return out
        self.__dict__.update(trial.__dict__)
        candidate['version']='BG11'
        candidate['startup_visible_arc_repair']=dict(used=True,current_frame=i,original_metrics=out['shape_observability'],
            original_candidates_xy=out['raw_candidates_xy'],excluded_candidates_xy=out['excluded_candidates_xy'],
            all_points_remeasured_current_frame=True,prior_is_shape_only=True)
        if not candidate.get('d_locally_observed',False):candidate['count_boundary_valid']=False
        return candidate
