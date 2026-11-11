"""CE4: coil-side identity needs current overlap or exposed-rope witnesses."""
import numpy as np
from coil_edge_axis import measure as overlap_fit
from coil_edge_axis_v3 import measure as exposed_fit
from common20 import PITCH_PRIOR

def measure(image,track,d,pitch=PITCH_PRIOR,surface=None):
    external=exposed_fit(image,track,d,pitch,surface=surface)
    pts=np.asarray(track.get('line_points_xy',[]),float).reshape(-1,2)
    overlap=pts[(pts[:,0]>=d-1100)&(pts[:,0]<=d-40)] if len(pts) else pts
    local=None
    if len(overlap)>=25 and np.ptp(overlap[:,0])>=200:local=overlap_fit(image,track,d,pitch)
    selected=external if external['valid'] else local if local and local['valid'] else external
    selected=dict(selected);selected['source']='CE4_current_coil_side_axis';selected['identity_basis']='current_exposed_bright_rope_witness_or_SG5_overlap'
    selected['external_trial']={k:external.get(k) for k in ('valid','reasons','y_at_d','external_current_bright_rope_witnesses','external_witness_span_px')}
    selected['overlap_trial']=None if local is None else {k:local.get(k) for k in ('valid','reasons','y_at_d','overlap_support_q90_px')}
    if selected['valid'] and local and local['valid'] and external['valid']:
        difference=abs(local['y_at_d']-external['y_at_d']);selected['current_fit_disagreement_px']=difference
        # Preserve current-model disagreement in the interval, not as hidden
        # smoothing or an unearned gain in precision.
        selected['sensitivity_px']=max(selected['sensitivity_px'],difference)
        if difference>.75*pitch:selected.update(valid=False,reasons=selected['reasons']+['current_local_axis_models_disagree'])
    return selected
