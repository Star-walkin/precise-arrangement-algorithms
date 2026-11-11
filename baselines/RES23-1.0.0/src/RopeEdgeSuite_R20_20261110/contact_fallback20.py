"""Current local tangency can flag proximity without claiming a locked d.

No fixed-column distance is supplied by this branch. It is independent current
optical contact evidence only, with the same visibility/uncertainty gates.
"""
from count20 import measure
from temporal_contact20 import contact_evidence

def evaluate(groups,data,boundary,geometry):
    axis=geometry.get('axis');tc=geometry.get('tangent') or {}
    result=dict(state='unavailable',physical_contact_confirmed=False,physical_contact_ground_truth_available=False,uses_prediction=False,
        source='current_local_tangency_without_locked_d',reasons=[])
    if not axis or not axis['current_coil_side_measured'] or not tc.get('valid'):
        result['reasons']=['current_coil_axis_or_tangency_unavailable'];return result
    if groups is None or data is None:result['reasons']=['current_F3_unavailable'];return result
    q=measure(groups,data,boundary,axis,dict(status='current_local_tangency_contact_only',d=tc['d'],interval=tc['interval']))
    result.update(contact_evidence(q,axis));result.update(source='current_local_tangency_without_locked_d',fixed_d_calibrated=False,
        diagnostic_column=tc['d'],diagnostic_column_interval=tc['interval'],local_measurement_available=q['available'],
        current_local_N=q.get('N'),current_local_N_interval=q.get('N_interval'),eligible_for_locked_column_count=False)
    return result
