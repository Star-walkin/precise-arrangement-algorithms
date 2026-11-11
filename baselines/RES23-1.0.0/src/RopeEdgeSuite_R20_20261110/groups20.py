"""Exact BG3 global yellow-point classes with per-frame float sampling caches."""
import sys,numpy as np
from common20 import BASE
sys.path.insert(0,str(BASE))
import transfer_engine as e

def group(data,boundary,d):
    fields=dict(data)
    fields['model_envelope']=e.model_envelope(data['surface'],np.asarray(boundary['top_curve_xy'])*.5)
    # cv2.remap's old helper already converts each of these to float32. Doing
    # that once avoids hundreds of identical full-mask allocations per frame.
    for k in ('surface','exclusion','broad','clean','body','model_envelope'):fields[k]=fields[k].astype(np.float32,copy=False)
    curves,diag,points=e.group_and_fit(fields,d*.5)
    cross,rejected=e.intersections(curves,points,fields,d*.5)
    result=dict(curves=[dict(c,xy=np.asarray(c['xy'])*2) for c in curves],diagnostics=diag,
        initialization_column=d,coordinates='original_normalized_xy',raw_point_fields_coordinates='half_resolution',
        crossings_at_initialization=[dict(c,y=c['y']*2) for c in cross],rejected_at_initialization=[dict(c,y=c['y']*2) for c in rejected])
    return result
