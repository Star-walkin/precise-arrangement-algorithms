"""Prefer current SG5 coil-side samples; independently search only if absent."""
import cv2,numpy as np
from common20 import plain,PITCH_PRIOR
from local_axis21 import CoilAxis as BlockSearch,line_y

class CoilAxis(BlockSearch):
    def step(self,image,track,d,pitch=PITCH_PRIOR):
        if track.get('valid'):
            p=np.asarray(track.get('line_points_xy',[]),float).reshape(-1,2)
            p=p[(p[:,0]>=d-1000)&(p[:,0]<=d-60)]
            if len(p)>=20 and np.ptp(p[:,0])>=300 and d-p[:,0].max()<=350:
                v=cv2.fitLine(p.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel();q=np.array([-v[1],v[0],v[1]*v[2]-v[0]*v[3]])
                residual=abs(p@q[:2]+q[2]);err=float(np.percentile(residual,90));ext=max(0.,d-p[:,0].max());unc=max(2.,err)*(1+ext/np.ptp(p[:,0]))
                delta=line_y(q,d)-line_y(track['line_abc'],d)
                if err<=3. and unc<=.25*pitch and abs(delta)<=.5*pitch:
                    return plain(dict(valid=True,reasons=[],points_xy=p,line_abc=q,y_at_d=line_y(q,d),
                        source='current_SG5_supported_coil_side_points_robust_refit',history_as_measurement=False,
                        width_px=float((track.get('diagnostics',{}).get('seed_validation') or {}).get('width_px') or pitch),
                        residual_q90_px=err,extrapolation_px=ext,sensitivity_px=unc,differs_from_upstream_y_px=delta,
                        supported_segments_xy=[p],physical_contact_confirmed=False))
        out=super().step(image,track,d,pitch)
        out['local_search_needed_because']='no_sufficient_current_SG5_coil_side_support'
        return out
