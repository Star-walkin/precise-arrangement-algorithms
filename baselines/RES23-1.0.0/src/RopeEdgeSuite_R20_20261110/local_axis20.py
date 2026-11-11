"""Independent current S1 bilateral measurements on the coil side of the guide."""
import sys,cv2,numpy as np
from common20 import PARENT,plain,PITCH_PRIOR
sys.path.append(str(PARENT/'RopeEdgeSuite_R12_20261109'))
from tracker_sg2 import Tracker,line_y

class CoilAxis:
    def __init__(self):self.sampler=Tracker(np.zeros((3000,4096),np.uint8))
    def step(self,image,track,d,pitch=PITCH_PRIOR):
        out=dict(valid=False,reasons=[],points_xy=[],line_abc=None,source='current_local_S1_bilateral_profiles',history_as_measurement=False)
        if not track.get('valid'):out['reasons'].append('upstream_identity_unavailable');return out
        abc=np.asarray(track['line_abc'],float);diag=track.get('diagnostics',{});wit=diag.get('seed_validation') or {}
        width=float(wit.get('width_px') or pitch);log=[];accepted=[]
        for x0 in [d-1000,d-760,d-500]:
            x0=max(80.,x0);x1=x0+360
            candidates=[]
            for shift in [0,-12,12,-24,24,-40,40,-60,60]:
                seed=np.array([[x0,line_y(abc,x0)+shift],[x1,line_y(abc,x1)+shift]])
                w=self.sampler._seed_witness(image,seed)
                if not w.get('valid'):continue
                p=np.asarray(w['points_xy']);v=cv2.fitLine(p.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel();q=np.array([-v[1],v[0],v[1]*v[2]-v[0]*v[3]])
                residual=float(np.percentile(abs(p@q[:2]+q[2]),90));span=float(np.ptp(p[:,0]));ratio=w['width_px']/width
                da=abs(np.degrees(np.arctan2(-q[0],q[1])-np.arctan2(-abc[0],abc[1])))
                if da>90:da=180-da
                if span<220 or len(p)<7 or residual>2.5 or da>2.5 or not .55<=ratio<=1.7:continue
                # Current points, not the search offsets, determine output geometry.
                offset=float(np.median(abs(p@abc[:2]+abc[2])))
                rank=float(w['score'])-offset*.002-residual*.02
                candidates.append(dict(points_xy=p,line_abc=q,width=w['width_px'],residual=residual,score=rank,shift=shift,offset=offset))
            candidates.sort(key=lambda q:-q['score'])
            if candidates:accepted.append(candidates[0]);log.append(dict(x0=x0,candidates=len(candidates),selected=candidates[0]))
            else:log.append(dict(x0=x0,candidates=0))
        out['windows']=plain(log)
        if not accepted:out['reasons'].append('no_coil_side_bilateral_segment');return out
        if len(accepted)>1:
            predictions=[line_y(a['line_abc'],d-300) for a in accepted]
            if np.ptp(predictions)>.35*pitch:out['reasons'].append('local_segments_disagree');return out
        pts=np.vstack([q['points_xy'] for q in accepted]);v=cv2.fitLine(pts.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel()
        q=np.array([-v[1],v[0],v[1]*v[2]-v[0]*v[3]]);res=float(np.percentile(abs(pts@q[:2]+q[2]),90))
        if res>3.:out['reasons'].append('merged_local_residual_excessive');return out
        y=line_y(q,d);span=float(np.ptp(pts[:,0]));ext=max(0.,d-pts[:,0].max())
        uncertainty=max(2.5,res)*(1+ext/max(220.,span))
        if ext>700 or uncertainty>.25*pitch or (ext>400 and len(accepted)<2):out['reasons'].append('only_remote_segment_available');return out
        out.update(valid=True,points_xy=pts,line_abc=q,y_at_d=y,width_px=float(np.median([p['width'] for p in accepted])),
            residual_q90_px=res,extrapolation_px=ext,sensitivity_px=uncertainty,
            differs_from_upstream_y_px=y-line_y(abc,d),physical_contact_confirmed=False)
        return plain(out)
