"""Short current S1 profile blocks with consensus, no filled measurement gaps."""
import sys,cv2,numpy as np
from common20 import PARENT,plain,PITCH_PRIOR
sys.path.append(str(PARENT/'RopeEdgeSuite_R12_20261109'))
from tracker_sg2 import Tracker,line_y

class CoilAxis:
    def __init__(self):self.sampler=Tracker(np.zeros((3000,4096),np.uint8))
    def step(self,image,track,d,pitch=PITCH_PRIOR):
        out=dict(valid=False,reasons=[],points_xy=[],line_abc=None,source='current_local_S1_short_blocks_consensus',history_as_measurement=False)
        if not track.get('valid'):out['reasons'].append('upstream_identity_unavailable');return out
        abc=np.asarray(track['line_abc'],float);wit=track.get('diagnostics',{}).get('seed_validation') or {};width=float(wit.get('width_px') or pitch)
        proposals=[]
        for j,x0 in enumerate(np.arange(d-1000,d-120,120)):
            x0=max(80.,x0);x1=x0+160
            for shift in [0,-12,12,-24,24,-40,40]:
                w=self.sampler._seed_witness(image,np.array([[x0,line_y(abc,x0)+shift],[x1,line_y(abc,x1)+shift]]))
                if not w.get('valid'):continue
                p=np.asarray(w['points_xy']);v=cv2.fitLine(p.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel();q=np.array([-v[1],v[0],v[1]*v[2]-v[0]*v[3]])
                err=float(np.percentile(abs(p@q[:2]+q[2]),90));da=abs(np.degrees(np.arctan2(-q[0],q[1])-np.arctan2(-abc[0],abc[1])))%180;da=min(da,180-da)
                if len(p)<6 or np.ptp(p[:,0])<80 or err>2.5 or da>4 or not .3<=w['width_px']/width<=1.8:continue
                if any(z['block']==j and abs(np.median(z['points_xy'][:,1])-np.median(p[:,1]))<3 for z in proposals):continue
                proposals.append(dict(block=j,points_xy=p,line_abc=q,width=w['width_px'],score=w['score'],residual=err))
        if len(proposals)<3:out['reasons'].append('insufficient_current_profile_blocks');out['candidate_blocks']=len(proposals);return out
        # Candidate lines come from measured blocks, then are evaluated against
        # every other block. At most one candidate per spatial block contributes.
        best=None
        for a in proposals:
            for b in proposals:
                if b['block']-a['block']<2:continue
                pa=np.median(a['points_xy'],axis=0);pb=np.median(b['points_xy'],axis=0);v=pb-pa;v=v/np.linalg.norm(v);q=np.array([-v[1],v[0],v[1]*pa[0]-v[0]*pa[1]])
                da=abs(np.degrees(np.arctan2(-q[0],q[1])-np.arctan2(-abc[0],abc[1])))%180;da=min(da,180-da)
                if da>2.5:continue
                chosen={}
                for c in proposals:
                    err=float(np.median(abs(c['points_xy']@q[:2]+q[2])))
                    if err>3.5:continue
                    score=float(c['score'])-.1*err
                    if c['block'] not in chosen or score>chosen[c['block']][0]:chosen[c['block']]=(score,c)
                if len(chosen)<3:continue
                pts=np.vstack([c['points_xy'] for _,c in chosen.values()]);span=np.ptp(pts[:,0]);offset=abs(line_y(q,d-800)-line_y(abc,d-800))
                rank=len(chosen)+.001*span-.01*offset
                if best is None or rank>best[0]:best=(rank,pts,[c for _,c in chosen.values()])
        if best is None:out['reasons'].append('no_local_consensus');return out
        _,p,blocks=best;v=cv2.fitLine(p.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel();q=np.array([-v[1],v[0],v[1]*v[2]-v[0]*v[3]])
        err=float(np.percentile(abs(p@q[:2]+q[2]),90));span=float(np.ptp(p[:,0]));ext=max(0.,d-p[:,0].max());unc=max(2.,err)*(1+ext/max(300.,span))
        if span<300 or err>3.5 or ext>500 or unc>.3*pitch:out['reasons'].append('consensus_prediction_uncertain')
        y=line_y(q,d)
        out.update(valid=not out['reasons'],points_xy=p,line_abc=q,y_at_d=y,width_px=float(np.median([c['width'] for c in blocks])),
            residual_q90_px=err,extrapolation_px=ext,sensitivity_px=unc,supported_segments_xy=[c['points_xy'] for c in blocks],
            differs_from_upstream_y_px=y-line_y(abc,d),accepted_block_count=len(blocks),candidate_block_count=len(proposals),physical_contact_confirmed=False)
        return plain(out)
