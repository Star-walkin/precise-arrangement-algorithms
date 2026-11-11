"""P1/H6 experiment with a current SG5-verified seed, original model orientation.

Only the seed provider changes; B750 weights, D4 observations and H6 remain frozen.
SG5 identity is a current optical association, not proof of material identity.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import argparse,time,cv2,copy
from common20 import *
from roi20 import masks
from probe_paths_original import AuditedP1,normalized
from lc1_p1_adapter import _raw_seed_fraction

class SeededP1(AuditedP1):
    def _entry_seeds(self,bgr,a7):
        t=self.current_track
        if not t.get('valid'):return [],[dict(used=False,reason='no_current_SG5_identity')]
        xy=np.array(t['seed_xy_original'],float)
        if self.camera=='right':xy[:,0]=4095-xy[:,0]
        width=float((t.get('diagnostics',{}).get('seed_validation') or {}).get('width_px') or self.pitch)
        gray=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),(2048,1500),interpolation=cv2.INTER_AREA).astype(np.float32)/255
        raw=_raw_seed_fraction(gray,xy,width)
        if raw<.3:return [],[dict(used=False,reason='current_SG5_seed_raw_support_insufficient',fraction=raw)]
        # Track seed ordering is entry toward coil in normalized coordinates;
        # verify direction against current surface, rather than camera naming.
        dist=cv2.distanceTransform((self.surface==0).astype(np.uint8),cv2.DIST_L2,3)
        ds=np.array([dist[int(round(y)),int(round(x))] for x,y in xy])
        if ds[0]<ds[-1]:xy=xy[::-1].copy();ds=ds[::-1]
        seed=dict(xy=xy.tolist(),source='current_SG5_identity_and_raw_rechecked_entry',entry_id='SG5_current',
            a7_candidate_id=None,a7_accepted=False,width_px=width,automatic=True,current_raw_support_fraction=raw,
            direction_ambiguous=bool(ds[0]-ds[-1]<20),outside_distance_endpoints_px=ds.tolist(),
            direction_basis='current_SG5_seed_toward_current_coil_mask',identity_confirmed=False)
        return [seed],[dict(used=True,source=seed['source'],raw_support_fraction=raw)]

    def _support(self,entry,candidate,all_ranked):
        valid,checks,reasons,segments,points=super()._support(entry,candidate,all_ranked)
        # The new provider has no A7 acceptance flag. Replace precisely that
        # provider-specific gate; never remove a H6/image/geometry rejection.
        if entry['seed']['source']=='current_SG5_identity_and_raw_rechecked_entry':
            reasons=[r for r in reasons if r!='A7_axis_not_uniquely_accepted']
            checks['entry_provider']='current_SG5_with_current_raw_seed_recheck'
        return not reasons,checks,reasons,segments,points

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--frames',required=True);a=ap.parse_args()
    item=next(x for x in plan() if x['id']==a.id);out=ROOT/'paths_sg5'/a.id;out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(item['source']);p1=None
    for i in map(int,a.frames.split(',')):
        target=out/f'p1_{i:06d}.json'
        if target.exists():continue
        cap.set(1,i);ok,raw=cap.read();assert ok;r=record(a.id,i);assert pixelsha(raw)==r['source_pixel_sha256']
        im=cv2.flip(raw,1) if item['camera']=='right' else raw
        surface,_,meta=masks(im,r);surface_original=cv2.flip(surface,1) if item['camera']=='right' else surface
        if p1 is None:p1=SeededP1(surface_original,PITCH_PRIOR,LC1/'supplied_assets/step_0750.pt',threads=2,camera=item['camera'],surface_mode='entry_band')
        else:p1.surface=surface_original
        p1.current_track=r['track'];p1.candidates=[];tick=time.time();result=p1.step(raw);result['candidate_paths_audit']=p1.candidates
        result['source']='real_P1_B750_H6_with_current_SG5_seed_D4_scene_adapter'
        dump(target,dict(id=a.id,frame=i,time_s=i/item['fps'],source_pixel_sha256=pixelsha(raw),analysis_pixel_sha256=pixelsha(im),
            input_record_sha256=sha(selected_run(a.id)/'frames'/f'{i:06d}.json.gz'),model_input_orientation='original_unmirrored',
            raw_result=result,result=normalized(result,item['camera']=='right'),elapsed_s=time.time()-tick,code_sha256=sha(__file__),roi=meta))
        print(a.id,i,result['status'],round(time.time()-tick,1),flush=True)
    cap.release()
if __name__=='__main__':main()
