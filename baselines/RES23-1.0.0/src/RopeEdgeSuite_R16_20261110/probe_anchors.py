import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,math
import cv2,numpy as np
from common import *
sys.path.insert(0,str(R12))
from tracker_sg2 import Tracker,angle,line_y
OLD=PARENT/'RopeEdgeSuite_R15_20261109';STRICT=PARENT/'RopeEdgeSuite_Batch_RES14_20261109'
def main():
    cv2.setNumThreads(1);sampler=Tracker(np.zeros((3000,4096),np.uint8))
    cases={'dual_20260923_113745__left':[1,300,370,397,400,500],
        'dual_20260924_111500__left':[1,150,200,400,500],
        'dual_20260923_113029__left':[1,200,400,500],
        'dual_20260923_101643__right':[1,350,400,450]}
    allrows=[]
    for id,ids in cases.items():
        item=next(x for x in read(ROOT/'plan.json')['videos'] if x['id']==id);cap=cv2.VideoCapture(item['source'])
        for i in ids:
            cap.set(1,i);ok,raw=cap.read();assert ok
            im=cv2.flip(raw,1) if item['camera']=='right' else raw
            r=record(OLD/'runs'/id,i);s=record(STRICT/'runs'/id,i)
            assert pixelsha(im)==r['analysis_pixel_sha256']
            row=dict(id=id,frame=i,strict_valid=s['track']['valid'],relaxed_valid=r['track']['valid'],checks=[])
            tr=r['track'];abc=tr['line_abc']
            if abc:
                x=tr['diagnostics']['spatial_anchor']['x_original']
                seeds=[('fixed_candidate',[[x,line_y(abc,x)],[x+220,line_y(abc,x+220)]])]
                original=record(STRICT/'runs'/id,10)['track']['diagnostics'].get('spatial_anchor',{}).get('x_original')
                if original is not None:
                    seeds.append(('strict_fixed_360',[[original,line_y(abc,original)],[original+360,line_y(abc,original+360)]]))
                seeds.append(('own_fixed_360',[[x,line_y(abc,x)],[x+360,line_y(abc,x+360)]]))
                for a in s['track']['diagnostics'].get('attempts',[]):
                    if a.get('seed_valid'):
                        seeds.append(('strict_current_seed',a['seed_xy']));break
                for name,q in seeds:
                    wit=sampler._seed_witness(im,q);m=dict(source=name,x=q[0][0],valid=wit['valid'])
                    if wit['valid']:
                        p=np.asarray(wit['points_xy']);line=cv2.fitLine(p.astype('float32'),cv2.DIST_HUBER,0,.01,.01).ravel()
                        a2=np.array([-line[1],line[0],line[1]*line[2]-line[0]*line[3]])
                        m.update(width=wit['width_px'],support=wit['support_count'],span=float(np.ptp(p[:,0])),
                            candidate_residual=float(np.median(abs(p@np.array(abc)[:2]+abc[2]))),
                            angle_delta=abs(math.degrees(angle(a2)-angle(abc))),points=p.tolist())
                    row['checks'].append(m)
            allrows.append(row);print({k:v for k,v in row.items() if k!='checks'},[{k:v for k,v in m.items() if k!='points'} for m in row['checks']],flush=True)
        cap.release()
    dump(ROOT/'anchor_probe.json',allrows)
if __name__=='__main__':main()
