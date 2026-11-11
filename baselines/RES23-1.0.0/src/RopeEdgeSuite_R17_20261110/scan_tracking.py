import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,cv2,time
from collections import Counter
from common import *
from tracker_sg5 import Fusion
OLD=PARENT/'RopeEdgeSuite_R15_20261109';STRICT=PARENT/'RopeEdgeSuite_Batch_RES14_20261109'
def main(id):
    cv2.setNumThreads(1);item=next(x for x in read(ROOT/'plan.json')['videos'] if x['id']==id)
    cap=cv2.VideoCapture(item['source']);f=Fusion();rows=[];tick=time.time()
    for i in range(item['planned_frames']):
        path=OLD/'runs'/id/'frames'/f'{i:06d}.json.gz'
        while not path.exists():time.sleep(.5)
        ok,raw=cap.read();assert ok;im=cv2.flip(raw,1) if item['camera']=='right' else raw
        r=record(OLD/'runs'/id,i);s=record(STRICT/'runs'/id,i)
        assert pixelsha(im)==r['analysis_pixel_sha256']==s['analysis_pixel_sha256']
        o=f.step(im,s['track'],r['track'],i/item['fps'],i);g=o['diagnostics']['SG5']
        row=dict(frame=i,valid=o['valid'],branch=g['branch'],strict_reasons=(g.get('strict_check') or {}).get('reasons',[]),
            fallback_reasons=(g.get('fallback_check') or {}).get('reasons',[]),
            original_strict_valid=s['track']['valid'],original_relaxed_valid=r['track']['valid'])
        rows.append(row)
    cap.release();out=dict(id=id,frames=len(rows),valid=sum(r['valid'] for r in rows),branches=Counter(r['branch'] for r in rows),
        rejected_strict=Counter(x for r in rows for x in r['strict_reasons']),rejected_fallback=Counter(x for r in rows for x in r['fallback_reasons']),
        rows=rows,elapsed_s=time.time()-tick,pixel_verified=True,source_unchanged=sha(item['source'])==item['source_sha256'])
    dump(ROOT/'tracking_scan'/f'{id}.json',out);print({k:v for k,v in out.items() if k!='rows'},flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);main(p.parse_args().id)
