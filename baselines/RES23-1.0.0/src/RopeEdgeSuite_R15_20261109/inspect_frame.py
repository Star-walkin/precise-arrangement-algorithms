import argparse
import cv2
from PIL import Image
from common import *
from render_batch import panel

def inspect(id,i):
    item=next(x for x in read(ROOT/'plan.json')['videos'] if x['id']==id)
    run=ROOT/'runs'/id;r=record(run,i);cap=cv2.VideoCapture(item['source'])
    if item.get('partial_source'):
        for k in range(i+1):ok,im=cap.read();assert ok
    else:cap.set(1,i);ok,im=cap.read()
    cap.release()
    assert ok and pixelsha(im)==r['source_pixel_sha256']
    out=ROOT/'review';out.mkdir(exist_ok=True);path=out/f'{id}__{i:06d}.jpg';Image.fromarray(panel(im,r,item)).save(path,quality=94)
    from collections import Counter
    b=r['boundary'];s=r['track'];reasons=Counter(q for a in s['diagnostics']['attempts'] for q in a.get('reasons',[]))
    d=dict(id=id,frame=i,image=str(path),tracking_valid=s['valid'],tracking_reason=s['diagnostics']['accept_reason'],
        attempts_rejected=dict(reasons),proposal_reasons=r['proposal']['reasons'],
        boundary={k:v for k,v in b.items() if k not in ['top_curve_xy','observed_segments','foreground_polygons','retained_anchors_xy','excluded_anchors_xy']})
    dump(out/f'{id}__{i:06d}.json',d);print(json.dumps({k:d[k] for k in ['id','frame','image','tracking_valid','attempts_rejected','proposal_reasons']},ensure_ascii=False));print('BG7:',b['reasons'])
    return path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('id');p.add_argument('frame',type=int);a=p.parse_args();inspect(a.id,a.frame)
