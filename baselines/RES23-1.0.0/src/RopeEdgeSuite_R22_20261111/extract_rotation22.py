import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,concurrent.futures as cf,cv2
from common22 import *

def rois_from_boundary(r,small):
    b=r['boundary'];n=b['normalized_x'];coef=b['fit_coefficients_ascending']
    rois=[];hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    # Image-normalized strips above the fitted coil rim. Four distinct sectors.
    for frac in [.28,.37,.46,.55,.64]:
        cx=4096*frac
        by=float(np.polynomial.polynomial.polyval((cx-n['center'])/n['scale'],coef))
        # Wide upper band covers the visible yellow annulus; fixed per video.
        x0=round(cx/4-65);x1=x0+130;y1=round(by/4-45);y0=y1-40
        x0=max(0,x0);x1=min(1024,x1);y0=max(0,y0);y1=min(750,y1)
        patch=hsv[y0:y1,x0:x1];mask=((patch[:,:,0]>=12)&(patch[:,:,0]<=40)&(patch[:,:,1]>55)&(patch[:,:,2]>40)).astype('uint8')
        if mask.mean()<.55:continue
        mask=cv2.erode(mask,np.ones((3,3),np.uint8))
        rois.append(dict(box=[x0,y0,x1,y1],mask=mask))
    return rois

def process(item):
    cv2.setNumThreads(1);id=item['id'];out=ROOT/'rotation'/id;out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists() and read(out/'receipt.json').get('regions'):return read(out/'receipt.json')
    rows=read(R21/'delivery'/id/'frames.json');cap=cv2.VideoCapture(item['source']);features=[];rois=None;verified=0
    for i,row in enumerate(rows):
        ok,raw=cap.read();assert ok
        assert pixelsha(raw)==row['source_pixel_sha256'];verified+=1
        im=cv2.flip(raw,1) if item['camera']=='right' else raw
        small=cv2.resize(im,(1024,750),interpolation=cv2.INTER_AREA)
        if not rois:
            fr=frontend(id,i)
            if fr['boundary'].get('fit_valid'):rois=rois_from_boundary(fr,small)
            if not rois:
                features.append(None)
                continue
            preview=small.copy()
            for j,r in enumerate(rois):
                x0,y0,x1,y1=r['box'];cv2.rectangle(preview,(x0,y0),(x1,y1),(20,30,250),2);cv2.putText(preview,str(j),(x0,y0-3),0,.5,(20,30,250),1)
            cv2.imencode('.jpg',preview)[1].tofile(out/'regions.jpg')
        f=[]
        for r in rois:
            x0,y0,x1,y1=r['box'];g=cv2.cvtColor(small[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
            h=(g-cv2.GaussianBlur(g,(0,0),3))*r['mask']
            f.append(cv2.resize(h,(60,20)).ravel())
        features.append(f)
    cap.release()
    count=len(rois or [])
    features=[f if f is not None else np.zeros((count,1200),np.float32) for f in features]
    dest=out/'features.npz';np.savez_compressed(dest,features=np.array(features,dtype=np.float32))
    receipt=dict(id=id,source=item['source'],source_sha256=item['source_sha256'],decoded_pixel_verified=verified,
        features_path=str(dest),features_sha256=sha(dest),regions=[r['box'] for r in rois or []],extractor_sha256=sha(Path(__file__)))
    dump(out/'receipt.json',receipt);print('texture extracted',id,verified,len(rois),flush=True);return receipt

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);ap.add_argument('--id');a=ap.parse_args()
    jobs=[x for x in plan() if not a.id or a.id in x['id']]
    jobs.sort(key=lambda x:('111500' not in x['id'],x['id']))
    with cf.ProcessPoolExecutor(a.workers) as pool:results=list(pool.map(process,jobs))
    dump(ROOT/'rotation_receipt.json',results)
