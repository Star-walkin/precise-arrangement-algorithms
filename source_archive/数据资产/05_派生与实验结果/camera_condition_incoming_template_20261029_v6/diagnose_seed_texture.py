"""Posthoc original-pixel texture diagnosis, fixed saved four-case seed pool."""
import json,sys
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'camera_condition_incoming_photometric_20261029_v5'))
import whole_io as io
def main():
    out=[]
    for p in sorted((HERE/'seed_probe_003').glob('*.json')):
        saved=io.read_json(p)
        if 'input' not in saved:continue
        image=io.read_image(io.DATA/saved['input']);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32);case={'input':saved['input'],'candidates':[]}
        for c in saved['generated']['candidates']:
            seed=c['seed'];seg=np.array(seed['observed_segment_xy']);d=np.array(seed['direction_xy']);n=np.array([-d[1],d[0]]);width=seed['width_px'];length=np.linalg.norm(seg[1]-seg[0]);s=np.arange(0,length,1.);ns=np.arange(-.32*width,.32*width+.1,.5);q=seg[0]+s[:,None,None]*d+ns[None,:,None]*n
            patch=cv2.remap(gray,q[...,0].astype(np.float32),q[...,1].astype(np.float32),cv2.INTER_LINEAR);gx=cv2.Sobel(patch,cv2.CV_32F,1,0)/4;gy=cv2.Sobel(patch,cv2.CV_32F,0,1)/8
            body=patch[:,2:-2] if patch.shape[1]>6 else patch;xx=gx[:,2:-2] if patch.shape[1]>6 else gx;yy=gy[:,2:-2] if patch.shape[1]>6 else gy
            jnn=float(np.mean(xx**2));jtt=float(np.mean(yy**2));jnt=float(np.mean(xx*yy));coherence=float(np.hypot(jnn-jtt,2*jnt)/max(jnn+jtt,1e-6));mod=np.std(body,axis=0);rec={'id':c['candidate_id'],'width':width,'body_mean':float(np.mean(body)),'tangent_energy':jtt,'normal_energy':jnn,'tangent_fraction':jtt/max(jtt+jnn,1e-6),'tensor_coherence':coherence,'longitudinal_modulation':float(np.mean(mod)),'saturated':float(np.mean(body>248)),'span':float(length)};case['candidates'].append(rec);print(Path(saved['input']).stem,rec)
        out.append(case)
    path=HERE/'seed_texture_diagnosis_001.json'
    if path.exists():raise ValueError('no overwrite')
    io.dump(path,{'scope':'Posthoc fixed4 raw texture descriptors; no prediction changes','cases':out,'source_sha256':io.sha(__file__)})
if __name__=='__main__':main()
