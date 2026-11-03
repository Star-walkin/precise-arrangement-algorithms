"""Mechanism-only image constructions; no project images, labels or devices."""
from pathlib import Path
import hashlib,json,time,argparse
import numpy as np
import cv2
from photometric_ribbon import build_fields,sample_ribbon,PARAMETERS

HERE=Path(__file__).resolve().parent
def image(case):
 yy,xx=np.mgrid[:240,:420].astype(np.float32);s=yy-120.;w=24.
 g=np.full(yy.shape,110.,np.float32)
 if case in ('bright','bright_strands','gradient_bright'):g[abs(s)<=w/2]=175.
 elif case=='dark':g[abs(s)<=w/2]=45.
 elif case=='narrow_strand':g[abs(s)<=2.]=25.
 elif case=='linear_gradient':g=70.+.45*yy
 elif case=='periodic_coils':g=125.+42*np.cos(2*np.pi*s/w)
 elif case=='saturated':g[:]=255.
 elif case=='black':g[:]=1.
 elif case=='low_contrast_noise':g=2.+.6*np.sin(xx*.6+yy*.9)
 elif case=='single_edge':g[s>=0]=175.
 elif case=='shadow_only':g[abs(s+14)<2]=25.
 elif case!='constant':raise ValueError(case)
 if case=='bright_strands':g+=(abs(s)<=w/2)*18.*np.sin(xx*1.0+s*.65)
 if case=='gradient_bright':g+=.15*(yy-120)
 return np.repeat(np.clip(g,0,255).astype(np.uint8)[:,:,None],3,axis=2)

def test(case,point=(210.,120.),tangent=(1.,0.),width=24.):
 img=image(case);before=img.copy();fields=build_fields(img,width,24.)
 t=time.time();r=sample_ribbon(fields,np.array([point]),np.array([tangent]),width)
 assert np.array_equal(img,before)
 assert fields['measurement_source']=='original_image_only' and r['identity_verified'] is False
 return {k:(float(v[0]) if np.asarray(v).dtype.kind not in ('b','u','i') else bool(v[0]) if np.asarray(v).dtype.kind=='b' else int(v[0])) for k,v in r.items() if isinstance(v,np.ndarray) and v.ndim==1},time.time()-t

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--receipt',default='photometric_ribbon_synthetic_001.json');args=parser.parse_args()
 results={}
 for case in ('bright','dark','bright_strands','gradient_bright','narrow_strand','linear_gradient','periodic_coils','constant','saturated','black','low_contrast_noise','single_edge','shadow_only'):
  r,seconds=test(case);results[case]={**r,'seconds':seconds};print(case,r['score'],r['supported'],r['unknown_reason_bits'],flush=True)
 for case in ('bright','dark','bright_strands','gradient_bright'):assert results[case]['supported'],(case,results[case])
 for case in ('narrow_strand','linear_gradient','periodic_coils','constant','saturated','black','low_contrast_noise','single_edge','shadow_only'):assert not results[case]['supported'],(case,results[case])
 assert results['narrow_strand']['score']<results['dark']['score']*.6
 assert results['periodic_coils']['score']<results['bright']['score']*.6
 assert abs(results['gradient_bright']['score']-results['bright']['score'])<.08
 reverse,_=test('bright',tangent=(-1.,0.));assert abs(reverse['score']-results['bright']['score'])<.02
 border,_=test('bright',point=(4.,120.));assert not border['reliable'] and not border['supported'] and border['unknown_reason_bits']&1
 fields=build_fields(image('bright'),24.,24.)
 xy=np.c_[np.linspace(80,340,288),np.full(288,120.)];widths=np.full(288,24.)
 many=sample_ribbon(fields,xy,np.array([1.,0.]),widths)
 assert many['score'].shape==(288,) and np.all(many['supported'])
 results['orientation_reversal']=reverse;results['image_border']=border
 report={'passed':True,'scope':'Constructed original-image photometric mechanisms only; no real-image accuracy or same-material claim','tests':results,'vectorized_288_supported':int(np.count_nonzero(many['supported'])),'input_images_unchanged':True,'parameters':PARAMETERS,'source_sha256':hashlib.sha256((HERE/'photometric_ribbon.py').read_bytes()).hexdigest()}
 out=HERE/args.receipt
 if out.exists():raise ValueError('Do not overwrite previous test receipt')
 out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps({'passed':True,'source_sha256':report['source_sha256']}))

if __name__=='__main__':main()
