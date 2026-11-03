"""Constructed original-groove tests, independent of real project labels."""
from pathlib import Path
import hashlib,json
import numpy as np
from photometric_ribbon import build_fields,sample_ribbon
from attached_groove import sample_attached_grooves,PARAMETERS

def make(case):
 yy,xx=np.mgrid[:220,:440].astype(float);g=np.full(yy.shape,160.)
 if case in ('two_grooves','one_groove','dark_grooves','thin_strands'):
  depth=6. if case=='dark_grooves' else 65.;sigma=.35 if case=='thin_strands' else 2.4
  if case=='dark_grooves':g[:]=12.
  for center in ([88.] if case=='one_groove' else [88.,112.]):g-=depth*np.exp(-.5*((yy-center)/sigma)**2)
 elif case=='linear_illumination':g=80.+.4*yy
 elif case=='saturated':g[:]=255.
 elif case=='black':g[:]=0.
 elif case!='flat':raise ValueError(case)
 return np.repeat(np.clip(g,0,255).astype(np.uint8)[:,:,None],3,axis=2)

def main():
 root=Path(__file__).resolve().parent;results={};b=np.array([[[220.,88.],[220.,112.]]])
 for case in ('two_grooves','one_groove','dark_grooves','thin_strands','linear_illumination','saturated','black','flat'):
  image=make(case);before=image.copy();fields=build_fields(image,24,24)
  r=sample_attached_grooves(fields,b,[1.,0.],24.)
  assert np.array_equal(image,before)
  item={k:v.tolist() for k,v in r.items() if isinstance(v,np.ndarray)};results[case]=item
  print(case,r['score'],r['supported'],r['groove_width_ratio'],flush=True)
 for c in ('two_grooves','dark_grooves'):assert results[c]['supported']==[True],results[c]
 for c in ('one_groove','thin_strands','linear_illumination','saturated','black','flat'):assert results[c]['supported']==[False],(c,results[c])
 assert sum(results['one_groove']['edge_support'][0])==1
 fields=build_fields(make('two_grooves'),24.,24.)
 old=sample_ribbon(fields,[[220.,100.]],[1.,0.],24.)
 assert not old['supported'][0], 'uniform adjacent rope bodies should not satisfy suspended-rope mean contrast'
 wrong=sample_attached_grooves(fields,b+np.array([0.,10.8]),[1.,0.],24.)
 assert not wrong['supported'][0]
 reverse=sample_attached_grooves(fields,b,[-1.,0.],24.)
 assert np.allclose(reverse['score'],results['two_grooves']['score'],atol=.03)
 many=np.repeat(b,288,axis=0);many[:,:,0]=np.linspace(80,360,288)[:,None]
 batch=sample_attached_grooves(fields,many,[1.,0.],np.full(288,24.))
 assert np.all(batch['supported']) and batch['edge_scores'].shape==(288,2)
 report={'passed':True,'scope':'synthetic valley-vs-step, groove-width, low-signal and vectorization mechanisms only; not accuracy','cases':results,'suspended_step_rejected_same_attached_pattern':not bool(old['supported'][0]),'wrong_boundary_rejected':True,'input_unchanged':True,'parameters':PARAMETERS,'source_sha256':hashlib.sha256((root/'attached_groove.py').read_bytes()).hexdigest()}
 out=root/'attached_groove_synthetic_001.json'
 if out.exists():raise ValueError('Preserve previous test receipt')
 out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'passed':True,'source_sha256':report['source_sha256']}))

if __name__=='__main__':main()
