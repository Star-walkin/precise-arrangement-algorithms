"""Saturated interior does not imply unavailable outer edges or usable means."""
from pathlib import Path
import json,hashlib
import numpy as np
from photometric_ribbon import build_fields,sample_ribbon,compare_ribbon_neighbors

def measure(case):
 yy,xx=np.mgrid[:220,:420];g=np.full(yy.shape,110,np.uint8)
 if case=='white_ribbon':g[abs(yy-110)<=12]=255
 elif case=='all_white':g[:]=255
 elif case=='one_white_edge':g[yy>=110]=255
 elif case=='thin_white_strand':g[abs(yy-110)<=1]=255
 elif case=='background_nearly_white':g[:]=250;g[abs(yy-110)<=12]=255
 elif case=='black_ribbon':g[abs(yy-110)<=12]=0
 else:raise ValueError(case)
 image=np.repeat(g[:,:,None],3,axis=2);fields=build_fields(image,24,24)
 r=sample_ribbon(fields,[[210.,110.]],[1.,0.],24)
 c=compare_ribbon_neighbors(fields,[[210.,110.]],[1.,0.],24)
 return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()},bool(c['comparison_reliable'][0]),float(c['comparison_score'][0])

def main():
 cases={}
 for case in ('white_ribbon','all_white','one_white_edge','thin_white_strand','background_nearly_white','black_ribbon'):
  r,comparison,adv=measure(case);outline=bool(r['saturated_outline_supported'][0]);print(case,outline,r['reliable'],r['score'],r['saturated_outline_score'],flush=True)
  assert outline==(case=='white_ribbon'),(case,r)
  if case=='white_ribbon':
   assert not r['reliable'][0] and not r['supported'][0] and r['score'][0]==0.
   assert not comparison and adv==0.,'clipped mean must not acquire differential advantage'
   assert r['paired_edge_contrast_raw'][0]>100 and r['saturated_outline_score'][0]>.9
  cases[case]={'measurement':r,'comparison_reliable':comparison,'comparison_score':adv}
 root=Path(__file__).resolve().parent;out=root/'saturated_outline_synthetic_001.json'
 if out.exists():raise ValueError('Preserve old mechanism receipt')
 report={'passed':True,'scope':'synthetic exterior-outline support only; clipped body means remain unavailable; not identity accuracy','cases':cases,'source_sha256':hashlib.sha256((root/'photometric_ribbon.py').read_bytes()).hexdigest()}
 out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'passed':True,'source_sha256':report['source_sha256']}))

if __name__=='__main__':main()
