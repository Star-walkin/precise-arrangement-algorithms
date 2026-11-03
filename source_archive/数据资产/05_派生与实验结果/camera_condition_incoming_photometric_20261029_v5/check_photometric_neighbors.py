"""Independent same-condition neighboring-band comparison checks."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
from photometric_ribbon import build_fields,sample_ribbon,compare_ribbon_neighbors
from check_photometric_ribbon import image

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--receipt',default='photometric_neighbors_synthetic_001.json');args=parser.parse_args()
 results={}
 for case in ('bright','dark','periodic_coils','narrow_strand'):
  img=image(case);fields=build_fields(img,24.,24.);xy=np.array([[210.,120.]])
  r=compare_ribbon_neighbors(fields,xy,[1.,0.],24.,24.)
  expected=[]
  for dy in (0.,-24.,24.):expected.append(sample_ribbon(fields,xy+np.array([0.,dy]),[1.,0.],24.))
  assert r['score'][0]==expected[0]['score'][0]
  assert r['neighbor_left_score'][0]==expected[1]['score'][0]
  assert r['neighbor_right_score'][0]==expected[2]['score'][0]
  if case in ('bright','dark'):
   assert not expected[1]['supported'][0] and not expected[2]['supported'][0], 'pure neighboring background acquired false paired edges'
   assert r['comparison_score'][0]>.15, 'isolated stripe should differ from its background neighbors'
  results[case]={k:bool(r[k][0]) if r[k].dtype.kind=='b' else float(r[k][0]) for k in ('score','supported','comparison_score','comparison_reliable','neighbor_left_score','neighbor_right_score')}
 assert abs(results['periodic_coils']['comparison_score'])<.06
 assert not results['narrow_strand']['supported']
 # 中心有信号、邻带全饱和的场景：邻带不可用不能虚增中心优势。
 yy,xx=np.mgrid[:240,:420];img=np.full((240,420,3),255,np.uint8);img[108:133,:,:]=60
 r=compare_ribbon_neighbors(build_fields(img,24,24),[[210.,120.]],[1.,0.],24.)
 assert not r['comparison_reliable'][0] and r['comparison_score'][0]==0.
 results['unavailable_neighbor']={'comparison_reliable':False,'comparison_score':0.}
 root=Path(__file__).resolve().parent;out=root/args.receipt
 if out.exists():raise ValueError('Preserve existing receipt')
 report={'passed':True,'scope':'synthetic local appearance comparison, not identity accuracy','tests':results,'source_sha256':hashlib.sha256((root/'photometric_ribbon.py').read_bytes()).hexdigest()}
 out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
