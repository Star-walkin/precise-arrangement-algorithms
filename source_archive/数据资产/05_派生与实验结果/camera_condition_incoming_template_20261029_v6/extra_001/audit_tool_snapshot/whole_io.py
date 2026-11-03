"""固定开放开发400图入口。旧算法只读，所有输入先校验字节哈希。"""
from pathlib import Path
import csv,hashlib,json
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
ASSETS=HERE.parent
DATA=ASSETS/'camera_condition_images_20261009_v2'
F3=ASSETS/'camera_condition_full_surface_joint_20261028_v3/all_001'
C3=ASSETS/'camera_condition_grouped_conic_20261026_v6/all_001'
C4=ASSETS/'camera_condition_full_surface_20261026_v7/all_001'
A1=ASSETS/'camera_condition_incoming_rope_20261026_v1/all_001'
A2=ASSETS/'camera_condition_incoming_rope_20261026_v2/all_001'
VERSION='IR-A6.0.0'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 def conv(x):
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  if isinstance(x,Path):return str(x)
  raise TypeError(type(x).__name__)
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=conv,allow_nan=False)+'\n',encoding='utf-8')
def read_image(p,flag=1):
 a=cv2.imdecode(np.fromfile(str(p),np.uint8),flag)
 if a is None:raise ValueError(str(p))
 return a
def write_image(p,a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 ok,b=cv2.imencode(p.suffix,a,[cv2.IMWRITE_JPEG_QUALITY,94] if p.suffix=='.jpg' else [])
 if not ok:raise ValueError(str(p))
 b.tofile(str(p))
def selection(mode):
 p=read_json(HERE/'protocol.json')
 assert sha(DATA/'manifest.csv')==p['input_manifest_sha256']
 rows=list(csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8-sig')))
 assert len(rows)==400 and sum(r['camera_role']=='left' for r in rows)==200
 for r in rows:
  if '20260925' in r['source_file']+r['output_file'] or any(s in r['dataset_use_status'].lower() for s in ['reserved','heldout']):raise ValueError('Reserved input')
 return rows if mode=='all' else [r for r in rows if r['output_file'] in p['extra_review_inputs' if mode=='extra' else 'pilot_inputs']]
_HASHES=None
def checked(p):
 global _HASHES
 if _HASHES is None:
  _HASHES={};protocol=read_json(HERE/'protocol.json')
  for s,h in protocol['parent_manifest_sha256'].items():
   m=Path(s);assert sha(m)==h
   j=read_json(m)
   if 'cases' in j:
    for c in j['cases']:
     for f in c['files']:
      q=f.get('absolute_path') or str(m.parent/f.get('path',f.get('relative_path')))
      _HASHES[str(Path(q))]=f['sha256']
   else:
    for f in j.get('files',j.get('file_manifest',[])):
     q=Path(f['path']);q=q if q.is_absolute() else m.parent/q
     _HASHES[str(q)]=f['sha256']
 h=sha(p)
 if _HASHES.get(str(p))!=h:raise ValueError('Parent prediction changed: '+str(p))
 return h
def inputs(r):
 rel=Path(r['output_file']);role=rel.parent.name;name=rel.stem
 paths={'surface':C4/'roi/surface_masks'/role/rel.name,
  'a1_record':A1/'records'/role/(name+'.json'),'a2_record':A2/'records'/role/(name+'.json'),
  'c3_record':C3/'boundaries/records'/role/(name+'.json'),
  'f3_record':F3/'记录/cases'/role/(name+'.json')}
 for key in ['merged','inferred','exclusion']:paths[key]=F3/'记录/masks'/key/role/rel.name
 deps={str(p):checked(p) for p in paths.values()}
 ori=DATA/rel;assert sha(ori)==r['output_sha256'];deps[str(ori)]=r['output_sha256']
 d={k:(read_json(p) if p.suffix=='.json' else read_image(p,0)) for k,p in paths.items()}
 d['original']=read_image(ori);d['pitch']=float(d['c3_record']['record']['pitch_px'])
 refs=[]
 for src,cs in [('C3',d['c3_record']['fit']['curves']),('D4_full',d['f3_record']['observations']['fit']['curves'])]:
  for c in cs:
   xy=c['sampled_points_xy']
   if len(xy)<8:continue
   refs.append({'curve_id':src+'_'+c['curve_id'],'source':src,'sampled_points_xy':xy,'point_states':c.get('point_states',[])})
 d['curve_refs']=refs;d['input_dependency_sha256']=deps
 return d
