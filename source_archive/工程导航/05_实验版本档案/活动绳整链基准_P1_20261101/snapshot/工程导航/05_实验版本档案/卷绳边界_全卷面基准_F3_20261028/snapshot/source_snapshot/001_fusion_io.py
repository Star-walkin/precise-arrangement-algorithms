"""F1只读取固定开放400图及已冻结的C3/D4结果，不导入检测或设备入口。"""
from pathlib import Path
import csv,hashlib,json
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
ASSETS=HERE.parent
ROOT=ASSETS.parents[1]
DATA=ASSETS/'camera_condition_images_20261009_v2'
C3=ASSETS/'camera_condition_grouped_conic_20261026_v6/all_001'
D4=ASSETS/'camera_condition_dark_width_20261027_v4/all_001'
D1=ASSETS/'camera_condition_dark_roi_20261026_v1/all_001'
C4=ASSETS/'camera_condition_full_surface_20261026_v7/all_001'
MASK_KEYS=('surface','bright_owner','dark_owner','uncovered','c3_boundaries','d4_boundaries','merged','inferred','accepted_yellow')
_inventory=None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read_json(path):return json.loads(Path(path).read_text(encoding='utf8'))

def dump(path,value):
    def convert(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert)+'\n',encoding='utf8')

def read_image(path,flag=cv2.IMREAD_COLOR):
    im=cv2.imdecode(np.fromfile(str(path),np.uint8),flag)
    if im is None:raise ValueError('decode failed: '+str(path))
    return im

def write_image(path,im):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    ok,buf=cv2.imencode(p.suffix,im,[cv2.IMWRITE_JPEG_QUALITY,94] if p.suffix=='.jpg' else [])
    if not ok:raise ValueError('encode failed')
    buf.tofile(str(p))

def selection(mode='all'):
    p=read_json(HERE/'protocol.json')
    assert sha(DATA/'manifest.csv')==p['input_manifest_sha256']
    with (DATA/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==400 and sum(r['camera_role']=='left' for r in rows)==200
    for r in rows:
        if '20260925' in r['source_file']+r['output_file'] or any(v in r['dataset_use_status'].lower() for v in ('reserved','heldout')):raise ValueError('reserved input')
    return [r for r in rows if mode=='all' or r['output_file'] in p['pilot_images']]

def source_paths(row):
    r=Path(row['output_file']);role=r.parent.name;n=r.name;j=r.stem+'.json'
    return {'original':DATA/r,'surface':C4/'roi/surface_masks'/role/n,
      'bright':C3/'roi/reference_masks'/role/n,'dark':D1/'记录/masks/dark_roi'/role/n,
      'enhanced':D1/'图像结果/02_对比度增强ROI'/role/n,
      'c3_smooth':C3/'boundaries/smooth_boundaries'/role/n,'c3_inferred':C3/'boundaries/geometric_gap'/role/n,
      'c3_record':C3/'boundaries/records'/role/j,
      'd4_smooth':D4/'记录/masks/smooth'/role/n,'d4_inferred':D4/'记录/masks/inferred'/role/n,
      'd4_record':D4/'记录/cases'/role/j}

def input_inventory():
    global _inventory
    if _inventory is None:
        protocol=read_json(HERE/'protocol.json')
        for p,h in protocol['parent_manifest_sha256'].items():
            if sha(p)!=h:raise ValueError('parent manifest changed: '+p)
        inv={r['path']:r['sha256'] for r in read_json(C3/'integrity_audit.json')['file_manifest'] if r.get('sha256')}
        inv.update({str(D1/r['path']):r['sha256'] for r in read_json(D1/'output_manifest.json')['files']})
        for c in read_json(C4/'output_manifest.json')['cases']:
            for r in c['files']:inv[r['absolute_path']]=r['sha256']
        for c in read_json(D4/'output_manifest.json')['cases']:
            for r in c['files']:inv[str(D4/r['path'])]=r['sha256']
        _inventory=inv
    return _inventory

def inputs(row):
    paths=source_paths(row);inv=input_inventory();deps={}
    for key,path in paths.items():
        actual=sha(path);expected=row['output_sha256'] if key=='original' else inv.get(str(path))
        if actual!=expected:raise ValueError('frozen input changed: '+str(path))
        deps[str(path)]=actual
    data={k:(read_json(p) if k.endswith('_record') else read_image(p,cv2.IMREAD_COLOR if k in ('original','enhanced') else cv2.IMREAD_GRAYSCALE)) for k,p in paths.items()}
    if data['original'].shape!=(3000,4096,3):raise ValueError('original dimensions')
    for k,im in data.items():
        if k.endswith('_record'):continue
        if im.shape[:2]!=(3000,4096):raise ValueError('dimensions: '+k)
        if k not in ('original','enhanced') and np.any((im!=0)&(im!=255)):raise ValueError('nonbinary: '+k)
    return data,deps

def sources():return {str(HERE/n):sha(HERE/n) for n in ('fusion_io.py','fusion.py','run_fusion.py','protocol.json')}

def baseline_unchanged():
    return all(sha(p)==h for p,h in read_json(HERE/'protocol.json')['baseline_source_sha256'].items())

if __name__=='__main__':
    if (HERE/'protocol.json').exists():raise FileExistsError('protocol')
    baseline=dict(read_json(D4.parent/'freeze_all.json')['source_sha256'])
    baseline.update(read_json(C3.parent/'freeze_all_001.json')['source_sha256'])
    assert all(sha(p)==h for p,h in baseline.items())
    dump(HERE/'protocol.json',{'version':'RB-F1.0.0','name':'亮暗融合 F1','date':'2026-10-28',
      'input_manifest_sha256':sha(DATA/'manifest.csv'),'input_count':400,'camera_counts':{'left':200,'right':200},
      'pilot_images':read_json(D4.parent/'protocol.json')['pilot_images'],
      'baseline_source_sha256':baseline,
      'parent_manifest_sha256':{str(p):sha(p) for p in (C3/'integrity_audit.json',D4/'output_manifest.json',D1/'output_manifest.json',C4/'output_manifest.json')},
      'owners':{'C3':'surface & bright_reference','D4':'surface & dark_roi & ~bright_reference'},
      'fusion':'Exact union of saved masks clipped by owner. No refit, extrapolation, bridging or cross-source identity assignment.',
      'background':'Original full surface; D1 accepted enhancement replaces only D4 owner. Outside surface=0.',
      'budget':'One fixed12 preview followed by source freeze and400. Display/structural bug repairs get separate runs.',
      'versions':{'bright':'RB-C3.0.0','dark':'DR-D4.0.0'},
      'accuracy':None,'equipment_access':False,'reserved_access':False,'html':False})
