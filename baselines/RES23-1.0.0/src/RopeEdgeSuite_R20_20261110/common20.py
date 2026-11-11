from pathlib import Path
import json,gzip,hashlib,importlib.util,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
R19=PARENT/'RopeEdgeSuite_R19_20261110'
R18=PARENT/'RopeEdgeSuite_R18_20261110'
LC1=PARENT/'LC1_locked_column_20261105_v1'
BASE=PARENT/'RopeEdgeSuite_R1_20261107'
PITCH_PRIOR=23.29412648284868
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(a) for k,a in v.items()}
    if isinstance(v,(tuple,list)):return [plain(a) for a in v]
    return v
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.part')
    tmp.write_text(json.dumps(plain(v),ensure_ascii=False,allow_nan=False),encoding='utf8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
    return h.hexdigest()
def pixelsha(im):return hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def plan():return read(R19/'plan.json')['videos']
def selected_run(id):return Path(next(r for r in read(R19/'delivery_manifest.json')['videos'] if r['id']==id)['selected_run'])
def record(id,i):
    with gzip.open(selected_run(id)/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:return json.load(f)
