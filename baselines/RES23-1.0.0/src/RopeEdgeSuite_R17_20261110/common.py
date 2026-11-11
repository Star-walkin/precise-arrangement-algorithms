from pathlib import Path
import json,gzip,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=Path(r'E:/study/program/济宁/双相机同步')
PARENT=ROOT.parent
BASE=PARENT/'RopeEdgeSuite_R1_20261107'
R11=PARENT/'RopeEdgeSuite_R11_20261108'
R12=PARENT/'RopeEdgeSuite_R12_20261109'
R13=PARENT/'RopeEdgeSuite_R13_20261109'
R14=PARENT/'RopeEdgeSuite_R14_20261109'
D=2238.7;PITCH=23.29412648284868
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    return v
def dump(path,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(plain(v),ensure_ascii=False,allow_nan=False),encoding='utf8');tmp.replace(path)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def pixelsha(im):return hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()
def record(run,i):
    with gzip.open(Path(run)/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:return json.load(f)
def save_record(run,i,r):
    path=Path(run)/'frames'/f'{i:06d}.json.gz';path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix('.part')
    with gzip.open(tmp,'wt',encoding='utf8',compresslevel=2) as f:json.dump(plain(r),f,ensure_ascii=False,allow_nan=False)
    tmp.replace(path)
