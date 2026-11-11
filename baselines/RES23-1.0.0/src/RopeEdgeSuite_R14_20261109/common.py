from pathlib import Path
import json,gzip,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
R11=ROOT.parent/'RopeEdgeSuite_R11_20261108'
R12=ROOT.parent/'RopeEdgeSuite_R12_20261109'
R13=ROOT.parent/'RopeEdgeSuite_R13_20261109'
RUN11=R11/'runs/dual_20260923_113745_refined'
RUN12=R12/'runs/dual_20260923_113745'
FRONT12=R12/'frontend/dual_20260923_113745_v2'
BASE=ROOT.parent/'RopeEdgeSuite_R1_20261107'
D=2238.7;PITCH=23.29412648284868
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p,x):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix(p.suffix+'.part')
    tmp.write_text(json.dumps(plain(x),ensure_ascii=False,allow_nan=False),encoding='utf8');tmp.replace(p)
def record(i,run=RUN12):
    with gzip.open(run/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:return json.load(f)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pixelsha(im):return hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()
