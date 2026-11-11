from pathlib import Path
import json,gzip,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'RopeEdgeSuite_R12_20261109'
RUN12=PREV/'runs/dual_20260923_113745'
FRONT12=PREV/'frontend/dual_20260923_113745_v2'
R11=ROOT.parent/'RopeEdgeSuite_R11_20261108'
BASE=ROOT.parent/'RopeEdgeSuite_R1_20261107'
D=2238.7;PITCH=23.29412648284868
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v
def dump(path,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part');tmp.write_text(json.dumps(plain(v),ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(path)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def record(i,run=RUN12):
    with gzip.open(run/'frames'/f'{i:06d}.json.gz','rt',encoding='utf-8') as f:return json.load(f)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pixelsha(im):return hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()
