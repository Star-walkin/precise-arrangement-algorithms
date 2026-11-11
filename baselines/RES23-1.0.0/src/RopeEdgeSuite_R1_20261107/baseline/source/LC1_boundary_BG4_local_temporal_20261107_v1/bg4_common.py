"""BG4 local evidence and causal count; baseline files are read-only."""
import sys,json,hashlib,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BG3=ROOT.parent/'LC1_boundary_BG3_point_global_20261107_v2'
BG1=ROOT.parent/'LC1_boundary_BG1_20261106_v1'
BASE=BG1/'full_002'
SOURCE=ROOT.parents[3]/'双相机同步'/'dual_20260924_111500'/'left_DA9627065.avi'
SOURCE_SHA='ead815ac73239dec6fcbd141ffb2873fb2ac575dfb79347a036ee877b65255f8'
D=2238.7; SCALE=.5; PITCH=23.29412648284868
sys.path.insert(0,str(BG3))
from global_points import Row,sample,supported_path,model_envelope
sys.path.insert(0,str(BG1))
from gated_rope import SurfaceObserver
sys.path.insert(0,str(ROOT))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x
def dump(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(plain(data),ensure_ascii=False,allow_nan=False),encoding='utf8')
    tmp.replace(path)
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
