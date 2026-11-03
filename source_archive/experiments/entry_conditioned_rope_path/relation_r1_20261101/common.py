"""R1 offline-only common paths. No label or device imports."""
from pathlib import Path
import json,hashlib,importlib.util,sys
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[1]
H6=BASE/'postprocess_b_20261031_v6';PAIR=BASE/'runs/run_20261030_002'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def mod(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def h6():
    freeze=load(H6/'freeze_final.json');assert all(sha(p)==h for p,h in (freeze['sources']|freeze['reused_sources']).items())
    old=sys.modules.get('common');sys.path.insert(0,str(H6))
    try:
        c=mod('_r1_frozen_h6_common',H6/'common.py');sys.modules['common']=c
        return mod('_r1_frozen_h6_gallery',H6/'run_h6.py')
    finally:
        sys.path.remove(str(H6))
        if old is not None:sys.modules['common']=old
        else:sys.modules.pop('common',None)
