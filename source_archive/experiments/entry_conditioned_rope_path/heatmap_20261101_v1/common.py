"""Offline H7 paths and immutable provenance helpers; no hardware imports."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parent.parent
H6=BASE/'postprocess_b_20261031_v6'
OLD_RUN=BASE/'runs/run_20261030_002'
OLD_INDEX=OLD_RUN/'data/index.json'
B_MODEL=OLD_RUN/'branches/B/model/checkpoints/step_0750.pt'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
