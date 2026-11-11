from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
R20=ROOT.parent/'RopeEdgeSuite_R20_20261110'
sys.path.insert(0,str(R20))
from common20 import read,dump,plain,sha,pixelsha,plan,record,load,PITCH_PRIOR,np,json,gzip

def gzread(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return json.load(f)

def gzwrite(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.part')
    with gzip.open(tmp,'wt',encoding='utf8',compresslevel=2) as f:json.dump(plain(obj),f,ensure_ascii=False,allow_nan=False)
    tmp.replace(path)

def frontend(id,i):
    fr=read(R20/'f3_v2'/id/'records'/f'{i:06d}.json')
    c=fr.get('fragments24_correction') or fr.get('fragments23_correction')
    if c:
        assert sha(c['frontend_override_path'])==c['frontend_override_sha256']
        return gzread(c['frontend_override_path'])
    return record(id,i)

def count20(id,i):return gzread(R20/'counts_v1'/id/'frames'/f'{i:06d}.json.gz')
def geometry(id,i):return gzread(R20/'geometry_v5'/id/'frames'/f'{i:06d}.json.gz')
