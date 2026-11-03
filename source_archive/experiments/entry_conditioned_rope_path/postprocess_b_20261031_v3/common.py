"""H3 offline-only paths and explicitly reused frozen H2 components."""
import hashlib,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
H2=HERE.parent/'postprocess_b_20261030_v2'
H1=HERE.parent/'postprocess_b_20261030_v1'
ROOT=HERE.parents[2]
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
surface=module('h3_frozen_h2_surface',H2/'surface_tracks.py')
h2_connect=module('h3_frozen_h2_connector',H2/'connect_paths.py')

def inputs():
    p=HERE/'inputs.json'
    if p.exists():return load(p)
    oldfreeze=load(H2/'freeze_final.json')
    assert all(sha(p)==s for p,s in oldfreeze['sources'].items()),'H2 source changed since freeze'
    config=load(H2/'inputs.json');config['version']='IR-H3.0.0'
    config['h2_input_path']=str(H2/'inputs.json');config['h2_input_sha256']=sha(H2/'inputs.json')
    config['new_inference_in_H3']=0;config['reuse_conditions_and_responses_unchanged']=True
    config['task']='A7 straight free segments; surface paths constrained to existing F3 center tracks; prefer small direction variation'
    config['budget']='same frozen12 pilot, at most one necessary correction, then one292-image full run; no training or model changes'
    assert sha(config['model_B']['path'])==config['model_B']['sha256']
    dump(p,config);return config
