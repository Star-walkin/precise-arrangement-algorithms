"""H4 offline paths, fixed inputs and read-only frozen helpers."""
import hashlib, importlib.util, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
H3=HERE.parent/'postprocess_b_20261031_v3'
H2=HERE.parent/'postprocess_b_20261030_v2'
H1=HERE.parent/'postprocess_b_20261030_v1'
ROOT=HERE.parents[2]
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
surface=module('h4_frozen_h2_surface',H2/'surface_tracks.py')
h2_connect=module('h4_frozen_h2_connector',H2/'connect_paths.py')

def inputs():
    p=HERE/'inputs.json'
    if p.exists():return load(p)
    oldfreeze=load(H3/'freeze_final.json')
    assert all(sha(q)==s for q,s in (oldfreeze['sources']|oldfreeze['reused_sources']).items()),'Frozen H3 sources changed'
    config=load(H3/'inputs.json');config['version']='IR-H4.0.0'
    config['h3_input_path']=str(H3/'inputs.json');config['h3_input_sha256']=sha(H3/'inputs.json')
    config['new_inference_in_H4']=0;config['reuse_conditions_and_responses_unchanged']=True
    config['task']='Re-estimate straight rope axis and extent from B center/dual-valley evidence; protect observed heat ridge against premature F3 connection'
    config['pilot_ids']=list(dict.fromkeys(config['pilot_ids']+['R_static_20260922_01__f000095__right']))
    config['budget']='fixed previous12 plus user counterexample, maximum one necessary revision, then one292 image run; no new training or inference'
    config['continue_gate']='Aligned finite float outputs, entry association, no unsupported infinite line; qualitative user counterexample full visible straight recovery checked; failures retained'
    config['identity_accuracy_not_measured']=True
    assert sha(config['model_B']['path'])==config['model_B']['sha256']
    dump(p,config);return config
