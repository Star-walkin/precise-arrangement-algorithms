"""H5 offline experiment paths and frozen input protocol."""
import hashlib,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
H4=HERE.parent/'postprocess_b_20261031_v4'
H3=HERE.parent/'postprocess_b_20261031_v3'
H2=HERE.parent/'postprocess_b_20261030_v2'
H1=HERE.parent/'postprocess_b_20261030_v1'
ROOT=HERE.parents[2]
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
surface=module('h5_frozen_h2_surface',H2/'surface_tracks.py')
h2_connect=module('h5_frozen_h2_connector',H2/'connect_paths.py')

def inputs():
    p=HERE/'inputs.json'
    if p.exists():return load(p)
    oldfreeze=load(H4/'freeze_final.json')
    assert all(sha(q)==s for q,s in (oldfreeze['sources']|oldfreeze['reused_sources']).items()),'H4 frozen source mismatch'
    c=load(H4/'inputs.json');samples=load(HERE/'diagnostics_source/fixed_samples_metadata.json')
    c['version']='IR-H5.0.0';c['h4_input_path']=str(H4/'inputs.json');c['h4_input_sha256']=sha(H4/'inputs.json')
    c['pilot_ids']=samples['development_ids'];c['extra_ids']=[r['id'] for r in samples['extra_open_check']]
    c['task']='Prefer supported straight extent, then small turn, then supported F3 center track; compare short hotspot competitors against longer entry-linked straight hypotheses.'
    c['budget']=dict(development_images=14,maximum_development_rounds=4,stop_after_no_clear_improvement_rounds=2,postfreeze_open_checks=8,full_run_images=292,model_inference=0,training=0)
    c['selection_protocol']='Use fixed14 failure/control cases; retain every round. Continue only with a diagnosed general scoring/candidate cause; no sample-specific thresholds. Freeze before extra8; do not tune on extra8. One full292 run with limitations recorded.'
    c['stopping_boundary']='Artifact stability, visible error checks and absence of fresh control regressions are finite gates, not proof of near-perfect target identity. If clear failures persist after bounded iterations, disclose them and preserve failed cases; never manufacture a guarantee.'
    c['new_inference_in_H5']=0;c['identity_accuracy_not_measured']=True
    c['extra_check_status']='already-open development sources, metadata-chosen, not independent blind holdout'
    assert sha(c['model_B']['path'])==c['model_B']['sha256']
    dump(p,c);return c
