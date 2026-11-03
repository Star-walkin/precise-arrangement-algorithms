"""H6 offline sources and fixed open-development protocol."""
import hashlib,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
H5=HERE.parent/'postprocess_b_20261031_v5'
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
surface=module('h6_frozen_h2_surface',H2/'surface_tracks.py')
h2_connect=module('h6_frozen_h2_connector',H2/'connect_paths.py')
def inputs():
    p=HERE/'inputs.json'
    if p.exists():return load(p)
    frozen=load(H5/'freeze_final.json')
    assert all(sha(q)==s for q,s in (frozen['sources']|frozen['reused_sources']).items())
    c=load(H5/'inputs.json');c['version']='IR-H6.0.0'
    c['h5_input_path']=str(H5/'inputs.json');c['h5_input_sha256']=sha(H5/'inputs.json')
    c['pilot_ids']=c['pilot_ids']+['dual_20260923_101517__pair0345__right','dual_20260923_101643__pair0255__right']
    c['task']='Restore continuous F3-derived center tracks and reject crossings or lane jumps in attached continuation; distinguish free-rope projection overlap from attachment.'
    c['budget']=dict(development_images=16,maximum_development_rounds=4,stop_after_no_generalizable_change_rounds=2,postfreeze_open_checks=8,full_run_images=292,new_model_inference=0,training=0)
    c['selection_protocol']='Fixed16 includes H5 fixed14 plus two supplied failures. Preserve all revisions. Recheck already-open8 after core freeze; no tuning on those8 or full292. Structural no-crossing is not physical identity validation.'
    c['extra_check_status']='Same already-viewed open8 regression checks, not fresh independent or blind data.'
    c['stopping_boundary']='Require continuous-source track provenance, no forbidden transition/surface intersections, fixed lane identity and meaningful failure/control diagnostics; preserve residual unknowns instead of guaranteeing global identity.'
    c['labels_read']=False;c['reserved_access']=False;c['devices']=False
    assert len(set(c['pilot_ids']))==16 and len(c['jobs'])==292
    assert sha(c['model_B']['path'])==c['model_B']['sha256']
    dump(p,c);return c
if __name__=='__main__':inputs()
