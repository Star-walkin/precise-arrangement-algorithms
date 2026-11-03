"""A2 固定开放400图入口：校验原图及A1/C3/C4自动产物，禁止任意目录挑图。"""
from pathlib import Path
import importlib.util
import json

HERE=Path(__file__).resolve().parent
A1=HERE.parent/'camera_condition_incoming_rope_20261026_v1'
spec=importlib.util.spec_from_file_location('a2_parent_io',A1/'experiment_io.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha,read_json,dump,read_image,write_image=old.sha,old.read_json,old.dump,old.read_image,old.write_image
ROOT,DATA,C3,C4=old.ROOT,old.DATA,old.C3,old.C4
_a1_hashes=None

def selection(mode):
    rows,qa=old.select('all')
    p=read_json(HERE/'protocol.json')
    if sha(A1/'all_001/output_manifest.json')!=p['a1_output_manifest_sha256']:
        raise ValueError('A1 output inventory changed')
    pilot=set(p['pilot_images'])
    assert len(rows)==400 and len(pilot)==49
    return ([r for r in rows if r['output_file'] in pilot] if mode=='pilot' else rows),qa,pilot

def inputs(row):
    global _a1_hashes
    image,surface,c3,c4,deps=old.inputs(row)
    if _a1_hashes is None:
        _a1_hashes={r['path']:r['sha256'] for r in read_json(A1/'all_001/output_manifest.json')['files']}
    rel=Path(row['output_file']);p=Path('records')/rel.parent.name/(rel.stem+'.json')
    actual=sha(A1/'all_001'/p)
    if actual!=_a1_hashes[p.as_posix()]:raise ValueError('Frozen A1 prediction changed')
    deps[str(A1/'all_001'/p)]=actual
    return image,surface,c3,c4,read_json(A1/'all_001'/p),deps

def sources():
    files=[HERE/n for n in ['experiment_io.py','run_experiment.py','paired_connection.py',
          'free_refinement.py','illumination_c3.py','render_results.py','protocol.json']]
    files += [A1/'experiment_io.py',A1/'protocol.json']
    import illumination_c3
    files += [Path(p) for p in getattr(illumination_c3,'SOURCE_DEPENDENCIES',[])]
    return {str(p):sha(p) for p in sorted(set(files))}

def initialize():
    from datetime import datetime,timezone
    path=HERE/'protocol.json'
    if path.exists():raise FileExistsError(path)
    rows,qa=old.select('all')
    records=[json.loads(l) for l in (A1/'all_001/run_log.jsonl').read_text(encoding='utf-8').splitlines()]
    singles={r['output_file'] for r in records if r['connection_status']=='SINGLE_IMAGE_HYPOTHESIS'}
    pilot=qa|singles
    assert len(pilot)==49 and len(singles)==11
    dump(path,{'version':'IR-A2.0.0','name':'照明归一化与双边接续 A2','created_utc':datetime.now(timezone.utc).isoformat(),
        'input_manifest_sha256':sha(old.MANIFEST),'baseline_manifest_sha256':sha(old.BASELINE),
        'a1_output_manifest_sha256':sha(A1/'all_001/output_manifest.json'),
        'denominator':400,'cameras':{'left':200,'right':200},'pilot_images':sorted(pilot),
        'pilot_meaning':'ExistingQA38 union every A1 auto-single11. Open development, not blind test. No manual contact labels enter algorithm.',
        'cached_automatic_input':'A1 frozen free-segment predictions for same400; all127 unknown and15 direction-only retained',
        'preprocessing_comparison':['raw','global_gamma','illumination'],
        'parameters_policy':'Uniform fixed parameters per variant; every revision separate run with source snapshot. No file-specific threshold.',
        'budget':'At most3 core revisions on fixed49, then freeze chosen source for full400. Stop scoring branch if visual wrong-route correction has no net evidence.',
        'acceptance':['Compare untouched original with raw/enhanced/C3 observations and paired free edges.',
            'Unknown is not correct. Increased contrast/yellow count/longer line is not accuracy.',
            'Keep straight-continuation hypothesis against every bend; no forced conic attachment.',
            'Image-supported continuation is not physical contact or material identity.'],
        'outputs':['400 enhanced surface ROI images, exterior0','original and enhanced overlays','C3 new observations/models','free optical width/angle','continuation evidence and rejects'],
        'equipment_access':False,'reserved_inputs':False,'online_replacement':False})
    dump(HERE/'index_before.json',{str(p):sha(p) for p in [ROOT/'数据资产/README.md',HERE.parent/'README.md',ROOT/'工程导航/05_实验版本档案/README.md']})
    print('Initialized400; fixed pilot49')

if __name__=='__main__':initialize()
