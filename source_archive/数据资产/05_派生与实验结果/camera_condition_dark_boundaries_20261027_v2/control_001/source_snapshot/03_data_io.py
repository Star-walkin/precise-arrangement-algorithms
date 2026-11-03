"""D2只读复用已获用户认可的D1暗ROI与增强像素，原C3不改。"""
from pathlib import Path
import importlib.util
import json
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
D1=HERE.parent/'camera_condition_dark_roi_20261026_v1'
spec=importlib.util.spec_from_file_location('d2_parent_io',D1/'data_io.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha,read_json,dump,read_image,write_image=old.sha,old.read_json,old.dump,old.read_image,old.write_image
ROOT,DATA,C3,C4=old.ROOT,old.DATA,old.C3,old.C4
_inventory=None

def selection(mode='all'):
    return old.selection(mode)

def inputs(row):
    global _inventory
    p=read_json(HERE/'protocol.json')
    if _inventory is None:
        manifest=D1/'all_001/output_manifest.json'
        if sha(manifest)!=p['d1_output_manifest_sha256']:raise ValueError('D1 inventory changed')
        _inventory={item['path']:item['sha256'] for item in read_json(manifest)['files']}
    image,surface,pitch,deps=old.inputs(row)
    rel=Path(row['output_file']);role=rel.parent.name;name=rel.name
    paths=[Path('记录/masks/dark_roi')/role/name,Path('图像结果/02_对比度增强ROI')/role/name,
           Path('记录/cases')/role/(rel.stem+'.json')]
    for part in paths:
        source=D1/'all_001'/part;value=sha(source)
        if value!=_inventory[part.as_posix()]:raise ValueError('Frozen D1 changed: '+str(source))
        deps[str(source)]=value
    roi=read_image(D1/'all_001'/paths[0],0)
    enhanced_roi=read_image(D1/'all_001'/paths[1])
    enhanced=cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
    enhanced[roi>0]=enhanced_roi[roi>0]
    enhanced[surface==0]=0
    return image,roi,enhanced,enhanced_roi,pitch,read_json(D1/'all_001'/paths[2]),deps

def sources():
    files=[HERE/n for n in ['data_io.py','run.py','dark_observation.py','dark_curves.py','protocol.json']]
    files += [D1/'data_io.py',D1/'protocol.json',old.A1/'experiment_io.py',old.A1/'protocol.json']
    files += [Path(p) for p in read_json(HERE/'protocol.json')['upstream_source_sha256']]
    return {str(p):sha(p) for p in sorted(set(files))}

def initialize():
    if (HERE/'protocol.json').exists():raise FileExistsError('Protocol already exists')
    rows,pilot=selection()
    baseline=[HERE.parent/'camera_condition_bright_rope_boundaries_20261025_v3/full_roi_observation.py',
              HERE.parent/'camera_condition_grouped_conic_20261026_v6/conic_family.py',
              D1/'dark_roi.py',D1/'dark_c3.py']
    dump(HERE/'protocol.json',{'version':'DR-D2.0.0','name':'暗纹曲线 D2','date':'2026-10-27','input_count':400,
        'd1_output_manifest_sha256':sha(D1/'all_001/output_manifest.json'),'pilot_images':sorted(pilot),
        'upstream_source_sha256':{str(p):sha(p) for p in baseline},
        'scope':'Keep D1 ROI/enhanced pixels; improve only dark boundary observations/grouping/fitting. No contact/identity task.',
        'budget':'Up to4 fixed12 pilot revisions including threshold-only simple control; then freeze and run400.',
        'comparison':'Same12 every round; compare original D1, simple relaxed B1/C3, D2. Fixed6 independent visual diagnostics; all failures retained.',
        'gates':['Actual original/enhanced groove support, no claims based on count alone.',
                 'Do not accept widespread compressed-block or wire-strand hallucinations.',
                 'Side curvature improvement must follow visible samples; no long unsupported extrapolation.',
                 'Keep every400 input including empty/rejected outputs; metadata and masks outside ROI0.'],
        'ground_truth_accuracy':None,'html':False,'equipment_access':False,'reserved_access':False})
    dump(HERE/'index_before.json',{str(p):sha(p) for p in [ROOT/'数据资产/README.md',HERE.parent/'README.md',ROOT/'工程导航/05_实验版本档案/README.md']})

if __name__=='__main__':initialize()
