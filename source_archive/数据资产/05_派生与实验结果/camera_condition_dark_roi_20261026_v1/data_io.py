"""只读取已授权400图及冻结卷面掩码；不连接设备。"""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
A1=HERE.parent/'camera_condition_incoming_rope_20261026_v1'
spec=importlib.util.spec_from_file_location('dark400_parent_io',A1/'experiment_io.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha,read_json,dump,read_image,write_image=old.sha,old.read_json,old.dump,old.read_image,old.write_image
ROOT,DATA,C3,C4=old.ROOT,old.DATA,old.C3,old.C4

def selection(mode='all'):
    rows,qa=old.select('all')
    pilot=set()
    for camera in ['left','right']:
        available=sorted(r['output_file'] for r in rows if r['camera_role']==camera and r['output_file'] in qa)
        assert len(available)==19
        pilot.update(available[i] for i in [0,3,6,10,14,18])
    return ([r for r in rows if r['output_file'] in pilot] if mode=='pilot' else rows),pilot

def inputs(row):
    image,surface,c3,c4,deps=old.inputs(row)
    return image,surface,float(c3['record']['pitch_px']),deps

def sources():
    from dark_c3 import SOURCE_DEPENDENCIES
    files=[HERE/n for n in ['data_io.py','run.py','dark_roi.py','dark_c3.py','protocol.json']]
    files += [A1/'experiment_io.py',A1/'protocol.json',*SOURCE_DEPENDENCIES]
    return {str(p):sha(p) for p in sorted(set(files))}

def initialize():
    if (HERE/'protocol.json').exists():raise FileExistsError('protocol exists')
    rows,pilot=selection()
    dump(HERE/'protocol.json',{'version':'DR-C3.1.0','date':'2026-10-26','input_count':400,
        'input_manifest_sha256':sha(old.MANIFEST),'baseline_manifest_sha256':sha(old.BASELINE),
        'camera_counts':{'left':200,'right':200},'pilot_images':sorted(pilot),
        'task':'Dark surface ROI, contrast expansion, original C3 detector/grouping/conic fit on entire dark ROI',
        'scope':'Same400 open development images. No incoming-rope prerequisite or corridor.',
        'fixed_pilot_rule':'Each camera sorted existing QA19 indices0,3,6,10,14,18;12 total.',
        'budget':'At most2 pilot revisions then freeze sources and process400.',
        'roi_outside_value':0,'output_groups':3,'html':False,'equipment_access':False,'reserved_access':False,
        'data_status':'unlabeled_open_development_no_new_test_split',
        'acceptance':'Exact400 per group, original ROI pixels preserved, actual contrast spread, C3 points/curves restricted to dark ROI; not ground-truth accuracy.'})
    indices=[ROOT/'数据资产/README.md',HERE.parent/'README.md',ROOT/'工程导航/05_实验版本档案/README.md']
    dump(HERE/'index_before.json',{str(p):sha(p) for p in indices})

if __name__=='__main__':initialize()
