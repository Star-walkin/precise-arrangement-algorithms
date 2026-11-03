"""F3固定开放400图及F2结果的只读输入；所有预测另存。"""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
F2=HERE.parent/'camera_condition_seam_joint_20261028_v2'
spec=importlib.util.spec_from_file_location('f3_f2_io',F2/'seam_io.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
ROOT,ASSETS,DATA,C3,D4,D1,C4=old.ROOT,old.ASSETS,old.DATA,old.C3,old.D4,old.D1,old.C4
sha,read_json,dump,read_image,write_image=old.sha,old.read_json,old.dump,old.read_image,old.write_image
_inventory=None

def selection(mode='all'):return old.selection(mode)

def inputs(row):
    global _inventory
    import cv2
    data,deps=old.inputs(row);rel=Path(row['output_file']);role=rel.parent.name
    if _inventory is None:
        p=F2/'all_001/output_manifest.json'
        assert sha(p)==read_json(HERE/'protocol.json')['f2_output_manifest_sha256']
        _inventory={r['path']:r['sha256'] for c in read_json(p)['cases'] for r in c['files']}
    for key,part in [('f2_image',Path('图像结果')/role/rel.name),
        ('f2_merged',Path('记录/masks/merged')/role/rel.name),
        ('f2_inferred',Path('记录/masks/inferred')/role/rel.name),
        ('f2_transition',Path('记录/masks/transition')/role/rel.name),
        ('f2_case',Path('记录/cases')/role/(rel.stem+'.json'))]:
        p=F2/'all_001'/part; h=sha(p)
        if h!=_inventory.get(part.as_posix()):raise ValueError('Frozen F2 file changed: '+str(p))
        data[key]=read_json(p) if key.endswith('_case') else read_image(p,cv2.IMREAD_COLOR if key=='f2_image' else cv2.IMREAD_GRAYSCALE)
        deps[str(p)]=h
    return data,deps

def baseline_unchanged():return all(sha(p)==h for p,h in read_json(HERE/'protocol.json')['baseline_source_sha256'].items())

def sources():
    result=dict(read_json(HERE/'protocol.json')['baseline_source_sha256'])
    result.update({str(HERE/n):sha(HERE/n) for n in ('surface_io.py','surface_observation.py','surface_fusion.py','run_surface.py','protocol.json')})
    return dict(sorted(result.items()))

if __name__=='__main__':
    if (HERE/'protocol.json').exists():raise FileExistsError('protocol')
    p=read_json(F2/'protocol.json');s=read_json(F2/'freeze_all.json')['source_sha256']
    assert all(sha(k)==h for k,h in s.items())
    dump(HERE/'protocol.json',{'version':'RB-F3.0.0','name':'全卷面协同 F3','date':'2026-10-28',
      'input_count':400,'camera_counts':{'left':200,'right':200},'pilot_images':p['pilot_images'],
      'input_manifest_sha256':p['input_manifest_sha256'],'baseline_source_sha256':s,
      'f2_output_manifest_sha256':sha(F2/'all_001/output_manifest.json'),
      'budget':'At most3 fixed12 pilot revisions, freeze code/parameters before same400, no full400 tuning. Preserve all failures.',
      'scope':'Actual frozen D4 detection/grouping/fitting on complete saved surface; combine C3/F2 visual context and new D4 image-supported paths to fill remaining transition gaps.',
      'gates':['Preserve original C3/D4/F1/F2 and saved surface ROI.',
        'No new points manufactured from curves; new yellow points must match actual full-surface D4 observations and accepted paths.',
        'F2 masks and display outside frozen transition region unchanged. Existing F2 boundaries retained.',
        'Added paths must stay inside surface and transition, avoid unknown/body, satisfy wide-groove and neighbor/row checks.',
        'Allow independently observed or one-side anchored valid fragments; never require missing old anchors merely to display real supported paths.',
        'Retain ambiguous/failed/empty outputs in full denominator, no physical material identity/accuracy claim.'],
      'comparison':'Same fixed12 F2/F3 plus full-surface D4 candidate display; inspect fixed seam crops selected from old ROI, not chosen by new result.',
      'accuracy':None,'reserved_access':False,'equipment_access':False,'html':False})
