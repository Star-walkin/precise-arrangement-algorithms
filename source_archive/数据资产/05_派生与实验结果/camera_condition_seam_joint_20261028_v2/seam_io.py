"""F2离线输入：固定开放400图、F1正式结果及C3/D4已冻结源。"""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
F1=HERE.parent/'camera_condition_bright_dark_fusion_20261028_v1'
spec=importlib.util.spec_from_file_location('f2_f1_io',F1/'fusion_io.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
ROOT,ASSETS,DATA,C3,D4,D1,C4=old.ROOT,old.ASSETS,old.DATA,old.C3,old.D4,old.D1,old.C4
sha,read_json,dump,read_image,write_image=old.sha,old.read_json,old.dump,old.read_image,old.write_image
_f1_inventory=None

def selection(mode='all'):return old.selection(mode)

def inputs(row):
    import cv2,numpy as np
    global _f1_inventory
    data,deps=old.inputs(row);rel=Path(row['output_file']);role=rel.parent.name
    inv=old.input_inventory()
    if _f1_inventory is None:
        p=F1/'all_001/output_manifest.json'
        assert sha(p)==read_json(HERE/'protocol.json')['f1_output_manifest_sha256']
        _f1_inventory={r['path']:r['sha256'] for c in read_json(p)['cases'] for r in c['files']}
    exclusion=np.zeros(data['surface'].shape,np.uint8)
    for part in ('boundaries/incoming_rope_occlusion','boundaries/unknown_exclusion'):
        p=C3/part/role/rel.name;h=sha(p)
        if h!=inv.get(str(p)):raise ValueError('frozen exclusion changed')
        deps[str(p)]=h;exclusion|=read_image(p,0)
    p=D4/'记录/fields/unknown'/role/rel.name;h=sha(p)
    if h!=inv.get(str(p)):raise ValueError('frozen D4 exclusion changed')
    deps[str(p)]=h;exclusion|=read_image(p,0)
    for key,part in [('f1_image',Path('图像结果')/role/rel.name),
                     ('f1_merged',Path('记录/masks/merged')/role/rel.name),
                     ('f1_inferred',Path('记录/masks/inferred')/role/rel.name),
                     ('f1_case',Path('记录/cases')/role/(rel.stem+'.json'))]:
        p=F1/'all_001'/part;h=sha(p)
        if h!=_f1_inventory.get(part.as_posix()):raise ValueError('F1 saved result changed: '+str(p))
        data[key]=(read_json(p) if key=='f1_case' else read_image(p,cv2.IMREAD_COLOR if key=='f1_image' else cv2.IMREAD_GRAYSCALE));deps[str(p)]=h
    data['c3_exclusion']=exclusion;data['pitch']=float(data['c3_record']['record']['pitch_px'])
    return data,deps

def baseline_unchanged():return all(sha(p)==h for p,h in read_json(HERE/'protocol.json')['baseline_source_sha256'].items())

def sources():
    result=dict(read_json(HERE/'protocol.json')['baseline_source_sha256'])
    result.update({str(HERE/n):sha(HERE/n) for n in ('seam_io.py','seam_observation.py','seam_join.py','run_seam.py','protocol.json')})
    return dict(sorted(result.items()))

if __name__=='__main__':
    if (HERE/'protocol.json').exists():raise FileExistsError('protocol')
    p=read_json(F1/'protocol.json');sources=dict(p['baseline_source_sha256'])
    sources.update(read_json(F1/'freeze_all.json')['source_sha256'])
    assert all(sha(p)==h for p,h in sources.items())
    dump(HERE/'protocol.json',{'version':'RB-F2.0.0','name':'重叠接续 F2','date':'2026-10-28',
      'input_count':400,'camera_counts':{'left':200,'right':200},'pilot_images':p['pilot_images'],
      'input_manifest_sha256':p['input_manifest_sha256'],'baseline_source_sha256':sources,
      'f1_output_manifest_sha256':sha(F1/'all_001/output_manifest.json'),
      'scope':'Same400 open development images, C3+D4 redetection in overlap seam, group compatible image boundaries and connect with image support.',
      'budget':'Up to3 fixed12 pilot revisions. Freeze before400; no full400 tuning. Preserve failed runs.',
      'gates':['Actual new C3 and D4 observations on same overlap region; no replayed old points as new detections.',
        'Formal boundary and inference pixels outside transition region exactly match F1.',
        'New joins have image evidence, compatible endpoint tangents, unique ordered pairing and no strong-body/unknown crossing.',
        'Retain rejected/ambiguous joins and all400 denominator; no forced nearest-rope continuation.',
        'New source point grouping/geometry is image correspondence, not physical material identity.'],
      'comparison':'Same12; fixed6 seam visual crops from F1 fixed6; baseline F1 unchanged. Simple endpoint-only bridge is diagnostic comparator, not accepted truth.',
      'reserved_access':False,'equipment_access':False,'html':False,'accuracy':None})
