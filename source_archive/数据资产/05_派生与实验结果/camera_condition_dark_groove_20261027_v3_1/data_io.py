"""D3仅访问既有开放400图；冻结D2黄点、D1 ROI/增强像素及C3源码。"""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
D2=HERE.parent/'camera_condition_dark_boundaries_20261027_v2'
spec=importlib.util.spec_from_file_location('d3_frozen_io',D2/'data_io.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha,read_json,dump,read_image,write_image=old.sha,old.read_json,old.dump,old.read_image,old.write_image
ROOT,DATA,C3,D1=old.ROOT,old.DATA,old.C3,old.D1
_inventory=None

def selection(mode='all'):return old.selection(mode)

def inputs(row):
    global _inventory
    if _inventory is None:
        p=D2/'all_002/output_manifest.json'
        if sha(p)!=read_json(HERE/'protocol.json')['d2_output_manifest_sha256']:
            raise ValueError('D2 manifest changed')
        _inventory={f['path']:f['sha256'] for c in read_json(p)['cases'] for f in c['files']}
    im,roi,_,enh,pitch,_,deps=old.inputs(row)
    rel=Path(row['output_file']);role=rel.parent.name
    part=Path('记录/cases')/role/(rel.stem+'.json');p=D2/'all_002'/part
    digest=sha(p)
    if digest!=_inventory[part.as_posix()]:raise ValueError('D2 fixed observations changed')
    deps[str(p)]=digest
    c3=read_json(C3/'boundaries/records'/role/(rel.stem+'.json'))
    return im,roi,enh,pitch,read_json(p),c3,deps

def sources():
    files=[HERE/n for n in ['data_io.py','run.py','groove_field.py','ordered_paths.py','protocol.json']]
    result=old.sources()
    result.update({str(p):sha(p) for p in [*files,D2/'data_io.py',D2/'all_002/output_manifest.json']})
    return dict(sorted(result.items()))

def initialize():
    if (HERE/'protocol.json').exists():raise FileExistsError('Protocol exists')
    _,pilot=selection()
    dump(HERE/'protocol.json',{'version':'DR-D3.0.0','name':'暗沟约束 D3','date':'2026-10-27',
      'input_count':400,'pilot_images':sorted(pilot),
      'd2_output_manifest_sha256':sha(D2/'all_002/output_manifest.json'),
      'frozen_d2_source_sha256':old.sources(),
      'fixed_observations':'All D2.1.1 yellow coordinates retained; old group IDs are provenance, not hard identity constraints. No newly invented yellow points.',
      'fixed_pixels':'D1 dark ROI and enhanced ROI bytes unchanged; original RGB unchanged.',
      'budget':'Up to3 fixed12 image-guided algorithm pilots, one same-parameter geometry-only control, then freeze a visually supported version and run400. Preserve failures. No full-set parameter tuning.',
      'review':'Same D2 fixed12; independent original/enhanced/old/new review of fixed6 and preselected extra12 after freeze. No labels.',
      'acceptance':['Keep all400 inputs and old points including rejections/unassigned points.',
         'Visible high-curvature wide groove adherence must improve, not just point residual or smooth appearance.',
         'Do not increase widespread wire-strand/compression-block following; no unsupported long-gap fill.',
         'New labels mean visual ordering only, not material identity or contact.'],
      'scope':'Offline dark boundary regrouping/image-guided fitting only; original C3 and D2 are not modified.',
      'html':False,'equipment_access':False,'reserved_access':False,'ground_truth_accuracy':None})
    dump(HERE/'index_before.json',{str(p):sha(p) for p in [ROOT/'数据资产/README.md',HERE.parent/'README.md',ROOT/'工程导航/05_实验版本档案/README.md']})

if __name__=='__main__':initialize()
