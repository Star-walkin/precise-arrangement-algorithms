"""D4只复用开放400输入与固定D1像素；旧黄点不作为本版检测输入。"""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
D3=HERE.parent/'camera_condition_dark_groove_20261027_v3_1'
D2=HERE.parent/'camera_condition_dark_boundaries_20261027_v2'
spec=importlib.util.spec_from_file_location('d4_upstream_io',D2/'data_io.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha,read_json,dump,read_image,write_image=old.sha,old.read_json,old.dump,old.read_image,old.write_image
ROOT,DATA,D1,C3=old.ROOT,old.DATA,old.D1,old.C3
def selection(mode='all'):return old.selection(mode)
def inputs(row):
    im,roi,_,enh,pitch,record,deps=old.inputs(row)
    rel=Path(row['output_file'])
    c3=read_json(C3/'boundaries/records'/rel.parent.name/(rel.stem+'.json'))
    return im,roi,enh,pitch,c3,deps
def sources():
    parent=read_json(D3/'freeze_all.json')['source_sha256']
    result=dict(parent)
    result.update({str(HERE/n):sha(HERE/n) for n in ['data_io.py','run.py','shadow_width.py','period_guard.py','shadow_tracks.py','ribbon_curves.py','protocol.json']})
    return dict(sorted(result.items()))
def initialize():
    if (HERE/'protocol.json').exists():raise FileExistsError('protocol')
    _,pilot=selection()
    dump(HERE/'protocol.json',{'version':'DR-D4.0.0','name':'沟宽筛选 D4','date':'2026-10-27',
      'input_count':400,'pilot_images':sorted(pilot),'fixed_pixels':'Original400 and D1 dark ROI/enhancement unchanged.',
      'new_observations':'Latest user explicitly authorizes replacing false old yellow points. Detect fresh broad-groove points; old D2/D3 points are comparison only, never truth.',
      'frozen_baseline_sha256':read_json(D3/'freeze_all.json')['source_sha256'],
      'budget':'Up to3 fixed12 image-guided pilots; raw/cleaned shadow ablation retained per image. Then freeze and same400; structural bug repair separately if needed. No full400 parameter tuning.',
      'acceptance':['Visible fine-strand candidates suppressed while broad grooves retained on fixed examples',
        'No accepted curve crossing strong independently image-supported body barriers',
        'Do not force phase/period identity or fill unknown occlusion',
        'Keep400 denominator and all rejected proposals/reasons; no point-count accuracy claims'],
      'review':'Fixed6 component boxes from D2 diagnosis; extra12 existing QA indices after freeze; inspect stage masks and overlay, no labels.',
      'equipment_access':False,'reserved_access':False,'html':False,'ground_truth_accuracy':None})
    dump(HERE/'index_before.json',{str(p):sha(p) for p in [ROOT/'数据资产/README.md',HERE.parent/'README.md',ROOT/'工程导航/05_实验版本档案/README.md']})
if __name__=='__main__':initialize()
