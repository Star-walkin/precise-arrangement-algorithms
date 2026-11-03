"""固定400补丁前后文件对照；用字节相同确认已有独立目视证据是否适用。"""
from pathlib import Path
from data_io import HERE,selection,read_json,dump,sha
parent=HERE.parent/'camera_condition_dark_groove_20261027_v3'
target=HERE/'parent_prediction_compare.json'
if target.exists():raise FileExistsError(target)
assert read_json(HERE/'all_001/summary.json')['completed']==400
rows,_=selection('all')
qa=read_json(parent/'all_001/记录/qa_additional/index.json')['selection']
entries=[]
for row in rows:
    rel=Path(row['output_file']);p=Path('图像结果')/rel.parent.name/rel.name
    before,after=sha(parent/'all_001'/p),sha(HERE/'all_001'/p)
    entries.append({'output_file':row['output_file'],'parent_png_sha256':before,
        'patch_png_sha256':after,'identical':before==after,'in_fixed_additional_12':row['output_file'] in qa})
changed=[c for c in entries if not c['identical']]
dump(target,{'parent':str(parent),'patch_version':'DR-D3.0.1','planned':400,
    'unchanged_image_count':400-len(changed),'changed_image_count':len(changed),
    'changed_fixed_additional_images':[c['output_file'] for c in changed if c['in_fixed_additional_12']],
    'entries':entries,'meaning':'Byte comparison is reproduction evidence, not visual accuracy.'})
print({'unchanged_images':400-len(changed),'changed_images':len(changed),
       'changed_additional_12':[c['output_file'] for c in changed if c['in_fixed_additional_12']]})
