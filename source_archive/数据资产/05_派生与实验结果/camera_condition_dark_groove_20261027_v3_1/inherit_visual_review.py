"""仅当固定额外12图逐字节相同时，明确继承原人工/多模态检查记录。"""
from data_io import HERE,read_json,sha,dump
parent=HERE.parent/'camera_condition_dark_groove_20261027_v3'
target=HERE/'all_001/qa_visual_additional.json'
if target.exists():raise FileExistsError(target)
comparison=read_json(HERE/'parent_prediction_compare.json')
assert not comparison['changed_fixed_additional_images'], 'changed QA images require new visual review'
pairs=[x for x in comparison['entries'] if x['in_fixed_additional_12']]
assert len(pairs)==12 and all(x['identical'] for x in pairs)
previous=parent/'all_001/qa_visual_additional.json'
dump(target,{'version':'DR-D3.0.1','method':'Exact PNG-byte equality transfer of prior fixed12 actual visual review; not a claim of re-opening unchanged images',
    'fixed_input_count':12,'parent_review_path':str(previous),'parent_review_sha256':sha(previous),
    'unchanged_image_pairs':pairs,'parent_visual_review':read_json(previous),
    'changed_other_images':comparison['changed_image_count'],
    'ground_truth_accuracy':None,'meaning':'The same pixels support the same qualitative strengths and regressions. Changed non-QA cases require their own checks.'})
print({'inherited_visual_images':12,'report_sha256':sha(target)})
