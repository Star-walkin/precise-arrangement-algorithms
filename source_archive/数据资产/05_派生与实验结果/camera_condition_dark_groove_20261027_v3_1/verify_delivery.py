"""交付末次只读核验与补充目视证据，不覆盖历史报告。"""
from pathlib import Path
from data_io import HERE,ROOT,read_json,dump,sha,sources
run=HERE/'all_001'
assert sources()==read_json(HERE/'freeze_all.json')['source_sha256']
for p,h in read_json(HERE/'protocol.json')['frozen_d2_source_sha256'].items():assert sha(p)==h
audit=read_json(run/'independent_audit.json');assert audit['passed']==400 and not audit['global_errors']
assert [len(list((run/'图像结果'/role).glob('*.png'))) for role in ['left_camera','right_camera']]==[200,200]
index=read_json(run/'记录/qa_additional/index.json')
independent=read_json(run/'qa_visual_additional.json');root=read_json(run/'qa_visual_root_final.json')
seen={n for c in independent['cases'] for n in c.get('new_panels_viewed',[])}
seen.update(Path(c['comparison']).name for c in root['entries'] if c['review_status']=='actually_opened_by_root_after_patch')
unchanged={x['output_file'] for x in read_json(HERE/'parent_prediction_compare.json')['entries'] if x['in_fixed_additional_12'] and x['identical']}
covered=[e for e in index['entries'] if Path(e['comparison']).name in seen or e['output_file'] in unchanged]
assert len(covered)==len(index['entries'])==26
indices=[ROOT/'数据资产/README.md',HERE.parent/'README.md',ROOT/'工程导航/05_实验版本档案/README.md']
for p in indices:assert p.read_text(encoding='utf8').count('## 暗沟约束 D3 / DR-D3.0.1')==1
target=HERE/'delivery_verification.json'
if target.exists():raise FileExistsError(target)
dump(target,{'version':'DR-D3.0.1','images':{'left':200,'right':200},'structure_audit_passed':400,
 'frozen_sources_unchanged':True,'C3_D2_sources_unchanged':True,
 'additional_QA_components_covered_by_actual_new_views_or_exact_byte_inheritance':26,
 'root_visual_addendum_sha256':sha(run/'qa_visual_root_final.json'),
 'independent_visual_sha256':sha(run/'qa_visual_additional.json'),
 'audit_sha256':sha(run/'independent_audit.json'),'output_manifest_sha256':sha(run/'output_manifest.json'),
 'report_sha256':sha(HERE/'experiment_report.json'),
 'post_receipt_helper_sha256':{n:sha(HERE/n) for n in ['save_root_visual_review.py','verify_delivery.py']},
 'index_sha256':{str(p):sha(p) for p in indices},
 'clarification':'Independent QA left3 changed components unviewed; root actually opened those3. Byte comparison filename is parent_prediction_compare.json. Visual evidence remains mixed, no truth accuracy claimed.'})
print({'images':400,'audit_passed':400,'qa_components_accounted':26,'sources_unchanged':True,'receipt_sha256':sha(target)})
