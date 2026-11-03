"""如实记录主智能体最终实际开图范围；未重看的组件不冒充已看。"""
from pathlib import Path
from data_io import HERE,read_json,dump,sha
parent=HERE.parent/'camera_condition_dark_groove_20261027_v3'
run=HERE/'all_001';target=run/'qa_visual_root_final.json'
if target.exists():raise FileExistsError(target)
index=read_json(run/'记录/qa_additional/index.json')
comparison=read_json(HERE/'parent_prediction_compare.json')
unchanged={x['output_file'] for x in comparison['entries'] if x['in_fixed_additional_12'] and x['identical']}
seen={
 'L_run_stop_run_20260922_01__f000046__left__part0.jpg':'旧版块区台阶消失，主体曲线平顺；暗块内真实沟槽仍不可确认。',
 'dual_20260923_101517__pair0315__left__part0.jpg':'中部长沟槽曲率更自然，顶排有仅黄点和橙线；未出现首轮同步锯齿。',
 'dual_20260923_101517__pair0315__right__part1.jpg':'普通青线覆盖明显减少，多条原青线变橙色疑义线；不能称全面改善。',
 'dual_20260924_111500__pair0015__left__part0.jpg':'可见中下长沟槽更顺，左下台阶减少；深暗/遮挡处仍有短线和缺线。',
 'dual_20260924_111500__pair0015__right__part0.jpg':'中下部曲线平顺，旧右缘竖直折角减弱；上部多橙紫线，未确认身份。',
 'R_run_stop_run_20260922_01__f000026__right__part0.jpg':'白活动绳遮挡保留空缺，右下块区台阶减少；部分弱证据短线仍在。',
 'ruler_left_04__part0.jpg':'左下旧台阶被平滑曲线替代，主体排列有序；上缘与暗边仍有疑义。',
 'R_run_stop_run_20260922_01__f000026__right__part1.jpg':'补充实际查看次组件；低光区域保留较大空缺，不把平滑候选等同正确。',
 'ruler_left_04__part1.jpg':'补充实际查看次组件；按增强图检查曲线和未覆盖区域，仍保留疑义说明。'}
entries=[]
for item in index['entries']:
    name=Path(item['comparison']).name
    if name in seen:
        status='actually_opened_by_root_after_patch';note=seen[name]
    elif item['output_file'] in unchanged:
        status='inherited_exact_png_equality_from_parent_review';note='原独立记录已经实际查看此组件，新整张PNG字节完全相同。'
    else:
        status='not_reopened_after_patch';note='父版本已看；本次只重看该图的指定组件，不能把此组件称为重新目视通过。'
    entries.append({**item,'review_status':status,'observation':note})
repair=read_json(run/'记录/qa_repair/index.json')
dump(target,{'version':'DR-D3.0.1','reviewer':'root actual image viewing',
 'method':'Actual selected-component review for all7 changed QA images; exact-byte inheritance for5 unchanged QA images. Other changed secondary components explicitly not reopened.',
 'fixed_images':12,'actual_reopened_components':9,'inherited_unchanged_images':5,
 'combination_note':'Combine with independent qa_visual_additional.json: its3 unviewed changed components have now been actually opened by root. Correct byte-comparison filename is parent_prediction_compare.json, not compare_parent.json.',
 'entries':entries,'additional_actual_repair_views':repair['entries'],
 'repair_visual_notes':'右图近距候选被删减，左图橙线稍移开；两处都弱曝光，间距修复不代表真值已确认。',
 'parent_all26_component_review':{'path':str(parent/'all_001/qa_visual_additional.json'),'sha256':sha(parent/'all_001/qa_visual_additional.json')},
 'overall':'Visible-groove smoothness and edge curvature improved in several examples, but coverage is mixed and ordinary-cyan coverage regresses locally. Extreme dark blocks and ambiguity remain; not uniformly better thanD2.',
 'ground_truth_accuracy':None,'physical_identity_verified':False})
print({'actual_new_components':9,'unchanged_images_inherited':5,'repair_views':2,'sha256':sha(target)})
