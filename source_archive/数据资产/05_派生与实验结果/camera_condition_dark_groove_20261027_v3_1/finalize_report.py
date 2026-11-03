"""只在固定400预测与独立审计完成后登记；不修改算法或旧产物。"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json
from data_io import HERE,ROOT,D2,selection,sources,sha,read_json,dump

run=HERE/'all_001'
parent=HERE.parent/'camera_condition_dark_groove_20261027_v3'
target=HERE/'experiment_report.json'
if target.exists():raise FileExistsError(target)
summary=read_json(run/'summary.json');audit=read_json(run/'independent_audit.json')
assert summary['completed']==400 and summary['failed']==0 and summary['source_unchanged']
assert audit['passed']==400 and audit['failed']==0 and not audit['global_errors']
assert sources()==read_json(HERE/'freeze_all.json')['source_sha256']
rows,_=selection('all')
qualities=Counter();shape=Counter();assigned=unassigned=reassigned=point_count=0
no_curves=[];pseudo=0;observation_loss=0.;hits=0
for row in rows:
    rel=Path(row['output_file'])
    case=read_json(run/'记录/cases'/rel.parent.name/(rel.stem+'.json'))
    d=case['record']['path_diagnostics']
    point_count+=case['record']['observed_point_count']
    assigned+=d.get('assigned_count',0);unassigned+=d.get('unassigned_count',0)
    reassigned+=d.get('reassigned_c3_group_count',0)
    qualities.update(c.get('quality','unknown') for c in case['fit']['curves'])
    for p in d.get('path_diagnostics',[]):
        if 'path_id' not in p:continue
        shape.update([p.get('reverse_bend_shape_guard',{}).get('status','not_reported')])
        pseudo+=p.get('coherent_image_pseudo_count',0)
    if not case['fit']['curves']:
        no_curves.append({'output_file':row['output_file'],'reason':d.get('reason',d.get('status'))})
    metric=d.get('yellow_full_denominator',{})
    observation_loss+=(metric.get('capped_pitch_loss_mean') or 0)*case['record']['observed_point_count']
    hits+=metric.get('hit_within_quarter_pitch_count',0)
assert point_count==1961902 and assigned+unassigned==point_count
evidence=[HERE/'freeze_all.json',run/'summary.json',run/'run_context.json',run/'run_log.jsonl',
          run/'output_manifest.json',run/'independent_audit.json',
          parent/'pilot_003/qa_visual_final.json',parent/'control_geometry_001/independent_audit.json',run/'记录/qa_additional/index.json',
          run/'qa_visual_additional.json',parent/'ablation_comparison.json',
          parent/'synthetic_check_003.json',HERE/'算法说明.txt',HERE/'repair_check.json',
          HERE/'parent_prediction_compare.json',parent/'all_001/independent_audit.json']
report={'version':'DR-D3.0.1','name':'暗沟约束 D3','date':'2026-10-27',
 'created_utc':datetime.now(timezone.utc).isoformat(),'status':'400 review outputs completed; not user acceptance',
 'selected_pilot':str(parent/'pilot_003'),'final_run':'all_001','summary':summary,
 'structural_patch':'Parent D3.0.0 full audit398/400 passed; two local-neighbor close-pair failures preserved. D3.0.1 repairs overlapping pair enumeration with unchanged fitting/image parameters.',
 'patch_pixel_comparison':{k:v for k,v in read_json(HERE/'parent_prediction_compare.json').items() if k!='entries'},
 'fixed_additional_visual_review':read_json(run/'qa_visual_additional.json'),
 'independent_structure_audit':{'passed':audit['passed'],'failed':audit['failed'],'global_errors':audit['global_errors']},
 'preservation':{'D2_yellow_points':point_count,'old_observed_lines_and_masks_exact':True,
    'D1_ROI_enhancement_and_C3_D2_sources_unchanged':True},
 'classification':{'assigned':assigned,'unassigned':unassigned,'changed_old_C3_group_id':reassigned,
    'interpretation':'visual assignment only, no identity ground truth'},
 'curve_quality_counts':dict(qualities),'shape_guard_path_counts':dict(shape),'image_soft_observation_count':pseudo,
 'no_curve_cases':no_curves,
 'yellow_consistency_not_accuracy':{'full_point_denominator':point_count,
    'within_quarter_pitch_count':hits,'mean_capped_pitch_distance':observation_loss/point_count,
    'note':'Existing yellow points are algorithm outputs, not truth labels. Missing predictions remain in denominator.'},
 'root_cause':'Old narrow search and strong smoothing can flatten visibly curved grooves even when yellow points are reasonable. Fixed6 review did not confirm widespread point-group mixing.',
 'algorithm':['Preserve original observations; multiscale raw bilateral-groove evidence with longer tangent integration',
    'Shared quartic family plus per-boundary cubic deviation; robust all-point fitting and two-round unique reassignment',
    'Small image soft corrections integrated over2.5pitch windows, not per-pixel greedy tracking',
    'Selective reverse-bend refit with observation-fit budget; ambiguity when constraints conflict',
    'Visible-ROI row ordering and spacing; short-gap inference only'],
 'visual_conclusion':'Fixed developmental examples show smoother, more appropriately bent visible dark-groove curves than D2; pilot001 fine-strand zigzags rejected. Final post-freeze review recorded separately; weak block textures and local offsets remain.',
 'ablation_conclusion':'Image soft evidence mainly makes small corrections. Shape improvement is principally due to the whole-curve model and constraints. Standalone image-guidance accuracy gain is not established.',
 'synthetic_limit':'Deliberately swapped local labels still leave4of5 curves; not a universal wrong-group recovery algorithm.',
 'ground_truth_accuracy':None,'physical_identity_or_contact_verified':False,
 'equipment_access':False,'reserved_access':False,'html_generated':False,
 'image_directory':str(run/'图像结果'),'source_sha256':sources(),
 'evidence_sha256':{str(p):sha(p) for p in evidence}}
dump(target,report)
links={ROOT/'数据资产/README.md':'05_派生与实验结果/',HERE.parent/'README.md':'',
       ROOT/'工程导航/05_实验版本档案/README.md':'../../数据资产/05_派生与实验结果/'}
text=('## 暗沟约束 D3 / DR-D3.0.1（2026-10-27）\n\n'
      '[400张图像目录]({prefix}camera_condition_dark_groove_20261027_v3_1/all_001/图像结果)：'
      '沿用同一开放400图（左右各200），D1欠曝光ROI/增强图及D2的1,961,902个黄点坐标全部保留，C3/D2代码不改。'
      '使用原图宽沟槽软证据、共享四次与逐边界三次曲线、唯一归属重拟合、反弯与行序约束。'
      '三轮固定12图开发及图像代价消融后冻结；首轮细股纹锯齿失败保留。D3.0.0全量398/400结构审计通过，两图局部邻线漏查失败保留；0.1只修复邻接检查后重跑。'
      '400/400运行及结构审计通过；固定样本可见侧缘形态改善，图像软代价主要作小幅校正，'
      '极暗块、周期歧义与局部偏位仍存在，不能把结构通过或黄点残差视作准确率。'
      '冻结后额外12图复查记录了改善、混合与普通青线覆盖退化案例（非准确率），详见同目录experiment_report.json；无边界案例按原分母保留。'
      '本版待用户复核，不替换已认可基准；未访问设备或保留集，不生成HTML。\n')
changed={}
for path,prefix in links.items():
    before=sha(path)
    with path.open('a',encoding='utf-8',newline='') as handle:handle.write('\n\n'+text.format(prefix=prefix))
    changed[str(path)]={'before_sha256':before,'after_sha256':sha(path),'operation':'append D3 evidence entry only'}
files={str(p.relative_to(HERE)):sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='delivery_receipt.json'}
dump(HERE/'delivery_receipt.json',{'no_git':True,'new_experiment_directory':str(HERE),'new_root_files_sha256':files,
   'index_changes':changed,'old_baselines_unchanged':True,'output_manifest_sha256':sha(run/'output_manifest.json')})
print({'completed':400,'yellow_points':point_count,'unassigned':unassigned,'no_curves':len(no_curves),
       'qualities':dict(qualities),'report_sha256':sha(target)})
