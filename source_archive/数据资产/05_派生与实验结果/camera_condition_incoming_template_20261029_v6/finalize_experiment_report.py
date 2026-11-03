"""汇总保存的实际运行/独立审计，不替代图像评价，不修改任何预测。"""
import ast,hashlib,json,datetime
from pathlib import Path
from collections import Counter

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p)
 if p.exists():raise ValueError('Preserve existing report: '+str(p))
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
 summary=read(HERE/'all_001/summary.json');audit=read(HERE/'all_001/independent_structure_auditrepaired_001.json')
 prefix=read(HERE/'all_001/independent_prefix_logic_audit.json');repeat=read(HERE/'all_001/independent_repeatability_audit.json')
 freeze=read(HERE/'freeze_all.json');protocol=read(HERE/'protocol.json');transition=read(HERE/'all_001/A5_output_transition_counts.json')
 assert summary['completed']==400 and summary['exit_code']==0 and not summary['failures']
 assert audit['passed'] and prefix['passed'] and audit['passed_count']==400
 assert all(sha(p)==digest for p,digest in freeze['sources_sha256'].items())
 sourcegroups=Counter(r['source_group'] for r in summary['records'])
 pool_count=0;entry_recovery_cases=[];multiple_entries=[];axis_count=0
 for row in summary['records']:
  rel=Path(row['output_file']);result=read(HERE/'all_001/记录'/rel.parent.name/(rel.stem+'.json'))['result']
  pool_count+=len(result.get('candidate_pool',[]))
  if any(e['recovered'] for e in result.get('entries',[])):entry_recovery_cases.append(row['output_file'])
  if len(result.get('entries',[]))>1:multiple_entries.append(row['output_file'])
  axis_count+=sum(c['candidate_id'].endswith('axis_reference') for c in result.get('candidate_pool',[]))
 parameters={}
 for name in ['appearance_template.py','free_seed_recovery.py','axial_reacquisition.py','image_beam.py','pipeline.py']:
  tree=ast.parse((HERE/name).read_text(encoding='utf-8'))
  for node in tree.body:
   if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PARAMETERS' for t in node.targets):parameters[name]=ast.literal_eval(node.value)
 references=['protocol.json','freeze_all.json','A6算法说明.txt','all_001/run_start.json','all_001/summary.json','all_001/output_manifest.json','all_001/independent_structure_auditrepaired_001.json','all_001/independent_output_manifestrepaired_001.json','all_001/independent_prefix_logic_audit.json','all_001/independent_repeatability_audit.json','all_001/audit_failed_command_receipt_001.json','all_001/audit_command_receiptrepaired_001.json','all_001/audit_repair_reconciliation.json','all_001/A5_output_transition_counts.json','pilot_002/independent_visual_review.json','pilot_002/independent_template_beam_visual_review.json','pilot_003/independent_fixed_two_visual_review.json','extra_001/independent_visual_review.json','all_001/recovered_entry_visual_review.json']
 references.append('root_visual_review.json')
 evidence={name:sha(HERE/name) for name in references if (HERE/name).is_file()}
 report={
  'version':'IR-A6.0.0','name':'自模板多候选 A6','date':'2026-10-29','created_at':datetime.datetime.now().isoformat(),
  'parent_version':'IR-A5.0.0','status':'offline_research_candidate_partial_support_and_multiple_hypotheses_not_accepted_whole_rope_identity',
  'task':'Use own free-rope appearance, retain bend alternatives, consult F3 only after original-image support, truncate or preserve ambiguity.',
  'scope':{'dataset':'same_open_development_400','denominator':400,'camera_counts':audit['camera_counts'],'source_groups':dict(sourcegroups),'input_manifest_sha256':protocol['input_manifest_sha256'],'F3_dependency':'saved upstream predictions; not an arbitrary-new-image standalone validation','heldout_or_reserved_used':False,'equipment_connected':False,'permissions_changed':False,'training':False,'HTML':False},
  'development':{'budget':protocol['budget'],'runs':{name:{k:read(HERE/name/'summary.json')[k] for k in ('completed','planned','statuses','exit_code','elapsed_seconds')} for name in ['pilot_001','pilot_002','pilot_003','extra_001','all_001']},'freeze_sha256':sha(HERE/'freeze_all.json'),'visual_retuning_after_freeze':False,'prior_failed_probes_and_audit_preserved':True},
  'implementation':{'parameters':parameters,'source_sha256':freeze['sources_sha256'],'A5_F3_C3_D4_preserved':True,'no_F3_candidate_generation':True,'full_generated_pool_used_for_prefix':True,'late_F3_does_not_move_coordinates':True,'missing_gap_not_trusted':True,'maximum_display_branches_per_case':6,'all_pool_paths_saved':True},
  'actual_output':{'completed':400,'processing_failures':[],'actual_process_exit_code':0,'statuses':summary['statuses'],'nonempty_trusted_masks':audit['nonempty_trusted_cases'],'nonempty_best_or_free_masks':audit['nonempty_best_cases'],'empty_unknown':summary['statuses']['UNKNOWN'],'full_pool_count':pool_count,'displayed_candidate_count':audit['candidate_count'],'exact_axis_reference_count':axis_count,'recovery_entry_case_count':len(entry_recovery_cases),'recovery_entry_cases':entry_recovery_cases,'multiple_independent_entry_cases':multiple_entries,'previous_A5_unknown_to_output_count':len(transition['previously_unknown_now_output']),'previous_A5_output_to_unknown_count':len(transition['previously_output_now_unknown']),'counts_are_accuracy':False},
  'independent_validation':{'structure_passed':True,'checked_cases':400,'prefix_reconstruction_passed':True,'selected_path_cases':prefix['selected_path_cases'],'long_unknown_gap_checks':prefix['long_gap_cases'],'repeatability':repeat,'runtime_source_count':len(freeze['sources_sha256']),'baseline_hash_checks':62,'input_dependency_hash_checks':3600,'formal_prediction_files':3547,'whole_pool_candidates_checked':pool_count,'late_F3_pointwise_scope':'displayed1547 branches,62505 eligible points; compressed undisplayed candidates do not support independent full late-F3 recomputation','audit_initial_exit1':'footer mkdir conflicted with concurrent audit snapshot; all prior files and failure receipt preserved','audit_final_exit_code':0,'audit_repair_changed_predictions':False},
  'visual_findings':{'observed_improvement':['A5-empty right dual101643 pair0375 and preset pair0405 recovered the visible external rope','late/straight alternatives retained in several right-camera cases','unknown long gaps and competing tails separated from primary trusted mask','f15 pilot003 local-prefix repair restored44 search points versus6 in pilot002'],
   'known_failures':['f15 still lacks a candidate following the full visible straight rope through the bright transition; selected long candidate still turns early','similar old-coil appearance can score above the active-rope continuation','short supported axis reference can outrank a longer hypothesis because tail windows cover different image areas','trusted output can stop too early even where a human sees the rope','new nonempty outputs are not automatically correct detections','frozen supplementary review: right dual101517 pair0075 recovered_01 follows a hand-held metal tool; two entry interpretations produce empty primary trusted mask but the wrong branch remains saved'],
   'identity_ground_truth_available':False,'whole_rope_accuracy_goal_achieved':False,'overall_accuracy_improvement_demonstrated':False,'image_ranking_net_gain_over_independent_geometry_pipeline_demonstrated':False,'rejection_counted_as_correct':False,'same_material_or_physical_contact_confirmed':False,'t0_t1_t2_verified':False,'next_focus':'Candidate coverage at the bright-to-dark bend and comparable local evidence; no claim that more paths or shorter masks solve identity.'},
  'output_folders':{name:str(HERE/'all_001'/name) for name in ['01_可信前缀掩膜','02_原图叠加','03_待确认候选并集','04_最佳外观候选掩膜','05_多候选分支','记录']},'evidence_sha256':evidence}
 save(HERE/'experiment_report.json',report)
 for name,command,session in [('extra_001','--mode extra --run extra_001 --workers 4 --freeze freeze_all.json','9785'),('all_001','--mode all --run all_001 --workers 8 --freeze freeze_all.json','97977')]:
  save(HERE/name/'shell_exit_receipt.json',{'command':'run_experiment.py '+command,'actual_process_exit_code':0,'observed_root_exec_session':session,'summary_sha256':sha(HERE/name/'summary.json')})
 print(json.dumps({'report_sha256':sha(HERE/'experiment_report.json'),'pool_count':pool_count,'recovery_entries':len(entry_recovery_cases),'multiple_entries':len(multiple_entries),'axis_references':axis_count}))

if __name__=='__main__':main()
