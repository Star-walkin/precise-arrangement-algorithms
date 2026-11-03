"""汇总A7实际运行、源码和独立图像复核证据，不产生识别准确率。"""
import ast,datetime,hashlib,json
from pathlib import Path
from collections import Counter

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump_new(p,value):
 p=Path(p)
 if p.exists():raise ValueError('Preserve previous report: '+str(p))
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
 freeze=read(HERE/'freeze_all.json');protocol=read(HERE/'protocol.json');summary=read(HERE/'all_001/summary.json')
 audit=read(HERE/'all_001/independent_structure_audit.json');repeat=read(HERE/'all_001/independent_repeatability_audit.json')
 assert summary['completed']==400 and summary['exit_code']==0 and not summary['failures']
 assert all(sha(p)==digest for p,digest in freeze['sources_sha256'].items())
 source_parameters={}
 for name in ['straight_detector.py','strand_signature.py','free_rope.py','free_seed_recovery.py']:
  for node in ast.parse((HERE/name).read_text(encoding='utf-8')).body:
   if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PARAMETERS' for t in node.targets):source_parameters[name]=ast.literal_eval(node.value)
 refs=['protocol.json','freeze_all.json','A7算法与结果说明.txt','股纹重复性模块说明.txt','strand_diagnosis_003/report.json','pilot_001/independent_visual_review.json','pilot_002/independent_four_case_visual_review.json','pilot_003/pilot002_mask_change_receipt.json','extra_001/independent_final_visual_review.json','all_001/run_start.json','all_001/summary.json','all_001/output_manifest.json','all_001/independent_structure_audit.json','all_001/independent_output_manifest.json','all_001/independent_repeatability_audit.json','all_001/shell_exit_receipt.json','all_001/audit_command_receipt.json','all_001/repeatability_audit_command_receipt.json']
 assert all((HERE/p).exists() for p in refs)
 counts_by_camera={camera:dict(Counter(r['status'] for r in summary['records'] if r['camera_role']==camera)) for camera in ['left','right']}
 report={
  'version':'IR-A7.0.0','name':'直绳双边 A7','date':'2026-10-30','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'status':'offline_straight_segment_research_candidate_user_visual_acceptance_pending',
  'requested_scope':'Unwound visible straight rope only; one original-left overlay-right image per input. No bent-tail reconstruction.',
  'data':{'dataset':'same_open_development400','input_manifest_sha256':protocol['input_manifest_sha256'],'camera_counts':summary['camera_counts'],'source_group_counts':dict(Counter(r['source_group'] for r in summary['records'])),'reserved_decoded':False,'S1_labels_read':False,'training':False,'new_blind_test_claim':False,'manual_geometry_input':False,'input_dependencies':['original PNG','C4 surface mask','A1 AUTO free seed','A2 AUTO free seed','C3 saved pixel pitch only'],'F3_pixels_or_curves_used':False},
  'changes':{'new_implementation':['straight_detector.py','strand_signature.py','a7_io.py','run_experiment.py'],'legacy_byte_copies':['free_rope.py','free_seed_recovery.py'],'external_legacy_dependency':'camera_condition_incoming_rope_20261026_v2/free_refinement.py','legacy_versions_unchanged':True,'principles':['undirected axis does not require winding direction','raw paired outer edges and robust straight axis','local repetitive strand signature distinct from generic diagonal scratches','surface is not a hard straight-segment cutoff','joint remeasurement of compatible collinear fragments','bounded visible pieces; no filling long occlusion; local original texture required for reacquisition'],'parameters':source_parameters,'source_sha256':freeze['sources_sha256']},
  'experiments':{'fixed_pilot_denominator':16,'extra_denominator':8,'full_denominator':400,'planned_budget':protocol['budget'],'runs':{run:{k:read(HERE/run/'summary.json')[k] for k in ['planned','completed','statuses','exit_code','elapsed_seconds']} for run in ['pilot_001','pilot_002','pilot_003','extra_001','all_001']},'freeze_sha256':sha(HERE/'freeze_all.json'),'retuned_after_freeze':False,'earlier_results_and_mechanism_probes_preserved':True,'pilot2_to3_saved_mask_changes':read(HERE/'pilot_003/pilot002_mask_change_receipt.json')['changed_count']},
  'actual_output':{'completed':400,'processing_exceptions':0,'actual_process_exit_code':0,'status_counts':summary['statuses'],'status_by_camera':counts_by_camera,'detection_count_is_ground_truth_success':False,'UNKNOWN_means_absence':False,'pair_shape':[3000,8192,3],'source_shape':[3000,4096,3],'left_half_original_pixels_preserved':True,'mask_values':[0,255],'formal_file_count':1200,'HTML':False},
  'independent_validation':{'actual_exit_code':0,'structure_checked_cases':400,'runtime_dependencies_checked':8,'input_file_hashes_checked':2000,'candidate_axes_checked':366,'mixed_background_points_checked':8044,'reacquired_pieces_checked':115,'pilot_extra_all_equal_pngs':48,'audit_summary':{k:v for k,v in audit.items() if k not in ['cases','source_checks','baseline_source_checks','parent_manifest_checks']},'repeatability':repeat,'does_not_establish_visual_accuracy':True},
  'qualitative_observations':{'fixed_known_improvements':['handheld tool in right dual101517 pair0075 rejected while true external rope accepted','right ruler05 and Rstop2521 duplicate collinear candidates combined and visually aligned','some strand-induced fragmentation reduced by tangent aggregation','frozen Rrun55 review shows visible segments on both sides of guide, without filling central long occlusion'],
   'remaining_failures':['left f15 still stops before much of visible bright-overlap straight rope','right375 and0075 still miss long visible portion on the other side of guide','some bright/dark transitions and drum overlap have missing pieces','Rrun55 reappearance endpoint slightly extends into occluder','dark tail precise width and extent remain uncertain','36 ambiguous and107 unknown are retained; neither is counted correct'],
   'presence_or_full_extent_truth_labels_available':False,'all400_visually_reviewed':False,'complete_visible_straight_segment_coverage_verified':False,'accuracy_percentage_reported':None,'overall_accuracy_gain_over_A6_demonstrated':False,'physical_contact_confirmed':False,'whole_same_rope_bend_verified':False,'t0_t1_t2_verified':False},
  'delivery_folders':{name:str(HERE/'all_001/图像结果'/name) for name in ['01_已检出','02_待确认','03_未检出']},
  'raw_and_log_locations':{'raw_images':str(HERE.parent/'camera_condition_images_20261009_v2/images'),'protocol':str(HERE/'protocol.json'),'logs':str(HERE/'all_001/run.log'),'records':str(HERE/'all_001/记录'),'masks':str(HERE/'all_001/masks')},
  'equipment_connected':False,'permission_changes':False,'evidence_sha256':{p:sha(HERE/p) for p in refs}}
 dump_new(HERE/'experiment_report.json',report)
 receipt_path=HERE/'documentation_change_receipt.json';receipt=read(receipt_path);receipt['after']={p:sha(p) for p in receipt['before']};receipt['experiment_report_sha256']=sha(HERE/'experiment_report.json')
 receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 dump_new(HERE/'delivery_file_hashes.json',{'version':'IR-A7.0.0','new_version_top_level_files':{p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.json','.txt']},'updated_indexes':receipt})
 print(json.dumps({'report_sha256':sha(HERE/'experiment_report.json'),'counts':summary['statuses'],'camera_statuses':counts_by_camera,'delivery_files_verified':True}))

if __name__=='__main__':main()
