"""只汇总实际保存的最终运行；不把输出覆盖数写成准确率。"""
import sys
from collections import Counter
import whole_io as io

def main():
 run=io.HERE/'all_001';summary=io.read_json(run/'summary.json');audit=io.read_json(run/'independent_structure_audit.json')
 assert summary['completed']==400 and summary['exit_code']==0 and audit['passed']
 equivalent=[]
 for small in ['pilot_003','extra_001']:
  saved=io.read_json(io.HERE/small/'output_manifest.json')
  for entry in saved['files']:
   if not entry['path'].endswith('.png'):continue
   path=run/entry['path'];equivalent.append({'small_run':small,'path':entry['path'],'sha256':io.sha(path),'reference_sha256':entry['sha256'],'equal':io.sha(path)==entry['sha256']})
 assert equivalent and all(r['equal'] for r in equivalent)
 io.dump(run/'small_full_equivalence.json',{'png_count':len(equivalent),'passed':True,'checks':equivalent})
 previous=io.Path(io.read_json(io.HERE/'protocol.json')['A4_output_root'])
 changed=[];selected=[]
 for row in io.selection('all'):
  rel=io.Path(row['output_file']);path=io.Path('01_整根候选掩膜')/rel.parent.name/rel.name
  if io.sha(run/path)!=io.sha(previous/path):changed.append(row['output_file'])
  record=io.read_json(run/'记录'/rel.parent.name/(rel.stem+'.json'));whole=record['whole']
  selected.append({'input':row['output_file'],'camera_role':row['camera_role'],'status':whole['status'],'selected_candidate_id':whole.get('selected_candidate_id'),'g0_selected_candidate_id':whole.get('g0_selected_candidate_id'),'same_material_identity_confirmed':False})
 receipts=[]
 for child in ['pilot_001','pilot_002','pilot_003','extra_001','all_001']:
  p=io.HERE/child/'summary.json';j=io.read_json(p)
  receipts.append({'run':child,'completed':j['completed'],'planned':j['planned'],'exit_code':j['exit_code'],'elapsed_seconds':j['elapsed_seconds'],'statuses':j['statuses'],'summary_sha256':io.sha(p),'sources_unchanged':j['sources_unchanged']})
 report={'version':io.VERSION,'name':'差分轨道 A5','date':'2026-10-29','current_run':'all_001','evidence_level':'open-development automatic predictions and qualitative review; not truth accuracy',
  'fixed_denominator':400,'camera_counts':{'left':200,'right':200},'processing':summary['completed'],'run_exit_code':summary['exit_code'],'elapsed_seconds':summary['elapsed_seconds'],
  'nonempty_mask_cases':audit['nonempty_mask_cases'],'empty_or_unknown_cases':audit['empty_mask_cases'],'statuses':summary['statuses'],
  'status_counts_are_not_accuracy':True,'verified_same_material_identity_cases':None,'ground_truth_available':False,
  'n_minus_one_used':False,'F3_baseline_unchanged':all(c['ok'] for c in audit['baseline_source_checks']),
  'new_method':['whole-width original gray contrast and two flanks; bright/dark polarity retained','separate saturated body outer-edge support; clipped means not used as relative background certainty','F3 formal adjacent boundary pairs with finite image-supported transitions','attached stage uses two original-image dark grooves, not free-rope step edges','local formal curvature and smooth transition consistency, retain competing alternatives'],
  'runs':receipts,'structure_audit':{'path':str(run/'independent_structure_audit.json'),'sha256':io.sha(run/'independent_structure_audit.json'),'passed':audit['passed'],'checked_count':audit['checked_count']},
  'small_full_equivalence':{'count':len(equivalent),'passed':True,'path':str(run/'small_full_equivalence.json')},
  'A4_mask_changes':{'count':len(changed),'meaning':'byte difference only, neither improvements nor accuracy','inputs':changed},
  'selected_routes':selected,'outputs':{k:str(run/k) for k in ['01_整根候选掩膜','02_原图叠加','03_图像支持','04_接续待确认']},
  'limitations':['All curved connections remain appearance candidates, not proven same material.','Missing original free-entry predictions remain unknown; empty output is not rope absence.','Local image support may belong to neighboring coils; no measured whole-rope accuracy.','Some original SAM/F3 dark tails are missing; no unsupported unlimited completion.','Optical width can include cast shadow, especially when free and attached widths differ.','No physical contact, t0/t1/t2, 3D or hardware-control acceptance.'],
  'reserved_data_accessed':False,'equipment_connected':False,'permissions_changed':False,'HTML_generated':False}
 io.dump(io.HERE/'experiment_report.json',report)
 text=f'''差分轨道 A5 / {io.VERSION}，2026-10-29\n\n最终运行：all_001。实际处理400/400，左、右各200；退出码0。独立结构审计{audit['passed_count']}/400通过。\n非空掩膜{audit['nonempty_mask_cases']}张，空/未知{audit['empty_mask_cases']}张。这是输出数量，不是准确率，也不能将空输出当作没有活动绳。\n固定20图三轮开发、冻结后8图复核与全量保存；{len(equivalent)}个小样本/全量PNG逐字节相同。\n\n02_原图叠加：直接查看400张结果。绿色代表局部原图外观支持，橙色代表接续推断或竞争；即使绿色也不等于同根材料已证明。\n01_整根候选掩膜：单通道0/255（背景0，候选绳体255），包括已选有限接续；不是已验收标签。\n03_图像支持与04_接续待确认：分别保留图像支持和需要确认的部分，二者可能重叠（局部像绳但同根仍未知）。\n\n本版取消n−1作为判定依据，比较整绳灰度/阴影与两侧背景，并在贴轨阶段检查原图双暗沟和F3正式相邻边界。暗侧邻圈竞争、尾段缺失和阴影混入仍存在；尚不能宣布整根识别已可靠达标。\n源码、参数、逐图原始输入哈希、来源相机/工况、竞争路径、退出码、失败及修订证据见protocol.json、各run/记录、source_snapshot和experiment_report.json。原始图像、F3/C3/D4/A4基准保持不变；无设备、保留集和权限修改，无HTML。\n'''
 (io.HERE/'结果说明.txt').write_text(text,encoding='utf-8')
 print({'completed':400,'nonempty':audit['nonempty_mask_cases'],'equivalent_png':len(equivalent),'A4_mask_changed':len(changed)})
 return 0
if __name__=='__main__':sys.exit(main())
