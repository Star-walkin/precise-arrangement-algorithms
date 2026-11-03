"""在实际全量运行和独立审计之后登记交付，不改预测或旧记录。"""
from pathlib import Path
from collections import Counter
import json,csv,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=HERE/'all_002'
OLD=HERE.parent/'camera_condition_whole_incoming_20261028_v3'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 assert not (HERE/'experiment_report.json').exists(), 'Never overwrite delivery'
 summary=load(RUN/'summary.json');audit=load(RUN/'structure_audit.json')
 assert summary['completed']==400 and summary['exit_code']==0 and not summary['failures']
 assert audit['passed'] and audit['checked_case_count']==400
 entry_audit=load(RUN/'candidate_surface_entry_audit.json');assert entry_audit['passed']
 equivalence=load(RUN/'final_repair_equivalence.json');assert equivalence['png_equivalence']['passed']
 frozen=load(HERE/'freeze_all_002.json')
 assert all(sha(n)==h for n,h in frozen['sources_sha256'].items())
 stats=Counter();raw=Counter();unknown=Counter();rows=[];changes=[]
 for rec in summary['records']:
  rel=Path(rec['output_file']);saved=load(RUN/'记录'/rel.parent.name/(rel.stem+'.json'))
  old=load(OLD/'all_001/记录'/rel.parent.name/(rel.stem+'.json'))
  whole=saved['whole'];occ=saved['occlusion'];free=saved['free'];st=whole['status']
  stats['raw_probes']+=len(occ.get('rows',[]))
  stats['independent_visual_rows']+=len(set(r.get('visual_row_id',r['row_id']) for r in occ.get('rows',[])))
  stats['raw_transition_chains']+=sum(bool(r.get('interruption_then_continuity_chain')) for r in occ.get('rows',[]))
  stats['restored_before_old_endpoint_filter']+=occ.get('diagnostics',{}).get('retained_before_old_lower',0)
  stats['selected_raw_transition_chains']+=bool(whole.get('diagnostics',{}).get('selected_raw_transition_chain'))
  for r in occ.get('rows',[]):raw[r['original_evidence_state']]+=1;unknown.update(r.get('unknown_reasons',[]))
  chosen=next((c for c in whole.get('candidates',[]) if c['candidate_id']==whole.get('selected_candidate_id')),None)
  oldchosen=next((c for c in old['whole'].get('candidates',[]) if c['candidate_id']==old['whole'].get('selected_candidate_id')),None)
  equal=sha(RUN/'01_整根候选掩膜'/rel.parent.name/rel.name)==sha(OLD/'all_001/01_整根候选掩膜'/rel.parent.name/rel.name)
  if not equal:changes.append({'output_file':str(rel),'old_status':old['whole']['status'],'new_status':st,'old_length':oldchosen.get('length_px') if oldchosen else None,'new_length':chosen.get('length_px') if chosen else None})
  rows.append({'output_file':str(rel),'camera_role':rec['camera_role'],'source_group':rec['source_group'],'status':st,'free_reason':free.get('diagnostics',{}).get('reason'),'candidate_count':len(whole.get('candidates',[])),'selected_length_px':chosen.get('length_px') if chosen else '', 'candidate_not_verified_same_rope':True,'mask_equal_A3':equal})
 # Exact frozen-pilot outputs must be reproduced in the full run.
 equivalent=[]
 for rec in load(HERE/'pilot_003/summary.json')['records']:
  rel=Path(rec['output_file'])
  for group in ['01_整根候选掩膜','02_原图叠加','03_实测支持掩膜','04_接续待确认']:
   a=HERE/'pilot_003'/group/rel.parent.name/rel.name;b=RUN/group/rel.parent.name/rel.name
   equivalent.append({'path':str(b.relative_to(RUN)),'pilot_sha256':sha(a),'all_sha256':sha(b),'equal':sha(a)==sha(b)})
 assert all(r['equal'] for r in equivalent)
 dump(RUN/'pilot_equivalence.json',{'compared_pngs':64,'all_equal':True,'files':equivalent})
 with (RUN/'case_index.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 statuses=summary['statuses'];whole_count=sum(v for k,v in statuses.items() if 'WHOLE_APPEARANCE' in k);partial=sum(v for k,v in statuses.items() if 'PARTIAL_APPEARANCE' in k);free_only=sum(v for k,v in statuses.items() if k.startswith('FREE_ONLY'))
 report={'version':'IR-A4.0.1','name':'断界寻绳 A4','date':'2026-10-29','started_local_date':'2026-10-28','baseline':'F3 / RB-F3.0.0 user-confirmed qualitative400 baseline, previously archived; C3/D4/F3 unchanged','input_scope':'same explicit open400 development images, 200 left + 200 right; original4096x3000, no heldout/truth input',
  'actual_run':{'completed':400,'failed':0,'exit_code':0,'workers':8,'elapsed_seconds':summary['elapsed_seconds'],'statuses':statuses,'nonempty':audit['nonempty_mask_cases'],'empty':audit['empty_mask_cases'],'whole_extent_appearance_candidates':whole_count,'partial_candidates':partial,'free_only':free_only},
  'changes':['Significant-component search domain removes isolated SAM fragments from entry/exit state; original S unchanged.', 'Original-image boundary probes start at first stable search-surface entry, not only the end of straight extension.', 'Separate crossing/near-tangent/multiple probes; explicit raw domain availability; unknown image evidence earns no connection bonus.', 'All curved continuations remain review-marked even with local raw support; no same-material identity claim.'],
  'observed_pilot_effects':['ruler_right05 saved continuation36.61px to654.52px,12984new mask pixels; native original review supports restored visible span, dark identity remains ambiguous.', 'Second stage probes162 to463, raw evidence1continuous to7continuous+6interrupted; no independent interruption-then-continuity chain.', 'Second stage changes one of16 selected paths by17.99px at endpoint; no verified identity gain, two leading paths separated by only0.0017 score.'],
  'full_raw_evidence':{'counts':dict(stats),'states':dict(raw),'unknown_reasons':dict(unknown)},'changed_whole_masks_vs_A3':len(changes),'changes_vs_A3':changes,
  'validation':{'structure400_passed':True,'pilot_full_png_equal':64,'including_posthoc_repair_png_equal':80,'all_saved_candidates_without_surface_entry':entry_audit['all_candidates_without_surface_entry'],'entry_audit_sha256':sha(RUN/'candidate_surface_entry_audit.json'),'accuracy':None,'same_material_identity_verified':False,'audit_sha256':sha(RUN/'structure_audit.json'),'output_manifest_sha256':sha(RUN/'output_manifest.json')},
  'limitations':['Not a solved whole-rope detector. Some paths end before visible rope ends; dark same-rope vs neighbor remains unresolved.', 'Cached A1/A2 automatic free seeds remain dependencies; empty output does not prove no rope.', 'No confirmed n-1 identity rule, physical contact, millimeter diameter or t0/t1/t2.', 'This version depends on saved F3/A1/A2 predictions and is not an arbitrary-new-image standalone pipeline.'],
  'original_data_root':str(HERE.parent/'camera_condition_images_20261009_v2'),'source_freeze':str(HERE/'freeze_all_002.json'),'logs':str(RUN/'分片记录'),'predictions':str(RUN),'equipment_connected':False,'reserved_data_used':False,'HTML_created':False,'preserved_failed_full_run':str(HERE/'all_001/qa_postrun/independent_visual_review.json'),'implementation_repair':'After all001 revealed a never-enter background tail, reject saved curved candidates that never enter significant search domain; preserve measured free/straight masks. Fixed16 pilot003 plus4 posthoc regression cases before second freeze/all002.'}
 dump(HERE/'experiment_report.json',report)
 note=f'''断界寻绳 A4 / IR-A4.0.1（2026-10-29）

请先看 all_002/02_原图叠加，左右相机各200张，原图分辨率4096×3000。
绿色：局部原图有支持的外观；橙色：弱证据、弯曲接续或近分竞争，需要复核。
01_整根候选掩膜：背景0、候选255；可能整段、部分或为空，不是人工确认真值。
03_实测支持掩膜：局部双边/原图测量支持，不等于材料身份。
04_接续待确认：不确定部分，可与实测支持重叠。

实际400/400处理及独立结构审计通过；{audit['nonempty_mask_cases']}张非空，{audit['empty_mask_cases']}张为空。
{whole_count}张有到达当前可见域边侧的外观候选，{partial}张部分路径，{free_only}张仅自由段；这些不是识别成功率。
与A3相比{len(changes)}张候选掩膜变化。固定16图的64个最终PNG与全量结果逐字节一致。

F3基准已经封存于工程导航/05_实验版本档案/卷绳边界_全卷面基准_F3_20261028，原代码和结果不改。
详细方法、失败、来源哈希与状态见算法说明.txt、experiment_report.json、freeze_all_002.json及all_002/记录。
尚未实现可靠整根同根识别，不能把侧边外观终止称为真实物理消失点。未连接设备、未访问保留集。
'''
 (HERE/'结果说明.txt').write_text(note,encoding='utf-8')
 (HERE/'算法说明.txt').write_text('''断界寻绳 A4 / IR-A4.0.1

1. 固定基准。沿用用户认可并已封存的全卷面F3；C3/D4/F3原源码和保存预测不改。此次读取开放400原图及已保存上游，不是对任意新图重新从零运行的独立流水线。
2. 自由段。复用A1/A2的自动入口，在原图的卷面外区域重新测双侧外边缘、中心方向和像素宽度，尽量避免把一股细丝当整根绳。当前没有缓存种子的图片仍拒识，不把拒识当作“没有绳”。
3. 股间阴影。量测自由绳内部的纹理方向，再随路径切线旋转比较。沿绳方向平均以压制窄股缝，同时用法向双侧对比约束整绳宽度；明条、暗条两种极性均可。纹理相近只能说明外观相近，不能证明同根。
4. 搜索域修复。原SAM卷面的小碎片可能先碰到直绳，旧程序于是误判已进入卷面，再把正常自由空间判成离开卷面。本版只用显著连通组件及有面积上限的凸包决定搜索域；不把碎片OR回去，不修改SAM原图，也不把凸包当绳体观测。
5. 原图断界。从C3记录和F3保存的全卷面D4曲线提出参考位置，包含尚未进入F3最终融合的候选；正式F3掩膜用于诊断，不把其排除孔洞当成遮挡。真正的连续/中断必须重测未画线原图。采样从首次稳定进入主要搜索卷面处开始，同一参考的不同交会分开保存，避免只看直段末端和只保留第一个交会位置。近切和横跨区别记录，域外、缺样、低对比等保留具体未知原因。
6. 中断后恢复。按自由直线上的实际跨越顺序检查独立视觉行；连续两行原图中断后下一行恢复连续才记录提示。未知行打断这条链，同一行的多个探针不能虚增圈数。未知证据不再给接续加分。即使形成提示，也不是确认的n-1材料身份。
7. 连续绳带候选。沿自由段方向逐步搜索，结合固定光学宽度、原图双侧支持、股纹方向和有限近切区域；保留提前结束和邻圈竞争路径，禁止无限延伸。最终保留的接续路径必须实际进入显著搜索域，不能用已经裁掉的弱尾段冒充入域；全程未进入时只输出已复测的自由/直段。输出最高外观分的一条候选，弯段统一另标待确认，不把不同候选并成整根。
8. 闭环与边界。两轮固定16图：第一轮只修搜索域，第二轮修断界采样/未知加分。第一批400图事后复核发现never-enter背景尾，按预先声明的一次实现修复预算，增加最终保存候选必须实际入域的检查；固定16图及4张事后回归通过后另存all_002，all_001失败不覆盖。预测先保存，再与原图和旧版看图比较；原失败输出留存。冻结后运行400图，没有全量阈值调参；明确记录了全量事后发现并修复的实现漏洞。没有同根真值，实际检查只支持部分可见覆盖和结构改进，不支持完整整根准确率。

运行：使用项目现有 vision_prototype/.venv/Scripts/python.exe，run_batch.py --run 新目录名 --workers 8 --freeze-file freeze_all_002.json；随后 audit_a4.py --mode all --run 同目录名 --workers 4。新目录不得覆盖旧运行。源码与参数冻结在freeze_all_002.json，逐图输入哈希和候选坐标在all_002/记录，原始分片日志/退出码在all_002/分片记录与summary.json。
''',encoding='utf-8')
 # Existing navigation files only; append new evidence without rewriting history.
 changed=[]
 indices=[ROOT/'数据资产/README.md',ROOT/'数据资产/05_派生与实验结果/README.md',ROOT/'工程导航/05_实验版本档案/README.md']
 dest=(RUN/'02_原图叠加').as_posix();rep=(HERE/'experiment_report.json').as_posix()
 entry=f'\n\n## 2026-10-29：断界寻绳 A4 / IR-A4.0.1（候选，整根身份未验收）\n\n[400张原图叠加]({dest}) · [实际实验记录]({rep})。同一开放400图，左右各200；保留F3基准与A3历史不改。修复SAM碎片造成的提前停止，扩大原图断界采样到前部交会，区分近切与横跨并取消未知证据加分。两轮固定16图及一次入域漏洞回归后重新冻结；all_001背景尾失败保留，all_002全量400/400与独立结构审计通过；64个pilot PNG逐字节一致。{audit["nonempty_mask_cases"]}张非空、{audit["empty_mask_cases"]}张空，{whole_count}张到达当前可见域边侧的外观候选并非整根成功率。暗侧同根、n-1规则与物理接触仍未验收；橙色保留待确认。无设备/保留集访问，无HTML。\n'
 for path in indices:
  before=sha(path);content=path.read_text(encoding='utf-8');assert '断界寻绳 A4 / IR-A4.0.1' not in content
  path.write_text(content+entry,encoding='utf-8');changed.append({'path':str(path),'before':before,'after':sha(path)})
 ap=ROOT/'AGENTS.md';content=ap.read_text(encoding='utf-8');before=sha(ap)
 old='最新[断界寻绳A3离线400图实验](数据资产/05_派生与实验结果/camera_condition_whole_incoming_20261028_v3/experiment_report.json)'
 new='最新[断界寻绳A4离线400图实验](数据资产/05_派生与实验结果/camera_condition_whole_incoming_20261028_v4/experiment_report.json)'
 assert content.count(old)==1
 ap.write_text(content.replace(old,new).replace('维护日期：2026-10-28。','维护日期：2026-10-29。'),encoding='utf-8');changed.append({'path':str(ap),'before':before,'after':sha(ap),'scope':'latest experiment pointer only; unverified identity boundary retained'})
 dump(HERE/'delivery_receipt.json',{'version':'IR-A4.0.1','new_source_files':['whole_io.py','free_rope.py','whole_path.py','boundary_occlusion.py','run_whole.py','run_batch.py'],'existing_navigation_changes':changed,'report_sha256':sha(HERE/'experiment_report.json'),'actual_run_exit_code':summary['exit_code'],'audit_passed':True,'source_unchanged':True,'no_equipment':True,'no_reserved_data':True})
 print(json.dumps(report['actual_run'],ensure_ascii=False));print('Delivery registered; historical predictions retained.')
if __name__=='__main__':main()
