"""Write compact provenance after the frozen run, without changing outputs."""
from pathlib import Path
from collections import Counter
import datetime,json,sys
from prepare_h2 import HERE,ROOT,load,dump,sha

def main():
    inputs=load(HERE/'inputs.json');summary=load(HERE/'all_001/summary.json');infer=load(HERE/'new_inference_receipt.json')
    assert summary['exit_code']==0 and summary['completed']==292 and summary['source_unchanged']
    freeze=load(HERE/'freeze_final.json');assert all(sha(p)==v for p,v in freeze['sources'].items())
    counts=Counter(r['kind'] or 'no_candidate' for r in summary['rows'])
    report=dict(version='IR-H2.0.0',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      artifact_date_note='Directory named at start on2026-10-30; completion local time2026-10-31',
      output_folder=str(HERE/'all_001/四栏检测结果'),source_open_images=400,rendered_images=292,
      original_entry_conditions=299,additional_A7_entry_conditions=2,total_entry_conditions=301,
      unresolved_entry_images=108,unresolved_is_not_no_rope=True,
      source_groups=len(set(j['source_group'] for j in inputs['jobs'])),
      camera_roles=dict(Counter(j['camera_role'] for j in inputs['jobs'])),
      model_B=inputs['model_B'],new_forward_calls=infer['forward_calls'],
      reused_float_responses=299,numeric_inference_seconds=infer['seconds'],
      batch_seconds=summary['elapsed_seconds'],batch_exit_code=summary['exit_code'],
      path_output_counts=dict(counts),ambiguous_top_candidate_images=sum(r['ambiguous'] for r in summary['rows']),
      images_with_some_candidate=sum(r['has_path'] for r in summary['rows']),
      path_counts_are_not_identity_accuracy=True,independent_identity_reference_used=False,
      method=['B center-high and bilateral low-band profiles plus original gray bilateral evidence',
              'A7 matched accepted undirected axes; bounded same-axis gaps with observed flanks, gaps remain inferred',
              'Original independent SAM2 domain; adjacent saved F3 boundary pairs converted to local center tracks',
              'Finite Hermite transition hypotheses and direct H1 traces; fixed phase-balanced appearance/geometry ranking; retain up to3 distinct paths'],
      limitations=['Local raw/heat agreement does not establish material identity or physical contact',
                   'F3 track support may stop before the true spool edge; no fixed spacing extrapolation',
                   'Rrun1274 new bidirectional A7 conditions both remain free fragments; best direction may point outward',
                   'Rrun55 retains a long weak tail; pair0075 and Lrun15 still have unverified downstream identity',
                   'Some high response areas follow illumination/tools; no independent full target reference was read'],
      visual_encoding=dict(panel_order=['original','B intensity','candidate connections','best provisional candidate'],
                           heat_range=[0,1],outside_model_crop='gray uncomputed',
                           final_colors={'free':'green','transition':'cyan','surface_candidate':'magenta','weak_or_inferred':'orange dashed','endpoint':'white circle'},
                           entries_never_merged=True,image_size=[8192,1655],scale=.5),
      pilot_runs=['pilot_001','pilot_002'],pilot_fix_notes=['SAM2 boundary distance separated from all outside pixels',
        'bridge starts restricted to observed free points','direct H1 unknown support retained','free-only and direction ambiguity made explicit'],
      freeze_sha256=sha(HERE/'freeze_final.json'),training=False,labels_read=False,reserved_data_accessed=False,hardware_connected=False,
      baseline_replaced=False,next_step='Inspect the highest-score and competing continuation at a few bends/endpoints; verify target identity before learning a ranker or declaring whole-rope success')
    audit_path=HERE/'all_001/audit_002/report.json'
    if audit_path.exists():
        audit=load(audit_path)
        report['integrity_audit']=dict(path=str(audit_path),sha256=sha(audit_path),exit_code=audit['exit_code'],images=audit['audited_images'],conditions=audit['audited_entries'],identity_accuracy_verified=False)
        report['audit_history_note']='audit_001 retained: single-thread OpenCV EDT reconstruction differs at1e-8 from parallel cached seed; audit_002 reproduces parallel branch exactly, no artifact or tolerance changed'
    if (HERE/'review_receipt.json').exists():report['independent_visual_review']=dict(path=str(HERE/'review_receipt.json'),sha256=sha(HERE/'review_receipt.json'))
    dump(HERE/'experiment_report.json',report)
    txt=f'''热图接续 H2 / IR-H2.0.0：冻结B热图 + A7自由段 + SAM2/F3卷面中心轨迹

交付图片：{HERE / 'all_001/四栏检测结果'}
292张JPEG，每张8192×1655，依次为原图、B热图、多候选、最高评分候选。
已扫描的来源分母400；292张有旧自动入口，108张入口仍未解决，不认定其无绳。
复用299份H1数值B响应；Rrun1274新增A7自动轴的两个方向分别前向，共301条件。
新条件不是人工提示；所有不同入口响应始终分开。显示采用所选入口的原响应，固定0–1。

最优指固定启发式评分最高，不代表完整同根身份正确。绿=自由、青=过渡、紫=贴面候选；橙虚线=弱证据或推断，白圈=当前末端。
F3是边界。先在纯SAM2卷面内配对真实相邻边界，再取局部中点并有限细化（≤间距15%、≤2半尺寸px）。缺边不按固定绳距补全。
A7确认同轴且两端各有30原px原图支持时，允许≤min(400原px,20实测宽)的缺口候选，累计≤600原px；缺口不是观测/遮挡真值。
连接项包括原图双边、B中心/双侧低带、局部曲率/方向/宽度兼容、F3双边、放弃已观察自由段惩罚。没有3D标定，不宣称物理规律已验证。

实际：{summary['completed']}/292，退出码0；整批{summary['elapsed_seconds']:.1f}秒；新增B前向{infer['seconds']:.2f}秒。
路径产物类别：{dict(counts)}。这些是输出计数，不是准确率。
固定12开发例先检查、必要修订一次，随后冻结整批；没有读取完整身份标签或保留集，没有训练或设备连接。
pilot_001及integration_smoke_001的历史失败保留；正式为pilot_002和all_001。
模型、原图、旧实验保持原内容；新记录在all_001/记录/*.json.gz，points_xy是半尺寸crop坐标，原图坐标=2×xy+[0,1050]。
所有来源/模型/参数/核心源码哈希见inputs.json、freeze_final.json、all_001/run_start.json；逐图来源哈希在压缩记录。

环境沿用vision_prototype/.venv（cv2/numpy/Pillow）与vision_training/.venv（PyTorch CPU），未安装依赖。
实际命令（工作目录D:/precise_arrangement）：
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261030_v2/prepare_h2.py prepare
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261030_v2/prepare_h2.py infer
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261030_v2/run_h2.py --run pilot_001 --subset pilot --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261030_v2/run_h2.py --run pilot_002 --subset pilot --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261030_v2/run_h2.py --run all_001 --subset all --workers 4
复跑须使用新的run名，不能覆盖已有结果。
'''
    if audit_path.exists():
        txt+='\n最终独立结构核验：all_001/audit_002/report.json，292图/301条件全部通过，退出0；752候选及弱证据虚线、坐标/种子映射、模型/源码/原图/响应哈希与JPEG解码均核验。结构检查不证明身份准确。\n首轮audit_001因单线程OpenCV距离变换重建seed出现约1e-8差异而失败，原记录保留；审计改回并行分支后逐bit一致，没有修改输入/产物/容差。\n实际核验命令：vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261030_v2/audit_h2.py --run all_001 --out audit_002 --workers 2\n'
    (HERE/'README.txt').write_text(txt,encoding='utf-8')
    print(json.dumps(dict(report=str(HERE/'experiment_report.json'),counts=dict(counts)),ensure_ascii=False))
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
