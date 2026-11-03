"""Actual-run receipt only. Never supplies labels or feeds back into detection."""
from collections import Counter
from datetime import datetime,timezone
import gzip,json,sys
from common import HERE,H3,load,dump,sha

def main():
    s=load(HERE/'all_001/summary.json');i=load(HERE/'inputs.json');f=load(HERE/'freeze_final.json')
    assert s['exit_code']==0 and s['completed']==292 and s['source_unchanged']
    assert all(sha(p)==v for p,v in (f['sources']|f['reused_sources']).items())
    counts=dict(Counter(r['kind'] or 'no_candidate' for r in s['rows']))
    report=dict(version='IR-H4.0.0',completed_utc=datetime.now(timezone.utc).isoformat(),
        source_open_images=400,processed_images=292,entry_conditions=301,unresolved_entry_images=108,
        unresolved_is_not_absent_rope=True,source_groups=len({j['source_group'] for j in i['jobs']}),
        outputs=str(HERE/'all_001/四栏检测结果'),path_output_counts=counts,
        elapsed_seconds=s['elapsed_seconds'],exit_code=s['exit_code'],B_model=i['model_B'],
        reused_float_responses=301,new_network_inference=0,training=False,
        modifications=dict(free='B center and two low-side morphology re-estimate finite entry-linked straight axis, position and observed extent. A7 is an initial estimate, not a fixed axis or endpoint. Raw corroboration and heat-only observations are separate.',
            connection='Reject departures from the protected heat ridge; gray F3 track tail remains exact. Image weight0.70, direction0.20, lookahead0.10; image quality uses heat center gated by two sides0.55, dual shape0.30, raw0.15.',
            abstention='Free-only is a genuine competing candidate at fixed utility0. A connection needs positive utility after0.25 subtraction and sustained compatible B evidence on the near lane. Utility is not confidence.',
            display='Green raw+B, yellow B-only, cyan transition, magenta surface, orange dashed weak/inferred; fixed0..1 heat colorbar.'),
        fixed_development_ids=i['pilot_ids'],parameters=f['parameters'],heat_parameters=f['heat_parameters'],
        pilots=['pilot_001','pilot_002'],
        bounded_revision='pilot_001 retained: own-length mean quality rewarded short bright pieces and near-angle refinement removed a longer eligible axis. One revision uses a common2000-original-px support integral and reserves one longest eligible coarse axis for refinement. Evidence, angle, width and gap thresholds unchanged.',
        limitations=['No independent full target identity reference; no localization accuracy reported.',
            'Response morphology, longer paths and exact F3 adherence do not prove same-rope identity.',
            'Only the existing292 automatic-entry images are processed;108 unresolved entries remain in the400 denominator.',
            'A straight heat ridge may still be a tool, reflection or incorrect entry; weak/competing endings are retained.',
            'L_run_20260923_01__f000015__left and f000142__left visibly continue farther than the selected short free output. These are retained visible-coverage failures, not correct rejections merely because output is unknown.',
            'dual_20260924_111500__pair0015__left follows a straight axis beyond the right spool side with weak/heat-only support; possible overextension retained, not counted as verified recovery.'],
        original_data_changed=False,labels_read=False,reserved_data_accessed=False,devices_connected=False,
        baseline_replaced=False,freeze_sha256=sha(HERE/'freeze_final.json'))
    for name,key in [('comparison_all_001/summary.json','paired_comparison'),('all_001/independent_geometry_audit.json','independent_audit'),('review_receipt.json','visual_review'),('free_checks_002/check_receipt.json','free_module_checks')]:
        p=HERE/name
        if p.exists():report[key]=dict(path=str(p),sha256=sha(p),summary=load(p))
    comparison_text=''
    if 'paired_comparison' in report:
        b=report['paired_comparison']['summary']['by_image'];before=b['H3']['sums'];after=b['H4']['sums']
        changes={k:100*(after[k]/before[k]-1) for k in before if before[k]}
        report['same_input_observations']=dict(percent_changes=changes,selected_entry_changes=report['paired_comparison']['summary']['selected_entry_changed'],
            free_longer_images=b['H4_free_longer_count'],free_shorter_images=b['H4_free_shorter_count'],
            interpretation='More matched heat-ridge evidence is not target identity accuracy. Selected full-path supported length decreased; no universal complete-path improvement claimed.')
        comparison_text=(f"\n同条件H3/H4：自由总长度{changes['free_length_original_px']:+.2f}%，自由热形态支持长度{changes['heat_supported_length_original_px']:+.2f}%，自由原图+热联合支持长度{changes['joint_supported_length_original_px']:+.2f}%；所选整路径联合支持长度{changes['selected_joint_length_original_px']:+.2f}%。按同一诊断门槛比较，不能换算真实定位准确率。完整贴面身份与末端未验证。\n")
    dump(HERE/'experiment_report.json',report)
    (HERE/'README.txt').write_text(f'''热图接续 H4 / IR-H4.0.0 — B热图重新估计直段，避免提前跑偏接旧圈

图库：{HERE/'all_001/四栏检测结果'}
实际292/292张、301入口、退出0、{s['elapsed_seconds']:.1f}秒。原400张中108张旧入口未解决，不称无活动绳。
四栏：原图 / 固定0～1的B热图 / 竞争路径 / 最高决策效用候选。原模型和301份浮点响应未改，无训练、新前向。
产物类别：{counts}。类别数量不是完整识别率。

先用有限角度和法向位置搜索重估B直脊，再决定有限可见延伸；A7只提供初轴/宽度，不锁死中心与旧端点。
绿色为原图+B局部支持；黄色为B双侧热形态单独支持；橙虚线为弱证据/推断。均不等于物理同根真值。
F3相邻边界中心轨迹保持源点；接轨不能舍弃持续的B直绳。近端缺乏B形态时保留自由段，不强制选平滑灰圈。
自由段不接轨是显式候选，分数为固定决策效用，不能当真实概率。
已知失败：左视角L_run f000015和f000142图中可见绳仍继续，但输出短段未覆盖；不是标成未知就算正确。更长/更强响应不能直接解释为同根接续成功。
疑似过延伸：左dual_20260924_111500 pair0015直轴尾伸到卷面右侧之外，黄/橙支持限定保留，未当成已确认完整路径。

记录：all_001/记录/*.json.gz，半尺寸crop坐标xy映射原图=2*xy+[0,1050]。
配置：inputs.json、freeze_final.json、all_001/run_start.json；source_snapshot为实际运行代码；progress.jsonl逐图运行日志。
环境：原vision_prototype/.venv，CPU后处理；原始文件、模型、冻结H1/H2/H3不改；无标签/保留集/设备操作。
固定旧12例加用户R_static_20260922_01__f000095__right反例共13例开发，之后一次全量。
pilot_001短亮段均值偏置反例保留；唯一修订将评分改为共同2000原px视野的支持积分，细化保留一条最长合格轴，阈值/搜索预算不变；pilot_002后冻结。

实际运行命令从D:/precise_arrangement：
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v4/run_h4.py --run pilot_001 --subset pilot --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v4/run_h4.py --run pilot_002 --subset pilot --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v4/run_h4.py --run all_001 --subset all --workers 4
复跑使用新run名，不能覆盖旧运行。
'''+comparison_text,encoding='utf-8')
    print(counts)
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
