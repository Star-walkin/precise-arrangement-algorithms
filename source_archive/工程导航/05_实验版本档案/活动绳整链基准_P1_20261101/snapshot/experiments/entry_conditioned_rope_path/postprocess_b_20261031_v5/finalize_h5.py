"""Actual H5 run receipt; no prediction or parameter changes."""
from collections import Counter
import sys
from common import HERE,ROOT,load,dump,sha

def main():
    s=load(HERE/'all_001/summary.json');cfg=load(HERE/'inputs.json');freeze=load(HERE/'freeze_final.json')
    assert s['exit_code']==0 and s['completed']==292 and s['source_unchanged']
    assert all(sha(p)==h for p,h in (freeze['sources']|freeze['reused_sources']).items())
    assert load(HERE/'all_001/independent_geometry_audit.json')['status']=='passed'
    assert load(HERE/'extra_001/independent_geometry_audit.json')['status']=='passed'
    assert (HERE/'review_receipt.json').exists()
    paired_reproduction=[]
    for name,n in [(freeze['selected_pilot'],14),('extra_001',8)]:
        subset=load(HERE/name/'summary.json');start=load(HERE/name/'run_start.json')
        assert subset['exit_code']==0 and subset['completed']==n and subset['source_unchanged']
        assert start['sources']==freeze['sources'] and start['reused_sources']==freeze['reused_sources']
        for row in subset['rows']:
            a=HERE/name/'四栏检测结果'/f'{row["id"]}.jpg';b=HERE/'all_001/四栏检测结果'/a.name
            assert sha(a)==sha(b)==row['sha256']
            paired_reproduction.append(dict(id=row['id'],source_run=name,jpg_sha256=row['sha256']))
    counts=dict(Counter(r['kind'] or 'no_candidate' for r in s['rows']))
    pilots=[p.name for p in sorted(HERE.glob('pilot_*')) if (p/'summary.json').exists()]
    report=dict(version='IR-H5.0.0',source_open_images=400,processed_images=292,entry_conditions=301,
        unresolved_entry_images=108,unresolved_is_not_absent_rope=True,source_groups=len({j['source_group'] for j in cfg['jobs']}),
        outputs=str(HERE/'all_001/四栏检测结果'),path_output_counts=counts,elapsed_seconds=s['elapsed_seconds'],exit_code=0,
        frozen_sources_unchanged=True,freeze_record=dict(path=str(HERE/'freeze_final.json'),sha256=sha(HERE/'freeze_final.json')),
        fixed22_subset_vs_full_jpg_identical=paired_reproduction,model_B=cfg['model_B'],reused_float_responses=301,new_inference=0,training=False,
        development_ids=cfg['pilot_ids'],postfreeze_open_check_ids=cfg['extra_ids'],pilot_runs=pilots,
        extra_check_is_blind_test=False,protocol=cfg['selection_protocol'],parameters=freeze['parameters'],heat_parameters=freeze['heat_parameters'],
        implementation=dict(primary='Measured supported straight intervals, excluding input hint and unknown gaps.48-original-pixel length bands have priority; no2000px reward cutoff.',
            secondary='Within a length band:0.60 length,0.25 smaller direction change,0.10 supported F3 center track,0.05 image evidence, minus bridge/width/abandonment costs. Secondary utility is bounded below1, so it cannot reverse length bands.',
            geometry='First departure from straight axis is measured geometrically, not inferred from stage labels; excessive supported straight loss makes a join ineligible.',
            surface='Exact source points of a single F3 adjacent-boundary center track. Free-only is a real alternative; no fabricated perfect-turn score for an unobserved continuation.',
            original_image_fallback='Finite, matched accepted-periodic A7 observations must agree with freshly measured original-image bilateral structure. These points are separate raw_a7_supported evidence; they never change B/protect_support or become identity truth.',
            candidate_preservation='B-only shadow scores preserve original B refinement parents; the finite generated coarse/fine pool competes using observed support. Actual sampling density is recorded per axis, and does not earn length by point count.',
            task_direction='Only paired opposite conditions of the same A7 candidate are contrasted. Actual outside-to-SAM2 progression,72px continuous B support inside and48px total joint support inside can exclude the opposite direction with less than36px B support inside. No infinite-line intersection or model-map merging.',
            scope='One selected free-axis hypothesis is connected; retained alternative axes are diagnostic, not a globally solved material-identity graph.',
            loss='Negative decision utility, not a neural training loss or calibrated identity probability.'),
        limitations=['No independent full identity/end-point reference was used; near-perfect recognition is not established.',
            'Longer image-supported straight output and a low geometric cost do not prove same-rope continuation.',
            'Misses, possible overextension and ambiguous endpoints must remain visible; unknown is not automatically a correct rejection.'],
        raw_data_changed=False,labels_read=False,reserved_data_accessed=False,devices_connected=False,baseline_replaced=False)
    for name,key in [('comparison_all_001/summary.json','comparison'),('all_001/independent_geometry_audit.json','audit'),('review_receipt.json','visual_review'),('score_contract_check.json','score_contract'),('direction_contract_check.json','direction_contract'),('free_checks_004/final_contract_audit.json','free_contracts'),('environment.json','environment'),('extra_001/summary.json','extra_run'),('extra_001/independent_geometry_audit.json','extra_audit')]:
        p=HERE/name
        if p.exists():report[key]=dict(path=str(p),sha256=sha(p),result=load(p))
    dump(HERE/'experiment_report.json',report)
    text=f'''热图接续 H5 / IR-H5.0.0：有支持的长直段优先，再比较方向与灰色中心轨迹

四栏图库：{HERE/'all_001/四栏检测结果'}
实际292/292、301入口、退出0，{s['elapsed_seconds']:.1f}秒；{counts}。这些是产物类别，不是识别准确率。
原400图中的108张旧入口未解决，仍留总分母，不能称无绳。

同一冻结B模型、301份浮点响应与自动入口；仅离线后处理，无训练/新前向/人工标签/保留集/设备访问。
自由段按实际支持区间长度排序，入口提示和未知缺口不计长度。48原px长度分档先于次级评分。
同档：0.60长度、0.25方向、0.10受支持F3轨迹、0.05图像，再扣桥接弱证据/宽度变化等；次级分固定小于1，不得反转长度主序。
第一次几何偏轴及SAM2远侧终域均为约束/诊断，不是真实接触或身份真值。未观察后段不能获得虚构零转向奖励。
热图弱但原图双边清楚时，仅利用已匹配、已接受且有股纹证据的A7有限观测进行保守补段；原图复测通过才计支持，单列raw_a7_supported，绝不改写B支持指标。
同一A7的相反方向条件仅在实际路径有明确外→内支持差异时筛任务方向；B图不合并，不能据此宣称物理接触或完整同根。
主轴之外备选轴留作诊断，尚未全局解决跨轴身份竞争。未证明“几乎全部正确”。

绿色=原图+B联合，黄色=仅热形态支持，浅蓝=A7+原图补段，青/紫=接续/贴面候选，橙虚线=弱证据或推断。
热图固定0～1，称响应强度。记录xy映射原图=2*xy+[0,1050]，浮点B在每条记录的response_file中保持原hash。
14张固定失败/对照开发；实际开发轮次：{pilots}。额外8图是冻结后开放开发核查，不是独立保留集，也不据此回流调参。
源/参数/运行命令：inputs.json、freeze_final.json、各run/run_start.json、source_snapshot、progress.jsonl。
查看comparison_all_001和review_receipt.json中的改善、退化与未解决案例；不能只按总长度判断提升。

运行环境：D:/precise_arrangement/vision_prototype/.venv/Scripts/python.exe，CPU，既有依赖不改。
实际全量命令：vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v5/run_h5.py --run all_001 --subset all --workers 4
复跑请用新run目录，不覆盖历史结果。
'''
    (HERE/'README.txt').write_text(text,encoding='utf-8')
    print(counts)
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
