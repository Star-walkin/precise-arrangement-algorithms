from collections import Counter
import sys
from common import HERE,H5,ROOT,load,dump,sha
def main():
    freeze=load(HERE/'freeze_final.json');s=load(HERE/'all_001/summary.json');cfg=load(HERE/'inputs.json')
    assert s['completed']==292 and s['exit_code']==0 and s['source_unchanged']
    assert all(sha(p)==v for p,v in (freeze['sources']|freeze['reused_sources']).items())
    audit=load(HERE/'all_001/independent_geometry_audit.json');assert audit['status']=='passed'
    assert load(HERE/'extra_001/independent_geometry_audit.json')['status']=='passed'
    assert (HERE/'review_receipt.json').exists() and (HERE/'comparison_all_001/summary.json').exists()
    comparison=load(HERE/'comparison_all_001/summary.json')
    coverage=comparison['by_image']['H6']
    removed=coverage['removed_tail_support_lengths']
    retained_delta=(coverage['full_observation_retained_by_selected_support_lengths']['original_image_supported']
        -comparison['by_image']['H5']['full_observation_retained_by_selected_support_lengths']['original_image_supported'])
    reproduced=[]
    for name,n in [(freeze['selected_pilot'],16),('extra_001',8)]:
        sub=load(HERE/name/'summary.json');start=load(HERE/name/'run_start.json')
        assert sub['completed']==n and sub['exit_code']==0 and sub['source_unchanged']
        assert start['sources']==freeze['sources'] and start['reused_sources']==freeze['reused_sources']
        for r in sub['rows']:
            p=HERE/name/'四栏检测结果'/f'{r["id"]}.jpg';q=HERE/'all_001/四栏检测结果'/p.name
            assert sha(p)==sha(q)==r['sha256'];reproduced.append(dict(id=r['id'],source_run=name,sha256=r['sha256']))
    counts=dict(Counter(r['kind'] or 'no_candidate' for r in s['rows']))
    report=dict(version='IR-H6.0.0',source_open_images=400,processed_images=292,entry_conditions=301,
        source_groups=len({j['source_group'] for j in cfg['jobs']}),unresolved_old_entries=108,
        output_counts=counts,outputs=str(HERE/'all_001/四栏检测结果'),elapsed_seconds=s['elapsed_seconds'],exit_code=0,
        reused_B_responses=301,model_B=cfg['model_B'],new_forward_passes=0,training=False,
        parameters=freeze['parameters'],surface_parameters=freeze['surface_parameters'],topology_parameters=freeze['topology_parameters'],
        coverage_tradeoff=dict(trimmed_images=coverage['task_trimmed_count'],removed_tail_support_lengths_original_px=removed,
            selected_original_support_retention_delta_original_px=retained_delta,
            selected_original_support_retention_decreased_images=comparison['by_image']['retained_original_image_support_decreased_count'],
            note='A prefix stopped before an unsupported crossing can exclude later original-image-supported segments. They remain in full_free_observation; trimming is not counted as correct recognition.'),
        source_frozen_before_extra8=True,extra8_status='already-viewed open regression cases, not independent blind test',
        fixed24_jpg_subset_full_identical=reproduced,
        implementation=dict(surface='Frozen accepted C3/D4/F2/F3 source vectors, source correspondence and F3 complete C4 domain; fixed adjacent boundary lineage pairs; exact midpoints, no B-dependent geometric shift or equal-spacing fabrication.',
            source_gaps='Only bounded gaps within one accepted saved vector and known exclusion are inferred. Missing source ends and uncertain adjacency are not bridged into invented tracks.',
            free='The complete H5 straight observation remains saved. A crossing of known centers requires a continuous48-original-px original+B or strict original-A7 support run covering that exact intersection in a96px window. Otherwise a task prefix ends one local width before the first unlicensed crossing and the remaining tail is unresolved.',
            attachment='The entire bridge and attached tail are checked against all other finite center tracks. Other-lane crossing/touch/overlap is ineligible. Target capture may occur only at the terminal endpoint or an exactly coincident suffix checked at the union of both curves knots.',
            identity='One fixed source boundary pair per attached tail, monotone source parameter. Raw and inferred source status remains explicit. Geometrically legal but image-weak candidates are diagnostic only with negative rank and cannot defeat the free-prefix alternative.'),
        limitations=['A permitted free projection intersection is not proof of free/attached physical state. No physical contact is inferred merely from SAM2 entry.',
            'No crossings of the available retained center graph does not establish correct target identity or rule out missing real ropes.',
            'The0255 counterexample still lacks an accepted complete continuation; removal of crossing bridges is not complete recognition.',
            f"Task-prefix clipping affected {coverage['task_trimmed_count']} images and excluded {removed['original_image_supported']:.1f} original-image-supported pixels of later path span. This coverage loss is retained, not called unknown truth or successful rejection.",
            'Missing visible tails, uncertain terminal identity and source gaps remain. No true identity accuracy or near-perfect guarantee is reported.'],
        original_changed=False,historical_frozen_sources_changed=False,labels_read=False,reserved_access=False,devices_connected=False,baseline_replaced=False)
    for name,key in [('all_001/independent_geometry_audit.json','audit'),('comparison_all_001/summary.json','comparison'),('review_receipt.json','visual_review'),
        ('surface_checks_001/receipt.json','surface_checks'),('surface_checks_001/source_end_gap_diagnosis.json','source_gap_diagnosis'),
        ('topology_contract_check.json','topology_checks'),('diagnostics_source/suffix_before_fix.json','preserved_regression_failure'),
        ('diagnostics_source/suffix_after_fix.json','regression_repair'),('extra_001/independent_geometry_audit.json','extra_audit'),
        ('source_continuity_all_001/summary.json','source_continuity_statistics'),('environment.json','environment'),
        ('comparison_all_001/comparison_review_receipt.json','comparison_review')]:
        p=HERE/name;report[key]=dict(path=str(p),sha256=sha(p),result=load(p))
    dump(HERE/'experiment_report.json',report)
    (HERE/'README.txt').write_text(f'''热图接续 H6 / IR-H6.0.0：F3来源中心轨迹与非跨轨接续

四栏图库：{HERE/'all_001/四栏检测结果'}
实际292/292、301个原自动入口条件，退出0，{s['elapsed_seconds']:.1f}秒。输出分类{counts}，不是识别正确率。
仍为原开放400图；108个旧入口未解决不改称无绳。未训练、未新增模型前向、未访问标签/保留集或连接设备。

改动文件：surface_tracks.py复用F3接受向量/来源谱系，替代旧raster连通块配对；topology.py检查完整几何；connect_paths.py拒跨轨桥并锁单轨尾；run_h6.py保留完整原直轴、任务前缀、竞争候选及四栏图。
F3完整C4范围与独立SAM2分开保存。灰实线为原F3支持，灰虚线为源内推断。源缺失不能用固定间距或邻近行强补。中心线固定双边来源，几何不随B热点漂移。
自由段图像投影相交与贴面跳轨分开。交点自身必须落在连续原图联合支持内，不能只因附近亮纹放行。弱尾跨轨被单列为unresolved_surface_projection，完整H5观测仍在full_free_observation。
全量{coverage['task_trimmed_count']}图任务前缀发生裁剪，移出的后段中仍包含{removed['original_image_supported']:.1f}原图联合支持像素长度；这是保留的覆盖代价，不能将全部裁尾称为无证据或正确拒绝。
计入最后接续路径后，按固定距离口径保留的原直段图像支持长度相对H5净变化{retained_delta:+.1f}原px，{comparison['by_image']['retained_original_image_support_decreased_count']}图下降。这也是覆盖诊断，不是身份准确率。
桥及贴面段不可碰穿非目标灰轨；目标只允许末端一次捕获或之后精确不离线，按两曲线联合节点检查。贴面尾保持同一来源边界对，不换线。
第三栏仅画拓扑许可候选；图像弱的合法候选标“弱证据未采用”，不能获选。橙虚线为弱证据，绿色也不等于同根真值。

固定16开发（旧14+用户0345/0255）与冻结后旧8开放回归；24图子集与全量JPG逐字节相同。图1弱直尾拦截后有单轨接续候选，图2的非法桥被去掉但完整接续未解决。不能把截短、无交叉或单轨约束符合称为整根识别成功。
首轮与共线后缀合成失败保留，修复结果另存。原图/旧F3/H5/model均不改。参数/输入/源码哈希、命令、失败/跳过见inputs.json、freeze_final.json、每个run的run_start/source_snapshot/summary/progress。
浮点B仍在每entry的response_file，以原hash绑定；坐标原图=2*crop_half_xy+[0,1050]。

实际全量命令：vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v6/run_h6.py --run all_001 --subset all --workers 4
既有CPU环境，不安装/修改依赖。复跑用新run名，勿覆盖历史。
''',encoding='utf-8')
    print(counts)
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
