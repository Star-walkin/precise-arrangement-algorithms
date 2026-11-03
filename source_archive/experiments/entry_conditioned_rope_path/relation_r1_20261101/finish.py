"""Write honest experiment evidence, summaries and navigation. No retuning."""
from common import *
from collections import Counter
import numpy as np,csv,gzip,sys,platform
def gz(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def main():
    protocol=load(HERE/'protocol.json');audit=load(HERE/'audit_receipt.json');render=load(HERE/'render_receipt.json')
    assert audit['exit_code']==0 and audit['checked_count']==292
    train=load(HERE/'model/training_receipt.json');ev=load(HERE/'evaluation/summary.json');checks=load(HERE/'checks.json')
    h6result=load(HERE/'h6_control/summary.json');batch=load(HERE/'batch_002/summary_gallery.json');diag=load(HERE/'training_diagnostics.json')
    chain_v2=load(HERE/'evaluation/chain_diagnostics_v2.json');chains={(r['branch'],r['mode'],r['id']):r for r in chain_v2['rows']}
    assert all(x['exit_code']==0 for x in [train,ev,checks,h6result,batch])
    # Explicit actual visual review, all fixed26 previews plus native failures.
    selection=load(HERE/'visual_review/selection.json')
    notes=dict(
        clear_failure_crosses_multiple_coil_bands=['L_run_20260923_01__f000585__left','dual_20260923_101643__pair0255__right','R_static_20260922_01__f000095__right','dual_20260923_101517__pair0015__right'],
        premature_turn_or_unresolved_attached_identity=['L_run_20260923_01__f000015__left','L_run_20260923_01__f000142__left','L_static_20260922_01__f000100__left','L_run_stop_run_20260922_01__f003393__left'],
        explicit_rejected_prefix_example='L_run_20260923_01__f000395__left',
        multi_entry_example='dual_20260923_101517__pair0075__right retained E1 instead of choosing visually better E2',
        caveat='These are direct visual failure observations, not an independently labelled success-rate inventory. Strong heat/background and supported-white paths can still be wrong material identities.')
    dump(HERE/'visual_review/review.json',dict(actual_viewed_gallery_ids=selection['gallery_fixed_ids'],actual_viewed_count=26,
        native_failures_viewed=['L_run_20260923_01__f000585__left__native_原图','L_run_20260923_01__f000585__left__native_热图与R1','dual_20260923_101643__pair0255__right__native_热图与R1'],
        notes=notes,overall_identity_acceptance=False,outputs_left_unchanged_to_preserve_failures=True))
    control_counts={v:dict(Counter(e['main_kind'] for row in h6result['rows'] for e in row['variants'][v])) for v in ['A','B']}
    drops={v:Counter() for v in ['A','B']}
    for rr in h6result['rows']:
        rec=gz(rr['path'])
        for v,entries in rec['variants'].items():
            for e in entries:drops[v].update(e['connection'].get('rejected',{}))
    # Readable per-case fixed8 metrics, with failures kept and first geometric gap.
    with (HERE/'evaluation/per_case_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        columns=['branch','mode','id','selection','all_coverage5px','attached_coverage5px','edge_near_coverage5px','mean_distance_px',
            'longest_gap_px','attached_longest_gap_px','first_ge60px_gap_start_xy','supported_length_px','inferred_length_px','reachable_generated_nodes','reachable_kept_nodes']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in ev['rows']:
            s=r['selected'];gap=r['first_sustained_reference_gap_ge60px'];chain=chains[(r['branch'],r['mode'],r['sample_id'])]
            writer.writerow(dict(branch=r['branch'],mode=r['mode'],id=r['sample_id'],selection=r['selection_status'],
                all_coverage5px=s['all']['covered_fraction'],attached_coverage5px=s['stages']['attached']['covered_fraction'],edge_near_coverage5px=s['stages']['edge_near']['covered_fraction'],
                mean_distance_px=s['all']['reference_to_prediction_mean_px'],longest_gap_px=s['all']['longest_uncovered_run_px'],
                attached_longest_gap_px=s['stages']['attached']['longest_uncovered_run_px'],first_ge60px_gap_start_xy=gap['from_xy'] if gap else None,
                supported_length_px=s['claimed_status_lengths_px']['visible_supported'],inferred_length_px=s['claimed_status_lengths_px']['inferred'],
                reachable_generated_nodes=chain['connected_reference_near_generated_nodes'],reachable_kept_nodes=chain['connected_reference_near_kept_nodes']))
    # Per-source deployment outputs and runtime; no unknown/tail is turned into GT.
    with (HERE/'per_source_output.csv').open('w',encoding='utf-8-sig',newline='') as f:
        cols=['id','display_entry','entry_count','status','length_px','supported_length_px','inferred_length_px','termination','seconds','output']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in sorted(render['rows'],key=lambda r:r['id']):
            rec=gz(r['conditions'][r['display_entry']]['path']);result=rec['result'];c=result['candidates'][0] if result['candidates'] else None
            w.writerow(dict(id=r['id'],display_entry=r['display_entry'],entry_count=r['total_entries'],status=r['status'],length_px=c['length_px'] if c else 0,
                supported_length_px=c['status_length_px']['visible_supported'] if c else 0,inferred_length_px=c['status_length_px']['inferred'] if c else 0,
                termination=c['termination'] if c else 'no_path',seconds=r['conditions'][r['display_entry']]['seconds'],output=r['output']))
    metrics=ev['summary'];statecounts=dict(Counter(r['status'] for r in render['rows']))
    modelcfg=load(HERE/'model/config.json');pairs=load(HERE/'pair_manifest.json');cost=np.asarray([r['seconds'] for r in batch['rows']])
    report=dict(version='IR-R1.0.0',date_local='2026-11-02',run_directory_date_note='directory keeps experiment start date20261101',
        actual_complete=True,main_outputs=str(HERE/'交付_292张原图与热力图'),png_count=292,unique_sources=292,condition_count=301,
        png_dimensions=[8364,3350],left_original_dimensions=[4096,3000],source_open_gallery=400,old_missing_entries=108,
        new_method='Frozen A750 + native-gray/F3 candidate hints +197dim spatial transition MLP before pool/beam pruning',
        relation_parameter_count=modelcfg['parameters'],relation_training_pairs=828,reviewed_center_low_pairs=804,engineering_weak_neighbor_pairs=24,
        independent_new_human_pair_labels=0,training_images=10,training_source_groups=4,evaluation_images=8,evaluation_source_groups=3,
        fixed_local_pool=dict(given_reviewed_training_history_states=658,reference_next_generated_within4px=658,
            training_initial_pair_order=train['before']['pair_order_fraction'],training_final_pair_order=train['after']['pair_order_fraction'],
            limitation='given correct history; not from-entry continuous reachability or generalization'),
        formal_training=dict(steps=train['steps'],seconds=train['seconds'],smallcheck=load(HERE/'model/smallcheck.json'),A_frozen=True),
        controlled_H6=dict(images=26,entry_conditions=30,counts=control_counts,rejected_by_rule={k:dict(v) for k,v in drops.items()},
            parameters_unchanged=True,identity_accuracy=None,scope='A/B same H6 only; candidate category differences are not correctness'),
        paired_same_new_search_R0_vs_R1=metrics,training_path_ablation=diag['aggregate'],reachability_diagnostic_correction=chain_v2,
        known_visual_failures=notes,identity_acceptance=False,baseline_replaced=False,full_arc_identity_accuracy=None,
        F3_feature_limitation='No exact original F3 scene priors for historical supervised18 outside open400. Constant F3-specific training columns masked to0 also at inference; F3 only supplies proposals/bounded heuristic penalty in gallery, not a learned global relation.',
        active_bridge_rule='weak cumulative distance >72 originalpx stops/trim; max_bridge_original_px48 is a reserved unused field in this prototype, not an enforced claim',
        gallery_stage_contract='gallery stage3 is an unmeasured placeholder, not attached/contact truth. Do not use gallery stage arrays for phase evaluation. Only fixed8 common scenes define posthoc phase proxies.',
        candidate_and_rule_boundary='Existing H6 was not modified. R0/R1 share a new limited search, using soft F3 guidance and native profiles; no claim isolates removal of48px-tier or single-track constraints as the cause of improvement. Their absence in R1 permits serious crossing failures.',
        performance=dict(gallery301_condition_search_wall_seconds=batch['seconds'],per_condition_seconds_quantiles={str(q):float(np.quantile(cost,q)) for q in [.1,.5,.9,1]},
            native292_render_seconds=render['seconds'],audit_seconds=audit['seconds']),
        failure_history=['initial frozenA replay one2-thread numerical mismatch2.115965e-6; inputs identical;8-thread recompute exact, failed artifact retained',
            'first pair-preparation import-path error preserved and repaired before training',
            'batch_001 interrupted for strided-array copying;23 completed conditions exactly reproduced after contiguous-array repair in batch_002'],
        checks=checks,independent_delivery_audit=audit,devices_connected=False,reserved_or_S1_read=False,
        next_step='Confirm a few actual post-bend target and competing-neighbor pairs and restore a validated physical non-crossing/identity check; do not enlarge model or continue training on the8 evaluated images.')
    dump(HERE/'experiment_report.json',report)
    lines=['活动绳接续 R1 离线实验  IR-R1.0.0  完成日期2026-11-02',
        '', '主要交付：'+report['main_outputs'],
        '恰好292张PNG，每个唯一源图一张；301入口分别保留元数据及浮点。全部8364×3350，两个4096×3000原像素主体；左图逐像素核验，未镜像/增强/裁剪。标题、图例和0～1色条均在图像主体之外。',
        '', '先回答结果',
        '已完成冻结A750、原灰补候选、R1小样本拟合/600步训练、剪枝前关系评分、固定输入消融、真实入口逐段搜索和292图导出。',
        '本轮不能作为已通过的整链改进替换P1/H6。局部参考覆盖有小幅增益，但目视明确发现多圈穿越、过早偏离和错误延伸。不能用覆盖数字或长路径抵消这些失败；完整弧段与正确末端成功率未定义。',
        'L_run f000585和右0255的原像素放大已看到白线离开活动绳后跨多条卷面绳纹；R_static095、右0015也有错误延伸。输出原样保留，没有删图或改热图掩盖。',
        '', '受控对比',
        '先冻结26张开放图库/30入口，A750与B750分别重跑同一原H6，无门槛校准。A接轨16、仅前缀12、无候选2；B接轨13、仅前缀15、无候选2。只是入口条件的输出类别，不是准确率。完整删除原因、评分项、48px档及拓扑拒绝示例保存在h6_control/records。',
        '原H6规则没有改写。R0/R1共用新的有限候选搜索，仅关系贡献权重从0切换为冻结R1；因此以下指标只比较R0/R1，不能与旧A连接器或B新版H6混成同一对照。',
        '', '固定八张历史开放评价  三个来源组  5原图px部分参考覆盖',
        '入口方式  方案  全参考覆盖  后段覆盖  近边覆盖  拒识/8']
    for mode in ['AUTO','CONDITIONAL']:
        for b in ['R0','R1']:
            m=metrics[b][mode];c=m['selected_reference_coverage'];lines.append(f"{mode}  {b}  {c['all']['coverage_5px']:.2%}  {c['attached']['coverage_5px']:.2%}  {c['edge_near']['coverage_5px']:.2%}  {m['rejected_images']}/8")
    lines += ['', 'AUTO三张旧入口失败保留在8图分母；CONDITIONAL是同图的已知短入口诊断，不是全自动结果。评价参考在32份两模式/两支预测全部保存并哈希后才读取。没有按评价结果改权重、门槛或挑入口。每图覆盖、均值距离、后段/近边覆盖、最长漏失、首个持续60px几何缺口和父链可达性见evaluation/per_case_metrics.csv及逐图JSON。',
        '', '学习与候选诊断',
        'R1为197→64→32→1的MLP，14785参数。输入包括6个有序空间位置的冻结A解码特征、原生灰度法向剖面、响应/模板、端点与切线几何、最近预测路径和入口关系。图像特征没有整幅平均成一个向量。A权重与归一化状态冻结。',
        '采用单一加权pairwise softplus排序损失。804对来自reviewed中心与已有有效低中心标签，24对来自4处历史Codex工程弱邻绳，均非新增独立人工身份真值。10训练图/4组；828对不是828个独立样本。',
        f"独立副本33对小检查100步通过后丢弃；新增模块重置，再正式600步，实际优化{train['seconds']:.3f}秒（CPU，缓存后的小MLP成本）。训练配对loss由{train['before']['pair_loss']:.6f}至{train['after']['pair_loss']:.8f}，排序55.07%至100%；只说明训练拟合。", 
        '在给对训练历史的658处局部池中，都有4px内的下一段候选；这不证明从真实入口存在一条连续正确路径。完整搜索只用自己的预测历史，父节点和淘汰原因保留。',
        '父链可达性诊断以evaluation/chain_diagnostics_v2.json为准：首次版本错误地用已剔除入口胶囊的参考检查根点，产生误导的全零计数，已另存修正。根点不在部分参考范围内时记不可评价，不把未知前缀判为错误或“没有生成正确候选”。局部边并集与单条父链分开。覆盖指标、入口排除口径、预测及PNG均未改变。',
        '原图补候选为有限双侧剖面峰、A响应峰及F3位置提示；域外和不完整原图剖面保持硬拒绝。R1对每条生成边先评分，再做pool合并和beam top8，最多150步、每步约24原图px、最终3候选。未扩大网络或追加下一点CNN。',
        '', '重要实现和证据限制',
        report['F3_feature_limitation'],
        '因此这版没有学成完整的灰轨关系；弱F3仅作提示和有限惩罚，也失去了旧H6单轨/非跨轨的强保护。实际跨轨反例说明不能把所有物理限制交给这组弱配对MLP。',
        '48px长直段分档、边界配对锁轨的删除原因已保存，但缺少对被拒候选的独立完整身份真值，无法证明这些规则应一律删除；本轮没有作该因果结论。R1的失败本身否定了“只换为软先验就够”的假设。',
        '实现以累计弱支持超过72px截断；max_bridge_original_px=48是未生效保留字段，不应误读为强制48px桥接上限。橙虚线表示弱证据连接，不等于真实遮挡确认。',
        '固定配对特征屏蔽只作训练侧诊断：屏蔽A空间块仍约99.76%训练排序，屏蔽剖面或几何块仍100%；不能据此宣称学到了不可替代的空间身份关系。',
        '图库路径记录的stage=3是未测量占位，不代表贴面、接触或物理阶段真值，禁止据此给292图分阶段评分。正式阶段评价仅使用固定8图相同的事后视觉场景分类。',
        '', '图像和入口规范',
        '主展示入口在新推理前固定为旧清单原编号E1；R1跨入口分数未校准，未沿用只适用于B/H6的终点排名来挑选A/R1展示。多入口总数及歧义在图上明确标注，其他入口有独立浮点和路径记录；没有按好看或参考覆盖选择入口，也没有取多入口max热图。',
        '右栏底图始终为冻结A750原始中心响应，0～1固定TURBO。原生975×2048 float32回映至4096×1950；域外灰色未计算；未逐图拉伸，没有把路径涂成高斯新热图。白色实线只表示算法声称的图像支持，目视已证实其中仍有错误身份，不能称可信概率。',
        '292图库有训练来源重叠，原400图另108张旧入口未解决；本轮没有400图全识别或292个独立测试的结论。',
        '', '实际成本和核验',
        f"301入口全路径运行{batch['seconds']:.1f}秒，单条件中位{np.quantile(cost,.5):.3f}秒/P90 {np.quantile(cost,.9):.3f}秒；292张原像素PNG绘制{render['seconds']:.1f}秒。另有冻结A特征前向、控制26图、固定52项路径和完整审计，均保留分阶段日志。", 
        '初次A复现一张2线程与原8线程差2.115965e-6，输入逐位相同，已按原8线程重算为精确一致；没有放宽容差。首次pair脚本导入错误及修复日志保留。batch_001因数组反复拷贝中断，batch_002仅改存储布局；23份已完成旧结果与新候选逐项完全一致。',
        '独立审计全部292张PNG解码、尺寸、完整左图像素、域外灰色、301份原浮点、候选原图坐标、关系在剪枝前参与、旧模型/源码哈希，均通过。另实际查看固定26图与3张失败原像素细节。结构通过不等于识别通过。',
        '', '下一项最有价值的工作',
        '先核验训练侧left f00240和right f00130分歧后的目标短段与竞争邻绳，最小包已在最小接续核验包。恢复经验证的非跨圈身份保护，再检验评分是否保住正确弯段；不再用训练100%或热带更长作为通过标准。本轮未据此回填新标签或重训。',
        '', '无设备连接、无保留集或S1标签访问。原图、A/B750、H6/P1/V1/V2和旧输出保持原状。R1未替换任何基准或在线流程。']
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    readme=['R1 离线实验说明', '主交付：交付_292张原图与热力图/  恰好292个原像素两栏PNG。',
        '原始浮点响应/<源图>/entry_00.npz 的response键为未叠加975×2048 float32；features是冻结A的8通道空间缓存，非标签。通过NTFS硬链接复用本轮不可变cache，避免重复磁盘占用。不要分别改写链接副本。',
        '原图=2×模型裁框坐标+[0,1050]；图像回映采用双线性align_corners=False。',
        '模型与训练日志：model/。冻结输入：protocol.json、cache_jobs.json、pair_manifest.json。运行与评价：h6_control/、development_001/、fixed_001/、batch_002/、evaluation/。',
        '失败与性能修复保存在batch_001/、batch_001_interruption.json、cache_all_initial_failed.log、checks.json，不覆盖为首次成功。',
        '全部源码为纯离线。训练用vision_training/.venv，图像和路径用vision_prototype/.venv，不安装依赖。',
        '实际命令入口及顺序如下，均在D:\\precise_arrangement运行。复现请先把本目录py文件复制到同级新实验目录并更换相应目录名，勿覆盖现有冻结运行：']
    for env,script,args in [
        ('vision_prototype','freeze.py',''),('vision_training','cache_a.py','--subset all --workers 4'),('vision_training','cache_a.py','--subset fixed --workers 2'),
        ('vision_prototype','prepare_pairs.py',''),('vision_training','train_relation.py',''),('vision_prototype','control_h6.py',''),
        ('vision_prototype','run_paths.py','--subset train --branches R0 R1 --out development_001 --workers 4'),('vision_prototype','audit_training.py',''),
        ('vision_prototype','freeze_selection.py',''),('vision_prototype','run_paths.py','--subset fixed --branches R0 R1 --out fixed_001 --workers 4'),
        ('vision_prototype','run_paths.py','--subset gallery --branches R1 --out batch_002 --workers 4'),('vision_prototype','render_batch.py',''),
        ('vision_prototype','evaluate_fixed.py',''),('vision_training','check_r1.py',''),('vision_prototype','make_review.py',''),('vision_prototype','audit_delivery.py','')]:
        readme.append(env+'/.venv/Scripts/python.exe -B '+str((HERE/script).relative_to(ROOT)).replace('\\','/')+' '+args)
    readme.append('结论：局部几何覆盖有变化，完整同根识别未通过；尤其保留明显跨圈错误，详见REPORT.txt。')
    (HERE/'README.txt').write_text('\n'.join(readme)+'\n',encoding='utf8')
    dump(HERE/'final_source_hashes.json',{str(p):sha(p) for p in HERE.glob('*.py')})
    # Current indices record a research failure/partial gain, never a replacement.
    note='\n\n## 2026-11-02 接续关系 R1 离线实验\n\n[R1报告](relation_r1_20261101/REPORT.txt) · [292张原像素两栏PNG](relation_r1_20261101/交付_292张原图与热力图/)。冻结A750，MLP 600步，在pool/beam前评分；同搜索R0/R1固定8图AUTO后段5px覆盖12.30%→19.42%，CONDITIONAL16.11%→17.98%，仅部分几何参考。实际看到L_run585、0255等跨多圈/错误延伸，完整身份未通过，不替换P1/H6。10训练图/4组，828局部弱配对，非独立身份真值；292图/301入口全产物及像素核验通过，原400另108旧入口未解决。无设备、保留集或S1访问。\n'
    register=[]
    for path,text in [(BASE/'README.md',note),(ROOT/'数据资产/README.md',note.replace('(relation_r1_20261101/','(../experiments/entry_conditioned_rope_path/relation_r1_20261101/'))]:
        before=sha(path)
        if b'2026-11-02' not in path.read_bytes() or b'relation_r1_20261101' not in path.read_bytes():
            with path.open('ab') as f:f.write(text.encode('utf8'))
        register.append(dict(path=str(path),before=before,after=sha(path)))
    agents=ROOT/'AGENTS.md';before=sha(agents);text=agents.read_text(encoding='utf8')
    if 'IR-R1.0.0' not in text:
        anchor='- V14训练拟合接近100%'
        bullet='- 2026-11-02新增[接续关系R1 / IR-R1.0.0](experiments/entry_conditioned_rope_path/relation_r1_20261101/experiment_report.json)：冻结A750，197维空间/原灰关系MLP 600步，10训练图/4组828局部配对（含24工程弱邻绳对），在pool/beam剪枝前参与；固定8图/3组同搜索R0/R1的AUTO后段5px覆盖12.30%→19.42%、CONDITIONAL16.11%→17.98%。局部指标不抵消已目视确认的L_run585、0255等跨多圈和错误延伸；F3关系训练不足、完整身份/末端未通过，不替换P1/H6。292源图/301条件完成，恰好292张原像素两栏PNG及301份浮点，独立产物审计通过；400分母另108旧入口未解决。无设备、保留集或S1访问。\n'
        assert anchor in text;text=text.replace(anchor,bullet+anchor,1);agents.write_text(text,encoding='utf8')
    register.append(dict(path=str(agents),before=before,after=sha(agents)))
    dump(HERE/'final_receipt.json',dict(exit_code=0,main_directory=str(HERE/'交付_292张原图与热力图'),source_count=292,float_conditions=301,
        output_bytes=sum(r['bytes'] for r in render['rows']),image_audit_passed=True,visual_identity_acceptance=False,
        report_path=str(HERE/'REPORT.txt'),report_sha256=sha(HERE/'REPORT.txt'),registration=register,baseline_replaced=False))
    print('Complete292 PNG,301 float conditions. Identity acceptance: FAILED. No baseline replaced.')
if __name__=='__main__':main()
