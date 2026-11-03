"""Summarize actual frozen experiment outputs; never infer identity accuracy."""
from collections import Counter
from common import HERE,BASE,H6,ROOT,load,dump,sha
def main():
    p=load(HERE/'protocol.json');choice=load(HERE/'selection.json');pair=load(HERE/'paired_comparison.json')
    train=load(HERE/'common_training_fit.json');ev=load(HERE/'evaluation/summary.json');batch=load(HERE/'batch_inference/manifest.json')
    out=HERE/'postprocess/all_001';summ=load(out/'summary.json');audit=load(out/'independent_geometry_audit.json')
    assert summ['completed']==292 and summ['exit_code']==0 and summ['source_unchanged']
    assert audit['status']=='passed' and batch['completed']==292 and batch['entry_conditions']==301 and batch['exit_code']==0
    assert ev['prediction_records']==48 and ev['all_frozen_prediction_files_unchanged']
    assert all(sha(k)==v for k,v in choice['source_hashes'].items())
    assert all(sha(k)==v for k,v in p['source_H6_core'].items())
    assert (HERE/'visual_review.json').exists()
    visual=load(HERE/'visual_review.json')
    repair=load(HERE/'batch_inference/repair_receipt.json')
    failed=load(HERE/'batch_inference_failed_001/failure_receipt.json')
    photo=load(HERE/'photometry_diagnostic/summary.json')
    response_stats=load(HERE/'batch_inference/response_diagnostics.json')
    assert repair['exit_code']==0 and repair['saved_first_failed_response_equals_rerun']
    assert repair['successful_forward_calls']==301 and repair['failed_forward_calls']==1 and failed['exit_code']==1
    assert photo['exit_code']==0 and photo['forward_calls']==24 and photo['source_unchanged']
    pilot=load(HERE/'postprocess/pilot_001/summary.json');assert pilot['completed']==len(p['gallery_fixed_diagnostic_ids']) and pilot['exit_code']==0
    reproduced=[]
    for row in pilot['rows']:
        for dirname in ['四栏检测结果','原图_旧B_新热力图']:
            a=HERE/'postprocess/pilot_001'/dirname/(row['id']+'.jpg');b=out/dirname/a.name
            assert sha(a)==sha(b);reproduced.append(dict(id=row['id'],directory=dirname,sha256=sha(a)))
    actual_training={v:load(HERE/f'branches/{v}/model/result.json') for v in ['C','D']}
    counts=dict(Counter(r['kind'] or 'no_candidate' for r in summ['rows']))
    variant=choice['selected_variant']
    receipts={}
    for name in ['protocol.json','paired_comparison.json','common_training_fit.json','selection.json','numerical_checks.json','smallcheck/result.json',
        'baseline_replay/receipt.json','h6_adapter_replay_002/receipt.json','diagnosis/freeze_diagnosis.json','diagnosis/input_diagnosis.json',
        'evaluation/freeze.json','evaluation/summary.json','batch_inference/manifest.json','postprocess/all_001/summary.json',
        'postprocess/all_001/independent_geometry_audit.json','visual_review.json',
        'batch_inference/repair_receipt.json','batch_inference_failed_001/failure_receipt.json',
        'photometry_diagnostic_protocol.json','photometry_diagnostic/summary.json','photometry_diagnostic/visual_review_receipt.json',
        'comparison_responses/response_comparison.json','training_execution_receipt.json','training_audit.json',
        'batch_inference/response_diagnostics.json','evaluation/execution_receipt.json','evaluation/independent_qa.json','review_receipt.json']:
        q=HERE/name;receipts[name]=dict(path=str(q),sha256=sha(q))
    report=dict(version='IR-H7.0.0',user_facing_date='2026-11-01',completed='two matched bounded continuations, frozen partial-reference evaluation, new301 forward responses, rerun292 galleries',
        selected_variant=variant,selected_by_training_only=True,training_guards=choice['guards'],models=choice['models'],
        train_images=10,train_groups=4,eval_images=8,eval_groups=3,eval_is_blind=False,new_positive_labels=0,
        source_open_images=400,run_images=292,entry_conditions=301,unresolved_old_entries=108,
        training_actual={v:{k:actual_training[v][k] for k in ['steps','training_seconds','stop_reason','exit_code']} for v in actual_training},
        main_common_additional_step=pair['main_comparison_step'],training_shared_hashes_exact=pair['shared_input_target_masks_hashes_exact'],
        common_training_response={v:dict(loss=train[v]['macro_total_loss'],center_loss=train[v]['macro_center_loss'],responses=train[v]['response_summary']) for v in ['B','C','D']},
        evaluation=ev['results'],output_counts=counts,batch_seconds=batch['seconds'],postprocess_seconds=summ['seconds'],
        batch_execution=dict(successful_forward_calls=301,aborted_attempt_forward_calls=1,total_forward_calls_including_retry=302,
            logging_only_repair=repair,numerical_or_model_repair=False),
        photometry_diagnostic=photo,visual_review=visual,
        full_search_numeric_diagnostic={k:v for k,v in response_stats.items() if k!='rows'},
        display=dict(raw_gray_unchanged=True,heat_range=[0,1],per_image_stretch=False,path_painted_into_response=False),
        explanation=['The400 gallery images are not400 training examples: frozen training remains10 images/4 source groups. The left L_run case source group is absent from training; the right101517 group contributes2 other training images.',
            'Original gray retains the visible rope, but foreground/background contrast polarity and periodic spool interference change along its projection. Old B also creates off-axis parallel responses on clear free portions.',
            'Existing sparse center supervision leaves97.03% unknown. Engineering neighboring-rope negatives are only742 model pixels in4 images; added positive path length was zero. Background pixel area is not its loss weight because groups are separately normalized.',
            'Earlier matched A/B data showed suppression of explicit negatives together with loss of true partial-center response. C tests gentler continuation/negative weighting as a combined change; D-versus-C isolates the auxiliary LCN input.',
            'LCN increases weak texture and background/compression patterns too. It does not supply missing target identity labels or prove that a visible peak belongs to the entry rope.'],
        limits=['All metrics concern frozen partial reference/engineering proxy domains. No complete same-rope identity accuracy or near-universal clarity rate is established.',
            'Known-entry and automatic-entry cases are distinct. Three of eight historical AUTO inputs remain unavailable; the400 gallery denominator still contains108 unresolved entries.',
            'H6 downstream thresholds remain fixed from B and have not been recalibrated for C/D. Source-track/no-cross checks only verify geometry, not target identity.',
            'No new observed positive labels were added. Remaining weak/crossed/neighbor responses cannot be repaired by displaying brighter colors or filling predicted paths.'],
        overall_finding='Partial-reference target response increases, but weak-domain coverage is not resolved and background/multiple peaks also increase. No general net identity gain or near-universal clarity is established; retain C as an experimental comparison only.',
        original_or_frozen_history_changed=False,production_model_replaced=False,devices_connected=False,reserved_access=False,
        fixed_gallery_reproduction=reproduced,receipts=receipts)
    dump(HERE/'experiment_report.json',report)
    lines=['入口条件目标绳热力图 H7 / IR-H7.0.0','',f'四栏图库：{out/"四栏检测结果"}',f'新旧热图三联：{out/"原图_旧B_新热力图"}',
        '',f'实际292/292图片、301个自动入口条件的新网络前向；输出分类{counts}，不是真实准确率。',
        f'训练10图/4组，评价8图/3组仍隔离；第750步B同起点续训C/D，主对照共同追加{pair["main_comparison_step"]}步。训练侧选择{variant}，不是评价侧选优。',
        'C保留原灰+入口，把工程弱负域权重从邻绳.5/背景.1减至.1/.05，使用lr.0002续训；中心阶段/高低平衡、ignore与全部标签不变。',
        'D与C相同，另加LCN辅助通道；保留原两通道计算支路并拷贝B权重，新增卷积零初始化，起点输出与B等价。原支路权重也参与续训，并非冻结。LCN在975×2048完整搜索域上算sigma16（对应原32px）、truncate4/kernel129、std下限2.5/255、固定clip±3；真实0灰度保留。',
        'LCN在非负原灰光度增强后计算，标准化/残差不做gamma。原灰通道保留；没有逐图拉伸模型输出，也没有将H6路径反画进热图。',
        '两臂共享原灰、种子、目标和所有有效域增强哈希逐步核验；小检查权重丢弃。C对旧B包含续训和权重共同变化，不能把全部变化单独归给某个因素。',
        '', '当前原因证据：左L_run来源没有进入旧训练；右101517来源有2张训练图。绳体叠到卷面后亮暗相位及周期纹理改变，旧B在清晰自由段也有假旁峰。97.03%输入域仍是ignore，新增邻负只有742模型px、新正中心长度0；400张图库不等于400张完整监督。',
        '', '以下仅为已有部分可信中心与明确背景工程代理的响应统计：中心先图内按弧长加权、再按图宏平均；背景按工程代理区域宏平均。它们不代表整根识别准确率：']
    for mode in ['CONDITIONAL','AUTO']:
        for arm in ['B','C','D']:
            q=ev['results'][arm][mode];a=q['center_image_macro']['attached'];g=q['background_region_macro']
            lines.append(f'{mode}/{arm}: 后段中心均值={a["mean"]:.4f}, 响应<.2比例={a["fraction_lt_0_2"]:.4f}, ±5px显著强峰覆盖={a["strong_peak_within_5px_response_coverage"]:.4f}, 多显著峰比例={a["multiple_prominent_peaks_fraction"]:.4f}, 明确背景均值={g["mean"]:.4f}；预测可用图{q["available_images"]}/8，后段参考有效图{a["available_images"]}，背景代理区域{q["background_available_regions"]}。')
    b=ev['results']['B']['CONDITIONAL'];c=ev['results']['C']['CONDITIONAL']
    ba=b['center_image_macro']['attached'];ca=c['center_image_macro']['attached']
    lines+=['',f'判读：已知入口的后段强峰覆盖由{ba["strong_peak_within_5px_response_coverage"]:.2%}变为{ca["strong_peak_within_5px_response_coverage"]:.2%}；但低响应比例由{ba["fraction_lt_0_2"]:.2%}变为{ca["fraction_lt_0_2"]:.2%}，多峰由{ba["multiple_prominent_peaks_fraction"]:.2%}变为{ca["multiple_prominent_peaks_fraction"]:.2%}，背景均值由{b["background_region_macro"]["mean"]:.4f}升至{c["background_region_macro"]["mean"]:.4f}。这是目标增强伴随干扰增强，不构成全面改善。',
        'LCN辅助臂D未赢预设训练侧选择；其评价侧后段强峰覆盖也未显示对C的优势。本轮不继续扩大增强幅度、模型或训练预算。',
        '工程背景代理区域内响应>0.5比例的区域宏平均由B0.2732%上升至C3.5687%（D2.7999%）。虽然全搜索域的中高响应减少，局部已知背景干扰仍变强；不能相互抵消或只选有利数字。',
        '阶段标记只是既有视觉位置分段，并非物理接触真值。±5原图像素内强峰覆盖分母是可提取完整±30px法向剖面的可信部分弧长，同一剖面仍允许多峰；不是完整后段覆盖率或路径身份准确率。',
        '入口置零/位移消融证明输出受入口影响，但不能证明受影响后选择了正确绳段。具体远端差异见evaluation/summary.json。',
        '退化反例：已知入口right500的强峰覆盖虽65.34%→70.52%，最长低响应参考段却79.98→259.93原图px；left550最长低段175.84→379.65px；自动入口left580后段均值.3656→.3366，低响应比例34.66%→43.65%。增强例right550已知入口后段均值.5760→.7118、强峰80.83%→92.08%，仍非整段身份验收。逐点证据见evaluation/。',
        '全301条件数值审查均有限、处于0～1且未退化为近常数（std<.01）。全搜索域>0.5比例均值旧B29.78%→C8.48%，>0.8比例0.27%→0.60%；这包含非目标和未知背景，不是目标覆盖或准确率。可见新图并非全局提亮制造清晰。',
        '', '另对两张用户案例执行24次数字光度诊断：B/C/D各原图及固定增益0.8、gamma1.25、水平增益0.6→1.4。全局增益在B/C被标准化基本抵消；C/D对水平非均匀增益的全域响应差异大于B，不能声称照明鲁棒性普遍提升。右图水平增益使1.4337%输入像素裁剪，已记录。该测试不是实际光源、阴影形状或信噪比变化测试。',
        '', '已有部分参考和固定26图对照不能证明几乎所有图完整同根识别成功。用户左视角反例仍未解决，C仅作为实验候选，不替换旧B。具体改善、退化、条件消融及未解决段见evaluation/、visual_review.json、comparison_responses/及diagnosis/。',
        '下一项最值得补的监督是左视角/卷面背景下少量可确认的正确目标短中心段与同处邻绳负段；未知接续保持未知，无需标整个卷面。',
        '未访问9月25日保留集、未读S1新标签、未连接设备、未替换在线模型。原图、旧标签和H6/B冻结文件不变。',
        '', '复跑入口（新目录/新run名，勿覆盖历史）：prepare_protocol.py → prepare_inference.py → checks/run train_heatmap.py → freeze_selection.py → infer_heatmaps.py evaluation及infer_batch_repaired.py → evaluate_heatmaps.py → postprocess_heatmap.py → audit_h7_paths.py。',
        '网络用vision_training/.venv/Scripts/python.exe（PyTorch CPU）；输入准备、图像与审计用vision_prototype/.venv/Scripts/python.exe。没有安装依赖。实际argv/配置/hash/退出码在每步回执及日志。',
        '首H6适配器回放检查误把elapsed_seconds也纳入相等比较，失败差异原样留h6_adapter_replay；排除耗时字段后4图路径/候选逐值一致，非算法修正。',
        '首批量推理已保存1份响应后，在进度日志处因缺json导入退出1；失败输出保留batch_inference_failed_001。infer_batch_repaired.py只补入标准库json再调用冻结推理，旧源码/数值流程/模型不改；重跑301条件退出0，首响应逐值相等。批量总前向302次=失败尝试1+完整重跑301，非301次全程首次成功。']
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (HERE/'README.txt').write_text('\n'.join(lines[:4])+f'\n\n完整实验说明：{HERE/"REPORT.txt"}\n配置/来源/执行证据：protocol.json、selection.json、experiment_report.json。\n',encoding='utf8')
    print('H7 finalized',variant,counts)
if __name__=='__main__':main()
