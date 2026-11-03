"""Summarize actual completed data, train, evaluation and native gallery evidence."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[1]
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    data=load(HERE/'dataset/statistics.json');sel=load(HERE/'selected_checkpoints.json');ev=load(HERE/'evaluation/summary.json')
    audit=load(HERE/'audit_repair_receipt_002.json');assert audit['exit_code']==0
    arms={b:load(HERE/'branches'/b/'training_receipt.json') for b in ['D0','D1']};small=load(HERE/'smallcheck/receipt.json')
    render=load(HERE/'render_receipt.json');assert render['exit_code']==0 and len(render['rows'])==292
    review=load(HERE/'final_visual_review.json');assert review['actual_review_completed']
    changes={}
    lines=[]
    for b in ['D0','D1']:
        changes[b]={}
        for mode,v in ev['summary'][b].items():
            g=v['path_geometry'];c=g['selected_reference_coverage']
            row=dict(all=c['all']['coverage_5px'],overcoil_attached_proxy=c['attached']['coverage_5px'],edge_near=c['edge_near']['coverage_5px'],
                rejected=g['rejected_images'],target_response=v['target_center_macro_image'],neighbor_delta=v['target_neighbor_delta_macro_pair'])
            changes[b][mode]=row
            lines.append(f"{b} {mode} 全参考{100*row['all']:.2f}% 后段视觉代理{100*row['overcoil_attached_proxy']:.2f}% 近边{100*row['edge_near']:.2f}% 拒识{row['rejected']}/8")
    fit=load(HERE/'training_fit_comparison.json')['models']
    report=f'''活动绳自动弱监督数据扩充 D0与D1 对照  IR-DATA-D1.0.0
启动日期2026年11月2日；完成日期2026年11月3日（北京时间）。运行目录保留启动日期。

主要交付：{HERE/'final_292'}
恰好292张唯一源图的双栏无损PNG。左侧4096×3000原彩像素不改，右侧为本轮D1第{sel['common_step']}步的真实原始中心响应。
右侧不含路径、种子、目标mask或补线；固定0～1 TURBO，域外中性灰。301个入口分别保存float32，主展示按预先冻结E1，不合并热图，不挑最好入口。

数据和质量
实际检查24个候选源图，新增{data['new_unique_images']}图、{len(data['new_groups'])}个采集来源组，旧10图继续保留。训练来源7组，历史8评价图3组隔离，无交集。
接受{data['segment_status_counts'].get('accepted_weak',0)}个自动弱中心片段，共{data['new_center_length_original_px']:.1f}原图px；{data['segment_status_counts'].get('uncertain',0)}段未知，0段另列rejected，0新增独立人工标签。
视觉阶段长度：{json.dumps(data['new_center_length_by_visual_phase'],ensure_ascii=False)}。阶段为自由/投影重叠等视觉位置代理，左右记录略有细分口径；它们不确认三维接触。没有可信阶段4边缘终点标签，也没有确认真正弯后整条同根接续。
采用原图上下文先看，再有限A7原图候选初始化、原生灰度双侧边缘法向±4px精修和逐图原生叠图复核。未用A/B热值作为准入门槛，未把旧预测直接当答案。
新增候选按允许来源内分散时间分位选取；主困难是投影进入卷面背景后的低对比与邻圈混淆。统计中的156段是局部精修片段，不能当156个独立接续事件。所有24图均只有部分有效中心，没有完整困难转弯或边缘身份真值，不能宣称这类覆盖已经补齐。
触及精修走廊边界、低对比、遮挡和身份不清尾段排除。中心sigma仍3原图px，局部有效半径6px；其余均ignore，未标成背景，不伪造主体mask。没有重新增加B的独立大面积负监督项，竞争邻圈确认数为0。
原先可信训练中心附近的定位一致性诊断见annotation_calibration.json；它使用已知参考初始化走廊，不是独立自动身份准确率，同一模型复核也不等于独立验证。

实际学习与控制
均从A750出发，旧EntryUNet base8、灰度+短入口、完整搜索框、0.5缩放、每裁框均值/std下限0.08及增强保持不变。
解冻全部135897参数，仅旧stage_balanced_mse，不训练关系网、不改LCN、beam或轨道规则。AdamW新优化器lr0.0002，两组相同。
独立小检查{small['steps']}步，宏平均有效域损失{small['before']['macro_loss']:.6f}降至{small['after']['macro_loss']:.6f}，通过后丢弃，正式重新载入共同A750。D0={arms['D0']['steps']}步/{arms['D0']['training_seconds']:.1f}秒，D1={arms['D1']['steps']}步/{arms['D1']['training_seconds']:.1f}秒。
实际设备CPU，PyTorch2.14.0+cpu、Python3.14、8线程、batch1，每组预定最多600步或2400秒，均达到600步。使用既有两个虚拟环境，未安装依赖。
主比较取共同保存第{sel['common_step']}步，与评价/图库观感无关。D1实际采样{arms['D1']['sampling_by_origin']}；共同旧图更新的图像和增强输入哈希核验{sel['shared_old_update_inputs_verified']}次。
网络参数变化及梯度、每步样本ID、来源、有效像素、耗时和真实退出记录均保存，新增图确实进入训练，而非使用冻结特征缓存替代前向。
训练配置保留从历史配置继承的未调用说明字段，例如旧800/1200步budget_policy、旧continuation_source及lambda_neighbor/background；这些不参与本轮训练。实际source_checkpoint、600步/2400秒与stage_balanced_mse由本轮代码和日志明确记录，不将这些旧字段解释为本轮执行参数。

固定评价
{chr(10).join(lines)}
预测和路径全部先保存并哈希，再读取原8张可信部分参考。路径统一使用旧冻结connect.py与同path_config；主交付不画这些路径。
响应、目标/邻圈2处工程代理、后段及近边覆盖、最长缺口、首次持续60px偏离和拒识见evaluation/summary.json及per_case_metrics.csv。
AUTO三张既有入口失败仍留8分母；CONDITIONAL是同8张图的给定短入口诊断，不是额外8独立样本。
旧/新训练拟合见training_fit_comparison.json，自动弱标注不能当作独立测试真值。完整弧段与真实末端成功率仍未定义。
D0→D1在旧图训练目标的宏平均阶段均衡MSE为{fit['D0']['old']['macro_loss']:.6f}→{fit['D1']['old']['macro_loss']:.6f}（退化）；新增弱标签图同一损失为{fit['D0']['new']['macro_loss']:.6f}→{fit['D1']['new']['macro_loss']:.6f}（拟合改善）。
新图有效域只到中心6px，最外高斯目标约0.135，没有target<0.1像素，因此该诊断低响应均值为null而不是0；实际损失高/低分组使用0.5阈值且都有有效样本。窄有效域与没有确认邻圈负例，限制了目标选择性的学习。
两处工程邻圈代理的目标减邻圈均值：AUTO {changes['D0']['AUTO']['neighbor_delta']:.4f}→{changes['D1']['AUTO']['neighbor_delta']:.4f}，CONDITIONAL {changes['D0']['CONDITIONAL']['neighbor_delta']:.4f}→{changes['D1']['CONDITIONAL']['neighbor_delta']:.4f}，均退化。目标变强不能替代目标与邻圈区分改善。

实际目视与解释
{review['findings']}
更多数据改变了训练覆盖与来源，也同时引入自动弱标签、窄监督域和采样混合变化；不能把全部差异解释成单纯图像数量效应。
本轮支持新增外观可被学习、部分参考覆盖可改善；不支持几乎所有热图已能明显独立识别目标。下一步优先补少量原图可确认的投影后接续及近邻竞争负例，在相同评价下检查选择性，避免直接扩大模型或把未知尾巴改成背景。
本轮未根据评价追加训练或重选漂亮模型。D1无论改善或退化均作为292主交付，未换回D0/A750/B750。

文件和复现
protocol.json，dataset/index.json/statistics.json，labels_left/labels_right为冻结源图、中心和质量状态；原图路径、哈希、原生裁图、叠图及逐图复核记录均保存。
branches/D0与D1含配置、环境、初始/每100步/最终权重、日志、梯度与采样；selected_checkpoints.json为共同步选择。
inference/gallery/D1含301原生浮点及元数据；final_292和render_receipt.json绑定292唯一图与实际模型/入口/用途。
final_292_index.json按源图哈希标记是否实际参与D1训练，并进一步区分历史开放评价来源、同训练来源未用帧和其他开放展示来源。PNG脚注只粗分训练/非训练展示；补充索引提供细分，不重写已审计图片。展示图不自动成为独立测试。
run_pipeline.ps1保留本次实际命令；execution_receipts.json记录退出码与耗时。重跑应使用新运行目录，勿覆盖本轮冻结文件。
重导出应从已选D1浮点使用render_export.py；现有final_292不会被覆盖，应改用新导出目录或复制脚本到独立重导出运行。
独立audit.py对全部PNG的原图、右栏逐像素重建、301浮点、数据及模型哈希均核验。结构检查不等于真实身份准确率。
首轮审计在入口seed逐位重建失败：OpenCV精确L2距离变换1线程与原生成2线程的远距离末位存在差异，seed最大差9.313225746154785e-9；全部24图采用原2线程重建后逐位一致。修正仅复现实验原线程条件，不放宽阈值，不修改冻结标签、模型或热图。audit_failed_001_*保留失败源码/receipt/log，独立修复重跑另存；execution_receipts.json中初次exit1保持真实。
原图、旧A/B750与P1/H6/R1保留；无设备连接、无保留集或S1标签访问，不替换线上模型或历史基线。
'''
    (HERE/'REPORT.txt').write_text(report,encoding='utf-8')
    summary=dict(version='IR-DATA-D1.0.0',started_date_local='2026-11-02',date_local='2026-11-03',source_count=292,entry_conditions=301,data=data,
        training={b:{k:arms[b][k] for k in ['steps','training_seconds','sampling_by_origin','sampling_by_source_group','initial_state_sha256','final_state_sha256']} for b in arms},
        main_step=sel['common_step'],evaluation=changes,output_directory=str(HERE/'final_292'),audit_exit_code=0,
        report_sha256=sha(HERE/'REPORT.txt'),labels_are_automatic_weak=True,new_independent_human_truth=0,complete_identity_success=None,
        no_device=True,no_reserved=True,baseline_replaced=False)
    dump(HERE/'experiment_report.json',summary)
    note=f"\n\n## 2026-11-03 自动弱监督数据扩充 D1\n\n[292张D1原始热图](data_expansion_d1_20261102_v1/final_292/) · [实际报告](data_expansion_d1_20261102_v1/REPORT.txt)。新增24张自动弱标注图、3来源组，旧10图保留；D0/D1从A750全参数各{arms['D0']['steps']}/{arms['D1']['steps']}步，共同第{sel['common_step']}步。AUTO部分后段5px参考覆盖25.88%→35.99%，但邻圈选择性及旧图拟合退化；不是完整身份准确率。292原像素双栏不叠路径，301浮点独立保存。未知尾段忽略，完整接续/边缘未有独立真值；不替换旧基线，无设备或保留集访问。\n"
    nav=BASE/'README.md'
    if b'data_expansion_d1_20261102_v1' not in nav.read_bytes():
        with nav.open('ab') as f:f.write(note.encode('utf-8'))
    print(json.dumps(dict(output=str(HERE/'final_292'),data_images=data['new_unique_images'],common_step=sel['common_step'],evaluation=changes),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
