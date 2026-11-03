"""Finalize real M1+R1 evidence after independent dataset/model/PNG audits."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=HERE.parents[2]

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    audit=load(HERE/'audit_delivery_receipt.json');assert audit['exit_code']==0 and not audit['errors']
    dataset=load(HERE/'dataset/statistics.json');index=load(HERE/'dataset/index.json')
    train=load(HERE/'model/training_receipt.json');small=load(HERE/'smallcheck/receipt.json')
    relation=load(HERE/'relation/model/training_receipt.json');r1audit=load(HERE/'r1_actual_audit.json')
    selected=load(HERE/'selected_checkpoints.json');evaluation=load(HERE/'evaluation/summary.json')
    rendered=load(HERE/'render_receipt.json');assert rendered['exit_code']==0 and len(rendered['rows'])==292
    fit=load(HERE/'training_fit.json')['fit'];manual=[r for r in fit['rows'] if r['origin']=='manual']
    red_mean=sum(r['manual_red_response'] for r in manual)/len(manual)
    high_mean=sum(r['high_response'] for r in manual)/len(manual)
    visual=load(HERE/'visual_review/selection.json');rejected=load(HERE/'visual_review/rejected_selection.json')
    viewed=[(HERE/'visual_review'/f'{i}_native_compare.png') for i in range(1,8)]
    viewed += [(HERE/'visual_review'/f'{i}_context.png') for i in (1,2,5,6,7,8,9)]
    findings=('已实际查看7组原生800×650局部对照（原图、旧D1、新U1、新U1+R1）与7张完整三栏上下文。'
        'L_run585、R_run572等人工训练图中，活动绳斜带保留，邻近卷面周期高响应比旧D1明显减弱；'
        'R_static095和L_static100未入训练来源也观察到部分卷面响应抑制，但弱响应、股纹/机架背景响应仍在。'
        '独立评价来源pair0555仍未显示完整贴面目标，R1可能只留很短前缀。'
        '明确失败：dual101517 pair0075 E1从左工具一侧的既有入口出发，沿卷面顶缘偏离并形成U形横跨多圈至法兰背景；'
        '这是错误路径，不能因有绿色支持标记或高分而视作同根成功。其另一入口只独立保存，未按效果替换主E1。'
        'pair0255 E1及L_stop1069 E1的拒识也实际查看保留；没有强制绿色补线。'
        '本轮提高部分目标与邻圈的响应可分性，但尚不能保证所有目标身份及真实末端。')
    dump(HERE/'visual_review/review_receipt.json',dict(actual_review_completed=True,findings=findings,
         files=[dict(path=str(p),sha256=sha(p)) for p in viewed],
         image_source='actual final292 and historicalD1 outputs; no presentation-generated heat',
         independent_identity_truth=False,checkpoint_or_parameters_changed_after_review=False))
    geometry=[]
    for arm in ('R0','R1_old','R1_new'):
        for mode in ('AUTO','CONDITIONAL'):
            s=evaluation['summary'][arm][mode];c=s['selected_reference_coverage']
            geometry.append(f"{arm} {mode}: 后段{c['attached']['coverage_5px']*100:.2f}%，近边{c['edge_near']['coverage_5px']*100:.2f}%，拒识{s['rejected_images']}/8")
    hm=evaluation['heatmap_summary'];output=Path(rendered['output_directory'])
    current_same=all(sha(x['source_annotation_path'])==x['annotation_sha256'] for x in index['items'] if x['origin']=='manual')
    report=f'''人工中心监督 U-Net＋匹配R1离线实验  IR-M1-R1.0.0  2026-11-03

已实际完成
使用用户人工确认的24图、24条目标中心线及51条明确邻绳红负线，保留旧10图；实际训练U-Net800步与R1 600步，新前向/完整路径推理301个入口条件，导出恰好292张唯一源图PNG。
主图目录：{output}
每张12492×3370，三个4096×3000原生视口：原图 / 新U1原始中心响应 / 新U1＋R1路径候选。中栏完全没有路径线、标签、形态补线或mask遮挡，固定0～1 TURBO；未经身份概率校准。y<1050中性灰未计算。
完整源图像素未缩小；网络原生网格仍2048×975，由固定映射回原图。这不是4096原生网络训练。
青绿色线是算法声称的图像支持，橙虚线是推断/弱支持，紫灰为独立竞争候选；这些颜色均不证明同根身份。拒识保留，主入口始终为冻结E1，不按热值、人工目标或好看选入口，不合并多入口热图。

监督与冻结
人工目标长度45039.916原px，红邻绳17484.611px；目标约20507.753px落在独立SAM2卷面ROI中，已含投影后段监督。按原图身份/哈希冻结24份JSON于label_audit/annotation_snapshot；没有改用户坐标。交付时用户源标注与快照一致={current_same}。
原生6px局部监督管在5图外缘共有33像素相接，无中心线相交。按相同中心sigma3与原网格规则构造模型格时，26个正负矛盾像素均ignore；保留全部24图与51红线。最终有58458模型格红负像素，全部valid且target=0，实际梯度检查通过。
未知/未标区域始终ignore，局部中心监督不是绳体mask；片段端帽不作为真实末端。用户状态字段均partial，阶段字段均默认stage1；未擅改成物理阶段，不能解释成只有自由段。红负与高正均在实际启用的同阶段MSE中，没有被损失跳过。
训练34图来自7来源组，历史评价8图/3来源组隔离。24人工图的6来源组含相邻时刻，同组不是独立工况。292展示图含24人工训练图、208同训练来源未用帧、20历史评价来源图、40其他开放来源，不是292张独立测试。

真实训练
从原A750出发，未从受质疑的自动弱标D1继续。EntryUNet base8、原灰＋独立短入口、全搜索框/标准化/增强保持原流程；标注界面LCN仅辅助人工观察，本轮网络输入未切成LCN。后段人工中心线只用于监督，未作为推理入口或远端提示。
全部135897参数解冻，AdamW lr0.0002/wd0.0001，CPU8线程、batch1。预定小检查100步/300秒，正式800步/2000秒，均达到步数上限，无评价驱动追加。
独立小检查100步/{small['seconds']:.2f}秒，宏平均阶段均衡MSE {small['before']['macro_loss']:.6f}→{small['after']['macro_loss']:.6f}；通过后丢弃小检查权重，正式重置A750。
正式800步/{train['training_seconds']:.2f}秒；旧图400次、人工400次，24图全部实际见到。共同采用最后实际保存第{selected['common_step']}步，不按图库或评价选模型。参数变化、非零梯度、采样/有效域与每步损失均保存。
人工24图最终训练目标高响应均值{high_mean:.4f}、红负均值{red_mean:.4f}，训练宏损失{fit['by_origin']['manual']['macro_loss']:.6f}；旧10图{fit['by_origin']['old']['macro_loss']:.6f}。这些是拟合，不是独立准确率。

匹配R1
197维描述子含新U1的48维有序dec1空间特征与6热值，均与新响应同次前向保存；没有混用A750旧特征。其余原灰剖面/几何/F3轨道模块与搜索参数保持冻结。
在人工路径内部teacher-forced历史下，先用原propose完整池生成候选/统一计算相对rank，再选距目标≤4px且有效的正候选，与距明确红线≤6px、离目标≥8px且valid低值的负候选配对。没有插入正确真值候选，也没有把未知当负。
共369对来自23图/6来源组；L_run1250虽有正确提议，但没有合法红负同池配对，未伪造负例，该图仍参加U-Net。1794个已知正确前缀局部状态中1786含正确候选、8缺失，只是条件局部覆盖，不是全自动完整识别率。
同结构197→64→32→1 MLP从固定初始化训练600步/{relation['seconds']:.3f}秒，训练排序{relation['before']['pair_order_fraction']:.4f}→{relation['after']['pair_order_fraction']:.4f}，损失{relation['before']['loss']:.6f}→{relation['after']['loss']:.8f}。独立重建全部369对描述子逐位相等；不等于身份泛化验证。
本轮R1仅使用上述人工配对，不混旧10关系弱对。图库有精确同图F3轨道，旧8评价源没有精确轨道，存在缺轨分布差异；不据评价调dropout或权重。原R1轨道是软惩罚，仍不保证不跨圈。

独立来源的有限评价
固定三臂同新U1：R0无关系项、R1_old旧冻结权重、R1_new新匹配权重。48项预测全保存并哈希后才读取既有部分参考；3张AUTO入口失败仍在8分母。以下是5原px部分参考弧长覆盖，非完整目标准确率：
{chr(10).join(geometry)}
新R1相对旧R1：AUTO后段略退，CONDITIONAL后段/近边略升，未获一致净增益。主交付仍按预定用新R1，未偷偷换回AUTO分数较高的旧R1。
同两处工程目标/邻圈代理的响应差：旧D1 AUTO {hm['D1']['AUTO']['target_neighbor_delta']:.4f}→U1 {hm['U1']['AUTO']['target_neighbor_delta']:.4f}，CONDITIONAL {hm['D1']['CONDITIONAL']['target_neighbor_delta']:.4f}→{hm['U1']['CONDITIONAL']['target_neighbor_delta']:.4f}。这是有限位置的可分程度改善，不能升级为全图身份正确。
同时有退化：AUTO后段中心均值{hm['D1']['AUTO']['center_response']['attached']:.4f}→{hm['U1']['AUTO']['center_response']['attached']:.4f}，近边{hm['D1']['AUTO']['center_response']['edge_near']:.4f}→{hm['U1']['AUTO']['center_response']['edge_near']:.4f}。不是所有目标响应都变强。
与旧D1比较同时改变了监督来源、明确负例和训练预算，不能把差异单独归因于红线或图像数量。完整同根弧段/真实末端仍没有独立完整真值，不报告全段成功率。

实际目视及失败
{findings}
图库301条件为298前缀/边缘候选、3拒识；固定E1主图为290候选/2拒识，都是产物类别，不是正确率。status_length_px仅计两端同状态的边；mixed边另记，推断长度0不保证无推断点。
下一步优先固定本轮新热图，修自动入口与R1错误接续/同轨保持，保留无正确候选和拒识，而不是继续盲目增加U-Net训练步数。未扩展在线控制、测速或三节点判断。

交付核验与修复记录
render_results.py逐292原图/原热图/右栏栅格核验通过；独立audit_delivery.py另对全292左原像素、中栏四邻点回映/TURBO逐像素检查以及301浮点/新特征/入口/候选坐标/hash全部通过，0错误。
首批10个ruler照片因相机角色解析失败，291成功JSON保持原字节，只修角色取值并补10路径；无重训/新前向，失败源码/receipt/log与修复AST差异保存在repairs/gallery_role_001。
首次评价基线A750已是原图3000×4096响应，脚本误按975×2048检查后退出1，失败目录/source保留；修正仅识别已回映格式。另保留按具体mode的实际入口排除范围修订前评价；最终数值见evaluation/summary.json，路径和模型未改变。
D盘当时仅2.37GB空余，新的dataset/inference/paths经明确junction存C盘；所有批量PNG位于C盘，不搬原始素材或历史目录。代码、训练配置、权重、命令和日志仍在本D盘独立实验目录。
复现实际命令：run_unet.ps1、run_r1.ps1、run_render.ps1与对应execution_receipts；检查audit_delivery.py、dataset_audit_receipt.json、r1_actual_audit.json。重跑写入应新建版本，不覆盖冻结产物。
未连接设备、未访问保留集或S1标签、未改原图/用户标注/历史模型，不替换P1/F3/H6等基线。
'''
    (HERE/'REPORT.txt').write_text(report,encoding='utf-8')
    dump(HERE/'experiment_report.json',dict(version='IR-M1-R1.0.0',date_local='2026-11-03',
         output_directory=str(output),unique_sources=292,entry_conditions=301,manual_images=24,old_images=10,
         train_steps=train['steps'],train_seconds=train['training_seconds'],relation_steps=relation['steps'],relation_seconds=relation['seconds'],
         U1_model=selected['paths']['U1'],R1_portable_path=str(HERE/'relation/model/final_portable.npz'),
         evaluation_summary=evaluation['summary'],heatmap_summary=hm,main_E1_selected=290,main_E1_rejected=2,
         report_sha256=sha(HERE/'REPORT.txt'),dataset_sha256=sha(HERE/'dataset/index.json'),
         audit_exit_code=0,source_use_counts=rendered['source_use_counts'],
         complete_identity_accuracy=None,known_failure='dual101517pair0075E1 wrong entry/path crosses coils toward flange',
         no_devices=True,no_reserved_or_S1=True,old_baseline_replaced=False))
    # A compact browsing index with every source, model, entrance and output SHA.
    dump(HERE/'final_292_index.json',dict(unique_sources=292,entry_conditions=301,rows=rendered['rows'],
         render_receipt_sha256=sha(HERE/'render_receipt.json'),audit_receipt_sha256=sha(HERE/'audit_delivery_receipt.json')))
    print('Finalized292 images, newU-Net800/R1 600, independent audit0; report and index saved.')

if __name__=='__main__':main()
