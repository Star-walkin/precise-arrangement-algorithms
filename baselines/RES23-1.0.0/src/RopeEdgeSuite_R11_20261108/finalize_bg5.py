"""Comparison report from completed, audited records; no accuracy claims."""
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'RopeEdgeSuite_R1_20261107';SESSION='dual_20260923_113745'

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def categories(rows):
    c=Counter()
    for r in rows:
        if r['count_state']!='unknown':continue
        if not r['s1_valid']:why='活动绳支撑不足'
        elif not r['region_valid']:why='卷面证据不足'
        elif 'active_at_or_beyond_reference_edge' in r['count_events']:why='活动绳投影越过参考边界'
        else:why='F3相位或连续观测不足'
        c[why]+=1
    return dict(c)

def main():
    run=ROOT/'runs'/(SESSION+'_refined');oldrun=OLD/'transfer_runs'/SESSION
    receipt=read(run/'receipt.json');audit=read(run/'audit_summary.json');video=read(run/'delivery/video_receipt.json')
    assert receipt['status']=='complete' and receipt['frames']==video['decoded_frames']==audit['frames']==600
    assert audit['checks_passed'];assert read(ROOT/'test_receipt.json')['passed'];assert read(ROOT/'refined_test_receipt.json')['passed'];assert sha(video['path'])==video['sha256']
    old=read(oldrun/'audit_summary.json');a=read(oldrun/'audit_rows.json');b=read(run/'audit_rows.json')
    comp=dict(algorithm='RES-1.1-BG5.1',session=SESSION,frames=600,experiment='boundary-only controlled ablation plus conditional color-fragment refinement',
        old=old,new=audit,old_unknown_first_blocker=categories(a),new_unknown_first_blocker=categories(b),
        newly_usable_region=[i for i,(x,y) in enumerate(zip(a,b)) if not x['region_valid'] and y['region_valid']],
        newly_rejected_region=[i for i,(x,y) in enumerate(zip(a,b)) if x['region_valid'] and not y['region_valid']],
        region_still_unavailable=[r['frame'] for r in b if not r['region_valid']],
        S1_identical=all(x['s1_valid']==y['s1_valid'] for x,y in zip(a,b)),
        caveat='Gate pass rates are coverage, not accuracy. S1/camera are same-frame reused; all downstream evidence recomputed. d was not recalibrated.',
        control_probe=read(ROOT/'control_probe.json'),tests=read(ROOT/'test_receipt.json'),refined_tests=read(ROOT/'refined_test_receipt.json'),
        refinement_probe=read(ROOT/'refinement_probe.json'),first_pass=read(ROOT/'runs'/SESSION/'audit_summary.json'))
    dump(ROOT/'comparison.json',comp)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    fig,ax=plt.subplots(3,1,figsize=(13,8),sharex=True,gridspec_kw={'height_ratios':[1.25,1,2.3]},layout='constrained')
    for j,(rows,label,col) in enumerate([(a,'RES-1.0 卷面','#d58c54'),(b,'BG5.1 卷面','#2f9d84')]):
        vals=np.array([r['region_valid'] for r in rows]);ax[0].bar(np.arange(600)/10,.78*vals,bottom=j,width=.1,color=col,align='edge')
    ax[0].set_yticks([.4,1.4],['RES-1.0 卷面','BG5.1 卷面']);ax[0].set_ylim(0,1.85)
    ax[0].set_title(f"113745 / 全部 600 帧：卷面可用 {old['region_valid']}/600 → {audit['region_valid']}/600（可用率，不是准确率）",loc='left')
    support=np.array([r['s1_valid'] for r in b]);ax[1].bar(np.arange(600)/10,support,width=.1,align='edge',color='#477ca2')
    ax[1].set_ylim(0,1.2);ax[1].set_yticks([.5],['旧版 S1 光学支撑']);ax[1].text(33,1.04,'S1 未改动；部分通过帧仍可能跟错绳',fontsize=10)
    for rows,label,col in [(a,'RES-1.0','#d58c54'),(b,'BG5.1','#2f9d84')]:
        vals=np.array([np.nan if r['display_N'] is None else r['display_N'] for r in rows]);ax[2].plot(np.arange(600)/10,vals,'.-',ms=2,lw=1,label=label,color=col)
    ax[2].set_ylim(-.2,5.5);ax[2].set_ylabel('显示的剩余根数');ax[2].set_xlabel('视频时间 / 秒');ax[2].legend(loc='upper right');ax[2].grid(alpha=.2)
    ax[2].set_title('仅显示 0–5 根；空白也可能是超过五根，或证据不足。没有逐帧根数真值。',loc='left',fontsize=10)
    for axis in ax:axis.set_xlim(0,60);axis.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'comparison_timeline.png',dpi=150);plt.close(fig)
    def rate(v):return f'{v}/600（{v/6:.1f}%）'
    diag=read(ROOT/'baseline_diagnosis.json')[SESSION]['summary']
    lines=[
        '113745 卷面未知问题：RES-1.1-BG5.1 优化与对照报告',
        '',
        '问题与原因',
        f"旧版卷面通过 {rate(old['region_valid'])}；剩余根数未知 {rate(old['count_states']['unknown'])}。这两个状态不同。",
        f"390 帧触发 outer_flange_not_confirmed；其中 {diag['outer_only']} 帧仅因这一项否决，其余局部卷面门槛已经通过。",
        '原逻辑要求黄色外轮缘在 x=d 附近有充分亮度边缘，并据此否决整片卷面。然而计数起点实际是下方黄灰交界。',
        '外轮缘受强反光、褶皱、局部缺口影响，在 d 附近常无可靠点；31.6 秒例中，外轮缘候选甚至落到活动绳上。',
        '旧版还有 28 帧黄色区域面积不足，以及 d 两侧支撑不足、边界残差过大等原因（原因可重叠）。',
        '黄色不足主要来自强反光将区域切成两个大块，而旧版仅保留最大块。28.0 秒两块面积约 12943、12369（1/4 分辨率像素），单块均不足 13824 的面积门槛。',
        'S1 本身有 291 帧未通过光学支撑，33.6 秒、52.6 秒有跟到卷面纹理的案例，卷面改善不能自动解决这些问题。',
        '',
        '实际修改',
        '1. 外轮缘定位与卷面可用性分开。外轮缘候选不再决定 F3 是否执行，也不再把候选线下的反光区补画为黄色。',
        '2. 用同一帧中的黄色连通块与卷面顶部观测点的上下邻接关系确认卷面，并要求 d 两侧也有黄色邻接证据。',
        '3. 保留顶部支撑比例、局部残差、d 两侧观测及计数邻域覆盖率的门槛。远处长缺口排除，不以拟合补成观测。',
        '4. 鲁棒整体曲线加平滑局部残差修正；ROI 裁剪、显示边界及计数起点共用同一曲线，并明确加入 x=d 的精确节点。',
        '5. 视频分别显示卷面可用/证据不足、活动绳状态及根数未知的具体原因。',
        '6. 对最大黄色块面积不足的帧，关联同一高度带中足够大的当前黄色块；不额外涂白色间隙。合并后仍执行所有边界、d 邻域和纹理门槛。',
        '',
        '全视频结果（600 帧，60 秒，10 fps，零跳帧）',
        f"卷面可用：{rate(old['region_valid'])} → {rate(audit['region_valid'])}",
        f"卷面未通过：{rate(600-old['region_valid'])} → {rate(600-audit['region_valid'])}",
        f"恢复卷面帧数：{len(comp['newly_usable_region'])}；新增拒绝：{len(comp['newly_rejected_region'])}。",
        f"分阶段卷面可用帧：RES-1.0 {old['region_valid']} → BG5 {comp['first_pass']['region_valid']} → BG5.1 {audit['region_valid']}。黄色分块补充实验恢复 {comp['refinement_probe']['valid']}/28 帧，其余保留拒绝。",
        f"根数未知：{rate(old['count_states']['unknown'])} → {rate(audit['count_states']['unknown'])}",
        f"0–5 根范围有数值显示：{old['displayed_count']} → {audit['displayed_count']} 帧；这不是实际五根内召回率。",
        f"当前版本剩余未知的首要阻断项：{json.dumps(comp['new_unknown_first_blocker'],ensure_ascii=False)}",
        f"新版卷面拒绝原因（可重叠）：{json.dumps(audit['region_reasons'],ensure_ascii=False)}",
        '',
        '验证与复现范围',
        f"全部帧逐帧验证：原图像素绑定、前端同帧来源、F3 门控、点归属唯一、每条曲线单次计数、显示/计数同源、预测寿命和误差门槛。审计通过。",
        f"{comp['tests']['tests']+comp['refined_tests']['tests']} 项测试通过，包括去黄色、遮住 d 附近、空白帧、伪外轮缘、禁止沿用历史掩膜、精确曲线节点回归、分块合并及远处黄色干扰物排除。",
        f"另一段 111143 视频等间隔 21 帧的卷面回归：旧版 {comp['control_probe']['old_valid']}/21，新版 {comp['control_probe']['new_valid']}/21；没有重跑该视频的完整 F3。",
        '开发样本与全视频来自同一条 113745 视频，本实验是修复回归，不是独立泛化准确率测试。',
        '为隔离卷面改动，逐帧复用 RES-1.0 的 S1 与相机连续性结果，并核对源视频及每帧像素哈希；卷面、F3、曲线、计数、贴边全部重新计算。',
        'BG5.1 在第一轮结果上只重新计算 28 个黄色面积失败帧的空间检测；其他帧复用经过哈希核对且分支不变的第一轮空间结果。计数及事件窗口对 600 帧从空状态按时间顺序全部重放。逐帧审计验证未修改分支的空间数据完全相同。',
        '未增加时间保持以掩盖卷面失败；未调整 BG4 的 0.3 秒预测期限；未重标定 d=2238.7。',
        '卷面可用率不是边界定位准确率；没有人工完整边界/计数真值。黄色可见交界也不等于已标定的实体接触边界。',
        '29.9 秒的左侧入绳交叠区，黄灰边界候选仍会受到活动绳上沿影响；d 附近与该区分开验证。不能把计数邻域可用理解为整圈被遮挡的真实边界都已经恢复。',
        '计数仍存在局部回升，详情见审计中的 display_step_diagnostics；本次没有修改计数滤波参数，也没有据可用率提升宣称根数更准确或严格单调。',
        'S1 丢失/误跟及跨视频 d 标定仍是后续剩余根数的限制，本版不声称解决。',
        f"第一轮下游重算耗时 {receipt['first_pass_elapsed_s']/60:.1f} 分钟；补充实验与第一轮后半段并行，耗时包含等待，不作为独立性能指标。复用前端、离线处理；视频不是实时处理性能演示。",
        '',
        '基准与产物',
        'RES-1.0 冻结文件与原视频保持不变。新实现：region_bg5.py、region_bg5_refined.py；主程序：run_bg5.py、run_refinement.py；测试：test_bg5.py、test_refined.py。',
        '运行：.venv_lc1/Scripts/python.exe 05_派生与实验结果/05_派生与实验结果/RopeEdgeSuite_R11_20261108/run_bg5.py --workers 8',
        '随后运行同目录 run_refinement.py，再用 audit_bg5.py / render_bg5.py 对 dual_20260923_113745_refined 审计和渲染。',
        '输出目录已存在时程序拒绝覆盖；再次实验请使用独立版本/目录。',
        f"完整视频：{video['path']}",
        f"逐帧记录：{run/'frames'}",
        'comparison.json 保存逐项结果；comparison_timeline.png 显示状态随时间变化。',
        'pilot60 是首轮试验；其中一帧发现显示采样曲线与计数起点差异，已用精确 d 节点修正，正式 600 帧运行使用修正后的代码。',
    ]
    (ROOT/'优化报告.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    files=[*ROOT.glob('*.py'),ROOT/'优化报告.txt',ROOT/'算法说明.txt',ROOT/'comparison.json',ROOT/'comparison_timeline.png',ROOT/'test_receipt.json',
           ROOT/'control_probe.json',ROOT/'baseline_diagnosis.json',ROOT/'refined_test_receipt.json',ROOT/'refinement_probe.json',ROOT/'color_diagnostic.json',ROOT/'visual_review.json',run/'binding.json',run/'receipt.json',run/'audit_summary.json',
           run/'audit_rows.json',run/'compact.json',run/'windows.json',run/'delivery/video_receipt.json',Path(video['path'])]
    files+=list((run/'delivery').glob('preview_*.jpg'))
    dump(ROOT/'delivery_manifest.json',dict(algorithm='RES-1.1-BG5.1',files=[dict(path=str(p.relative_to(ROOT)),size=p.stat().st_size,sha256=sha(p)) for p in files],
        absolute_accuracy_validated=False,baseline_preserved=True))
    print(json.dumps({k:comp[k] for k in ['algorithm','frames','S1_identical','new_unknown_first_blocker']},ensure_ascii=False))

if __name__=='__main__':main()
