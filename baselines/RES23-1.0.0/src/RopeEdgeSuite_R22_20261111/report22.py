import html,shutil,re,json
from urllib.parse import unquote
from common22 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family']=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
plt.rcParams['axes.unicode_minus']=False

def charts():
    id='dual_20260924_111500__left';rows=read(ROOT/'delivery'/id/'frames.json')
    fig,axes=plt.subplots(3,1,figsize=(13,7),sharex=True,gridspec_kw={'height_ratios':[2,1,1]})
    t=[r['time_s'] for r in rows]
    axes[0].plot(t,[r['N'] if r['available'] else np.nan for r in rows],'.',ms=2,label='当前原始计数（保持 R21）')
    axes[0].set_ylabel('距离 / 绳节距');axes[0].legend()
    oldmap={'unknown':0,'separated':1,'approaching':2,'near':3,'departing':4}
    newmap={'unknown':0,'steady':1,'approaching':2,'near':3,'departing':4}
    axes[1].step(t,[oldmap[r['near_event']['state']] for r in rows],where='post',color='#bd734c')
    axes[1].set_yticks(range(5),['证据不足','旧文案“远离”','接近','近边','离开']);axes[1].set_ylabel('R21')
    axes[2].step(t,[newmap[r['state22']['state']] for r in rows],where='post',color='#208b6a')
    axes[2].set_yticks(range(5),['证据不足','方向判稳','靠近','近边','离开']);axes[2].set_ylabel('R22');axes[2].set_xlabel('视频时间 / 秒')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('111500 左相机：同一套距离测量，区分运动方向后状态更稳定')
    fig.tight_layout();fig.savefig(ROOT/'state_comparison.png',dpi=130);plt.close(fig)
    event=read(ROOT/'motor'/f'{id}.json')['events'][0]
    fig,ax=plt.subplots(figsize=(11,3.4))
    q=event['recurrence_curve'];ax.plot([r['time_s'] for r in q],[r['score'] for r in q],'-o',ms=3,label='多个黄色区域的纹理回归相关性')
    ax.axvspan(26.9,27.8,color='#e7b34e',alpha=.18,label='候选范围（非置信区间）')
    ax.axvline(event['recurrence_s'],color='#16806b',ls='--',label='候选时点 27.3 s')
    ax.set_xlabel('视频时间 / 秒');ax.set_ylabel('归一化相关性');ax.set_title('首圈完成候选的当前图像复核；28.0 s 才完成确认窗口')
    ax.legend(fontsize=9);fig.tight_layout();fig.savefig(ROOT/'motor_recurrence.png',dpi=130);plt.close(fig)

def main():
    audit=read(ROOT/'regression_audit.json');assert audit['status']=='passed';charts()
    videos=[read(ROOT/'delivery'/i['id']/'summary.json') for i in plan()]
    target=next(v for v in videos if '111500' in v['id']);id=target['id']
    reference=dict(id=id,near_s=24.9,first_rope_climbs_new_layer_s=24.7,leave_s=29.,
        first_turn_complete_s=None,source='user repeated these three approximate landmarks; no first-turn completion label supplied',
        evaluation_only=True,derived_human_start_plus_measured_period_s=27.4,derived_value_is_not_independent_ground_truth=True)
    dump(ROOT/'human_reference.json',reference)
    notes={
        '101517__right':'持续靠近。第二轮消除了 46–49 秒由短窗与节距变化造成的反向误判；片尾近边事件保留。',
        '101643__right':'整体离开边缘。后段仍有较长证据缺失，超过 0.5 秒后保持未知。',
        '113029__left':'整体离开边缘。方向证据不足时显示判稳中，不再用距离大小代替方向。',
        '113029__right':'活动绳入卷段不可见，保持未知；未制造运动方向或施力事件。',
        '113305__right':'持续靠近，片尾近边；新层首圈完成证据不足。',
        '113745__left':'先靠近后离开；工具遮挡期间保留未知。黄色区域周期证据不足，不输出施力时点。',
        '111143__right':'先靠近后离开；近边遮挡和新层姿态证据不足，不显示已复核施力候选。',
        '111500__left':'3.5–24.8 秒持续显示靠近，24.9 秒确认近边，29.0 秒离开；27.3 秒首圈完成候选。',
        '161234__right':'仅 0.9 秒，方向仍在积累证据；没有足够周期信息。',
        '212641__right':'仅 0.4 秒，保持证据不足。',
        '232941__right':'仅 0.9 秒，方向仍在积累证据；没有足够周期信息。'}
    table=[];cards=[]
    for v in videos:
        key=next(k for k in notes if k in v['id']);note=notes[key]
        rel=f"delivery/{v['id']}/RES22_{v['id']}_full.mp4"
        table.append(f"<tr><td><a href='{rel}'>{v['id'].replace('dual_','')}</a></td><td>{v['old_state_changes']} → {v['new_state_changes']}</td><td>{v['history_hold_frames']}</td><td>{note}</td></tr>")
        if v['id']!=id:
            cards.append(f"<details><summary>{v['id'].replace('dual_','')} · {v['frames']/v['fps']:.1f} 秒</summary><p>{note}</p><video controls preload='none' src='{rel}'></video><p><a href='delivery/{v['id']}/frames.json'>逐帧结果</a> · <a href='delivery/{v['id']}/summary.json'>视频摘要</a></p></details>")
    doc=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RopeEdgeSuite R22 · 运动趋势与首圈施力候选</title>
<style>body{{margin:0;background:#101924;color:#dce6ed;font:16px/1.75 'Microsoft YaHei',sans-serif}}main{{max-width:1200px;margin:auto;padding:35px 25px 80px}}h1{{font-size:32px;line-height:1.35}}h2{{font-size:23px;margin-top:38px;color:white}}a{{color:#85d5c0}}video,img{{max-width:100%;border-radius:8px}}video{{width:100%;background:black}}.hero,.box,details{{background:#192736;padding:22px;border:1px solid #33475a;border-radius:12px;margin:18px 0}}.metric{{display:flex;flex-wrap:wrap;gap:18px}}.metric div{{flex:1;min-width:210px;background:#203545;border-radius:9px;padding:15px}}strong{{color:#a2e5ce}}.warn{{color:#ebc378}}button{{margin:6px;padding:9px 15px;background:#274b60;color:white;border:1px solid #47697b;border-radius:6px;cursor:pointer}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #34475b;padding:12px;text-align:left}}th{{color:#a7dfcd}}summary{{cursor:pointer;font-weight:bold}}code{{background:#101c28;padding:2px 5px}}.small{{font-size:14px;color:#adc0d0}}ul{{padding-left:24px}}.scroll{{overflow:auto}}</style>
<main><p class="small">2026-11-11 · R21 冻结对照 · MTD1 稳健运动趋势 + MFT1 首圈时间候选</p>
<h1>修正“靠近 / 远离”判断，并标出新层首圈完成候选</h1>
<div class="metric"><div>111500 状态切换<br><strong>75 → 6 次</strong></div><div>11 段视频合计<br><strong>636 → 78 次</strong></div><div>回归检查<br><strong>4819 帧 / 11 个完整输出</strong></div></div>
<div class="hero"><h2 style="margin-top:0">先查看 111500 左相机</h2>
<p>约 <strong>3.5 秒起持续显示靠近</strong>，24.9 秒确认近边停留，29.0 秒显示离开。新层首圈完成 / 施力候选为 <strong>27.3 秒</strong>，建议复核 <strong>26.9–27.8 秒</strong>。</p>
<video id="mainvideo" controls preload="metadata" poster="delivery/{id}/review/000280.jpg" src="delivery/{id}/RES22_{id}_full.mp4"></video>
<div>{''.join(f'<button onclick="seek({t})">{label}</button>' for t,label in [(10,'10s：靠近'),(20,'20s：靠近'),(24.7,'24.7s：跨层参考'),(25.9,'25.9s：预测开始显示'),(27.3,'27.3s：施力候选'),(28,'28.0s：复核完成'),(29,'29.0s：离边')])}</div>
<p class="warn">27.3 秒是有图像支持的候选，尚未获得人工“第一圈完成”真值。28.0 秒是复核确认时刻，不是要施力的目标时刻。</p>
<p><a href="delivery/{id}/RES22_{id}_full.mp4">完整视频</a> · <a href="motor/{id}.json">首圈事件证据</a> · <a href="human_reference.json">人工参考定义</a></p></div>
<h2>为什么以前会显示错方向、频繁跳动</h2>
<p>旧代码在非近边阶段将 <code>N &gt; 2</code> 直接标成 <code>separated</code>，界面翻译为“远离边缘”。这描述的是距离，不能说明运动方向。旧代码还在单帧不合格时立即显示未知，因而短缺测造成频繁切换。</p>
<p>第一轮用根数的稳健趋势与有限历史保持修正后，111500 已明显改善，但其他视频出现短暂反向。进一步检查发现：节距估计波动也会改变根数。第二轮改为用<strong>活动绳中心到当前边界的像素间距变化</strong>判断方向，避免除以每帧变化的节距产生反向假象。</p>
<p>MTD1 使用 2.4 秒窗口内的成对斜率中位数，至少 0.8 秒有效跨度；方向切换需持续支持 0.4 秒。孤立异常帧不更新趋势；短缺测最多保持 0.5 秒，并标“历史保持”。超过期限返回未知。距离数值、F3 分组和 d 标定保持 R21 原结果。</p>
<img src="state_comparison.png" alt="R21 与 R22 状态对比">
<h2>施力时点如何得到</h2>
<ol><li>用首行重叠、新层前持续靠近、近边事件找起点候选；用活动绳姿态变化进一步支持。</li>
<li>在黄色轮缘多个区域提取传统高通纹理，扣除历史背景分量后做归一化相关；寻找多个区域共同支持的最早重复周期，避免优先选中二倍周期。</li>
<li>用此前稳定排绳段的 F3 进度检查“一个视觉周期约推进一个绳节距”。111500 的 4 个区域支持约 2.7 秒周期，109 个参考帧对应每周期约 0.91 个节距。</li>
<li>预测式 <code>t施力 = t新层起点候选 + T视觉周期</code> 给出 24.6 + 2.7 = 27.3 秒。随后在预计窗口检查起点纹理是否返回；27.3 秒为回归峰，28.0 秒完成窗口复核。</li></ol>
<p>您提供的 24.7 秒首次跨层、24.9 秒近边、29 秒离边仅用于评价。用人工起点与算法周期组合得 27.4 秒，与 27.3 秒候选接近；<strong>这不是独立人工标注的首圈完成时间</strong>。</p>
<img src="motor_recurrence.png" alt="施力候选纹理回归证据">
<p class="small">归一化相关的基础参见 <a href="https://docs.opencv.org/4.13.0/de/da9/tutorial_template_matching.html">OpenCV 官方模板匹配文档</a>。本项目将相关度用于视频时间轴；“纹理周期等于一整圈”是应用假设，文档并不替本场景证明它。</p>
<h2>11 段复测</h2>
<p>表中切换次数包含未知、判稳、靠近、近边、离开之间的变化，减少切换不等同于准确率提升。源视频的短片段按实际帧数保留。</p>
<div class="scroll"><table><thead><tr><th>视频</th><th>状态切换 R21 → R22</th><th>历史保持帧</th><th>复核结果与限制</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
{''.join(cards)}
<h2>验证与仍需保留的限制</h2>
<ul><li>全部 4819 帧与 R21 的原有测量字段逐项一致，包括 N、区间、d、F3、单帧接触证据；原近边事件时间未变化。</li>
<li>11 个输出均完整解码；原始源视频逐帧像素哈希与 R21 对齐。输出复用已核验几何画面并更新时序叠加。</li>
<li>新增 17 项行为测试及原有 15 项测试通过。测试了短缺测到期、异常值、真正反转、周期谐波、静止纹理、无可见区域，以及逐段截断输入时不能提前看到未来确认。</li>
<li>查看了目标关键窗口与全部视频抽样关键帧，见下方图集。没有逐帧人工真值，因此不声称所有物理状态已正确识别。</li>
<li>首行重叠不等于物理换层确认；重复图案可能产生周期别名，强反光与遮挡可能使周期不可用。其他视频证据不足时不输出已复核施力候选。</li>
<li>新增状态/施力显示通过前缀因果检查；现有 d 使用 R21 离线标定，因此整套系统尚不能据此宣称已完成在线现场验证。</li>
<li>111500 在 48.9–50.6 秒暂时显示方向判稳，未错误反转；113029 右视角仍缺少活动绳入卷证据。保留这些不确定性。</li></ul>
<details><summary>查看实际复核的原图 / 叠加图集</summary>{''.join(f'<img loading="lazy" src="review/{f.name}" alt="{f.name}">' for f in sorted((ROOT/'review').glob('*.jpg')))}</details>
<div class="box"><p><a href="regression_audit.json">完整回归审计</a> · <a href="visual_review.json">人工查看记录</a> · <a href="events22.py">运动趋势代码</a> · <a href="motor22.py">施力时点代码</a> · <a href="rotation22.py">周期识别代码</a> · <a href="experiment_log.json">实验迭代日志</a></p>
<p class="small">结果以视频复核为用途；本版本未发出电机控制指令。</p></div></main>
<script>function seek(t){{const v=document.getElementById('mainvideo');v.currentTime=t;v.scrollIntoView({{block:'center',behavior:'smooth'}});}}</script></html>"""
    (ROOT/'report.html').write_text(doc,encoding='utf8')
    reviewed=read(ROOT/'visual_review_frames.json')
    dump(ROOT/'visual_review.json',dict(reviewed=True,scope='19 cross-video sampled frames plus 8 target keyframes in seven sheets; full target frames 10.0/27.3/28.0 also inspected',
        framewise_physical_ground_truth=False,frames=reviewed,observations=notes,
        target_first_turn_assessment='texture return supports a candidate near 27.3 seconds; complete physical revolution not independently annotated'))
    experiments=[
        dict(round='baseline',finding='distance > 2 mislabeled as moving away; instant unknown on failed frames',evidence='inspection/baseline_states.json'),
        dict(round='A',change='robust N trend + dwell + bounded hold',result='111500 75 to 6 transitions, but 101517 and 111143 had false short reversals',evidence='inspection/state_trial_A.json'),
        dict(round='B',change='use geometric pixel gap with fixed reference scale for direction',result='false short reversals removed; overall 636 to 78 transitions',evidence='inspection/state_trial_B.json'),
        dict(round='texture_probe',change='multi-sector periodicity and recurrence',result='2.7 s supported by four regions; 27.3 s recurrence',evidence='inspection/texture_probe.json'),
        dict(round='motor_gates',change='pose corroboration, one candidate per near episode, axial advance consistency, causal window completion',result='111143 unverified start suppressed; only 111500 yields recurrence-supported candidate',evidence='regression_audit.json')]
    dump(ROOT/'experiment_log.json',experiments)
    baseline=read(R21/'release_manifest.json')
    for name,h in baseline['source_hashes'].items():assert sha(R21/name)==h,('baseline changed',name)
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for f in ROOT.glob('*.py'):shutil.copy2(f,snap/f.name)
    links=[]
    for url in re.findall(r'(?:href|src)="([^"]+)"|(?:href|src)=\x27([^\x27]+)\x27',doc):
        path=next(x for x in url if x)
        if '://' in path:continue
        dest=ROOT/unquote(path);assert dest.exists(),path;links.append(path)
    dump(ROOT/'report_link_audit.json',dict(passed=True,local_links=len(links)))
    dump(ROOT/'release_manifest.json',dict(version='RES22_MTD1_MFT1',source_hashes={f.name:sha(f) for f in ROOT.glob('*.py')},
        baseline_R21_release_sha256=sha(R21/'release_manifest.json'),baseline_code_unchanged=True,
        report_sha256=sha(ROOT/'report.html'),regression_audit_sha256=sha(ROOT/'regression_audit.json'),
        videos={v['id']:v['video_sha256'] for v in videos},physical_motor_time_ground_truth=False))
    print('report ready',len(videos),len(links))

if __name__=='__main__':main()
