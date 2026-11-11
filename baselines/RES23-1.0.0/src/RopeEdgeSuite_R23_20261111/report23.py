import html,re,shutil
from urllib.parse import unquote
from common23 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family']=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
plt.rcParams['axes.unicode_minus']=False

def charts(rows,e):
    clock=e['timeline'];p=e['count_period'];fit=min(p['fits'],key=lambda x:abs(x['period_s']-p['period_s']))
    indices=set(fit['support_frames_indices'])
    fig,ax=plt.subplots(figsize=(11,4))
    select=[r for r in rows if r['time_s']<=clock['t0'] and r['available']]
    ax.plot([r['time_s'] for r in select],[r['N'] for r in select],'.',color='#a3a8ac',ms=3,label='全部可用距离')
    pts=[r for r in rows if r['frame'] in indices]
    ax.plot([r['time_s'] for r in pts],[r['N'] for r in pts],'.',color='#17876b',ms=4,label='远于 3 根、当前可信的参考')
    tt=np.array([fit['first_s'],fit['last_s']]);yy=fit['fit_intercept_at_t0']+fit['slope_pitch_per_s']*(tt-clock['t0'])
    ax.plot(tt,yy,color='#234f7d',lw=2,label='稳健拟合示例')
    ax.set(xlabel='时间 / 秒',ylabel='距离 / 绳节距',title=f"111500：根数变化周期 {p['period_s']:.2f}s，纹理周期 {e['texture_period']['period_s']:.2f}s")
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/'count_period.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,3.8))
    q=e['recurrence_curve'];ax.plot([r['time_s'] for r in q],[r['score'] for r in q],'-o',ms=4,color='#218572',label='第二次纹理回归')
    ax.axvline(30.,color='#666',ls=':',label='人工约 30 秒参考')
    ax.axvline(clock['t2'],color='#bf8c27',ls='--',label=f"两周期预测 {clock['t2']:.2f}s")
    ax.axvline(e['candidate_s'],color='#2473a2',ls='-.',label=f"图像候选 {e['candidate_s']:.1f}s")
    ax.set(xlabel='视频时间 / 秒',ylabel='归一化相关性',title=f"四个区域回归峰均在 {e['candidate_s']:.1f}s；{e['confirmation_s']:.1f}s 完成确认窗口")
    ax.legend(fontsize=9);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/'second_cycle.png',dpi=140);plt.close(fig)

def main():
    audit=read(ROOT/'regression_audit.json');assert audit['status']=='passed'
    videos=[read(ROOT/'delivery'/i['id']/'summary.json') for i in plan()]
    target=next(v for v in videos if '111500' in v['id']);id=target['id'];e=target['events'][0];clock=e['timeline']
    rows=read(ROOT/'delivery'/id/'frames.json');charts(rows,e)
    ref=dict(id=id,first_climb_s=24.7,near_s=24.9,leave_s=29.,motor_s_approximate=30.,
        t1_definition='跨层过渡结束、开始稳定绕新层第一圈',t2_definition='新层第一圈完成、第二圈即将开始',
        definition_confirmed_by_user=True,evaluation_only=True,same_video_used_for_development=True,
        human_motor_timestamp_precision_unknown=True,detector_reads_this_file=False)
    dump(ROOT/'human_reference.json',ref)
    old=read(R22/'motor'/f'{id}.json')['events'][0]['recurrence_s']
    evaluation=dict(old_candidate_s=old,new_model_prediction_s=clock['t2'],new_image_candidate_s=e['candidate_s'],
        approximate_human_reference_s=30.,old_difference_s=old-30.,new_difference_s=e['candidate_s']-30.,
        manual_time_not_a_fitted_detector_parameter=True,not_independent_generalization_test=True)
    dump(ROOT/'evaluation.json',evaluation)
    notes={
        '101517__right':'周期约 2.54 秒，参考 t2≈62.8 秒；超出本视频结尾，无法用本片确认。',
        '101643__right':'没有新的近边锚点，保持原离开趋势，不输出施力事件。',
        '113029__left':'没有新的近边锚点，保持原离开趋势，不输出施力事件。',
        '113029__right':'入卷活动绳不可见，保留未知，不生成施力事件。',
        '113305__right':'根数周期约 3.40 秒；参考 t2≈65.7 秒超出片尾，且纹理周期不足。',
        '113745__left':'根数周期约 3.64 秒，参考 t2≈40.5 秒；工具/手部遮挡，纹理不足，仅保留参考。',
        '111143__right':'仅纹理周期约 2.70 秒，参考 t2≈17.1 秒；远处计数历史较短且有遮挡，不升级为图像支持候选。',
        '111500__left':'根数与纹理一致，参考与第二次回归均约 30.4 秒，当前离开证据支持；31.1 秒完成核验。',
        '161234__right':'源片仅 0.9 秒，没有近边锚点和周期历史。',
        '212641__right':'源片仅 0.4 秒，没有近边锚点和周期历史。',
        '232941__right':'源片仅 0.9 秒，没有近边锚点和周期历史。'}
    table=[];cards=[]
    for v in videos:
        key=next(k for k in notes if k in v['id']);link=f"delivery/{v['id']}/RES23_{v['id']}_full.mp4"
        predictions='、'.join(f"{ev['timeline']['t2']:.1f}s" for ev in v['events'] if ev['timeline']) or '—'
        supported='、'.join(f"{ev['candidate_s']:.1f}s" for ev in v['events'] if ev['image_supported']) or '无'
        table.append(f"<tr><td><a href='{link}'>{v['id'].replace('dual_','')}</a></td><td>{predictions}</td><td>{supported}</td><td>{notes[key]}</td></tr>")
        if v['id']!=id:cards.append(f"<details><summary>{v['id'].replace('dual_','')} · {v['frames']/v['fps']:.1f} 秒</summary><p>{notes[key]}</p><video controls preload='none' src='{link}'></video><p><a href='events/{v['id']}.json'>周期 / 事件证据</a></p></details>")
    document=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RES23 · 根数周期与两周期施力参考</title><style>
body{{margin:0;background:#111b25;color:#dce6ee;font:16px/1.8 'Microsoft YaHei',sans-serif}}main{{max-width:1200px;margin:auto;padding:35px 24px 80px}}h1{{font-size:32px;line-height:1.4}}h2{{font-size:24px;margin-top:32px;color:white}}a{{color:#83d7c2}}video,img{{max-width:100%;border-radius:8px}}video{{width:100%;background:black}}.box,details{{background:#1a2939;border:1px solid #334c60;border-radius:10px;padding:22px;margin:18px 0}}.metrics{{display:flex;gap:18px;flex-wrap:wrap}}.metrics div{{flex:1;min-width:210px;background:#203647;border-radius:8px;padding:16px}}strong{{color:#9ae1c9}}.note{{color:#ebc37b}}.small{{font-size:14px;color:#acc0cf}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #35495b;padding:12px;text-align:left}}button{{background:#2a4e63;color:white;border:1px solid #547486;border-radius:5px;padding:8px 14px;margin:5px;cursor:pointer}}summary{{cursor:pointer;font-weight:bold}}code{{background:#101d29;padding:2px 5px}}.scroll{{overflow:auto}}</style>
<main><p class="small">2026-11-11 · RopeEdgeSuite R23 · CPL1 根数周期估计 + MFT2 两周期施力参考</p>
<h1>把首次爬升、稳定绕新层、首圈完成分开判断</h1>
<div class="metrics"><div>111500 原候选<br><strong>27.3 秒</strong>，约早 2.7 秒</div><div>新施力候选<br><strong>30.4 秒</strong>，较人工约 30 秒晚约 0.4 秒</div><div>回归验证<br><strong>11 段 / 4819 帧 / 45 项测试</strong></div></div>
<div class="box"><h2 style="margin-top:0">111500 优化结果</h2><p>根数变化周期 <strong>{e['count_period']['period_s']:.2f}s</strong>；纹理周期 <strong>{e['texture_period']['period_s']:.2f}s</strong>；综合周期 <strong>{clock['T']:.2f}s</strong>。</p>
<video id="mainvideo" controls preload="metadata" poster="delivery/{id}/review/000311.jpg" src="delivery/{id}/RES23_{id}_full.mp4"></video>
<div>{''.join(f'<button onclick="jump({t})">{label}</button>' for t,label in [(24.7,'24.7s 首次爬升'),(24.9,'24.9s 近边 t0'),(27.3,'27.3s 旧候选'),(27.7,'27.7s 首圈参考开始显示'),(30,'30s 人工参考'),(30.4,'30.4s 新施力候选'),(31.1,'31.1s 核验完成')])}</div>
<p>预测 t2={clock['t2']:.2f}s，模型范围 {clock['t2_interval'][0]:.2f}–{clock['t2_interval'][1]:.2f}s（非统计置信区间）。当前图像回归峰在 <strong>30.4s</strong>，31.1s 完成核验。<strong>31.1s 是复核确认时刻，施力目标仍为约 30.4s。</strong></p>
<p><a href="delivery/{id}/RES23_{id}_full.mp4">完整目标视频</a> · <a href="events/{id}.json">逐项事件证据</a> · <a href="evaluation.json">与人工约 30 秒参考对比</a></p></div>
<h2>偏早原因</h2>
<p>R22 把 24.6s 的“首行投影重叠”当作新层起点，再加一个 2.7s 周期，得到 27.3s。纹理返回证明图像相位重复，但不能说明当时已经稳定绕新层第一圈。经您的定义澄清，<strong>首次爬升与 t1（跨层过渡结束、开始稳定绕新层第一圈）是不同事件</strong>。前版因此少计了一个阶段。</p>
<table><tr><th>事件</th><th>时间</th><th>依据</th></tr>
<tr><td>首次爬升</td><td>约 24.7s</td><td>用户人工参考，单独保留</td></tr>
<tr><td>t0：近边</td><td>24.9s</td><td>已有算法的持续近边确认，与人工参考一致</td></tr>
<tr><td>t1：过渡结束，稳定绕新层第一圈</td><td>约 {clock['t1']:.1f}s</td><td>t0+T 的模型估计；定义已确认，时刻仍需图像/人工核验</td></tr>
<tr><td>t2：首圈完成，第二圈即将开始</td><td>约 30.4s</td><td>t0+2T 预测、第二次纹理回归及当前离开证据共同支持</td></tr></table>
<h2>周期参考如何参与综合判断</h2>
<p><code>N(t)≈a+v·t，T根数=1/|v|</code>。这个关系用于稳定、均匀排绳段。近边停留与爬升期间不满足简单轴向推进关系，因此不能直接用近边最后几个帧拟合。</p>
<p>诊断中，仅取最后 6 秒曾将 111500 的周期估到约 10.9 秒。CPL1 改为筛选距离大于 3 根、当前入卷轴可靠、无遮挡、误差范围合格的帧，用 0.5 秒时间分组中位数及稳健成对斜率拟合；同时检查至少 6 秒跨度、推进量、前后速度和多窗口一致性。</p>
<img src="count_period.png" alt="可信根数变化拟合">
<p>根数周期与纹理周期分别计算。两者明显冲突时不融合；相容时按固定等权频率组合，保留两者误差范围的包络。再按您提出的假设生成 <code>t1=t0+T，t2=t0+2T</code>。</p>
<p>到达预测时间只产生“周期参考”，不能自动宣布首圈完成。升级为图像支持候选还需要：两种周期一致、至少两个纹理区域回归一致、当前活动绳证据可靠，以及持续离边/回升证据。111500 的四个区域峰值均在 30.4s，并有 7 个当前离开支持帧。</p>
<img src="second_cycle.png" alt="第二次纹理回归验证">
<p class="small">稳健斜率采用成对斜率中位数思路，参见 <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html">SciPy 官方 Theil–Sen 文档</a>；纹理相关基础见 <a href="https://docs.opencv.org/4.13.0/de/da9/tutorial_template_matching.html">OpenCV 官方模板匹配文档</a>。周期与物理阶段之间的关系属于本场景假设，不能由这些文档直接保证。</p>
<h2>全量复测</h2><p>原 R22 的 F3、d、距离、运动方向及近边主状态全部保留。新增参考单独显示，遮挡时不会把计时推算冒充当前测量。</p>
<div class="scroll"><table><tr><th>视频</th><th>参考 t2</th><th>图像支持候选</th><th>结果与限制</th></tr>{''.join(table)}</table></div>
{''.join(cards)}
<h2>验证范围与下一步使用边界</h2>
<ul><li>新增 13 项行为测试和原有 32 项测试通过；覆盖异常帧、短样本、静止、反转、速度变化、周期冲突、两周期公式、无当前图像不能确认和输入截断时的因果一致性。</li>
<li>全部 4819 帧原字段逐项一致；11 个输出完整解码通过。R22 的源码与原视频哈希未改变。</li>
<li>查看了全部视频的抽样关键帧，以及目标 24.9–32 秒的关键阶段，共 24 个复核帧。未逐帧建立其他视频的物理真值。</li>
<li>约 30 秒是本次开发使用的近似人工参考，时间精度未标定；约 0.4 秒差异不能当成总体精度结论。检测规则没有读取该人工时间。</li>
<li>t0 后两周期对应施力节点，是本轮明确采用并得到这段视频支持的假设。转速明显变化、滑移、层间过渡改变或纹理重复别名时仍可能失效，需要当前证据交叉核验。</li>
<li>新增参考模块通过因果前缀测试；d 仍沿用离线标定，尚未完成整套现场在线闭环验证。本版本不发送电机控制指令。</li></ul>
<details><summary>展开实际复核图集</summary>{''.join(f'<img loading="lazy" src="review/{p.name}" alt="{p.name}">' for p in sorted((ROOT/'review').glob('*.jpg')))}</details>
<div class="box"><a href="regression_audit.json">回归审计</a> · <a href="visual_review.json">查看记录</a> · <a href="experiment_log.json">迭代日志</a> · <a href="human_reference.json">人工时间与事件定义</a> · <a href="period23.py">CPL1 周期代码</a> · <a href="fusion23.py">MFT2 综合判断代码</a></div>
</main><script>function jump(t){{const v=document.getElementById('mainvideo');v.currentTime=t;v.scrollIntoView({{block:'center',behavior:'smooth'}});}}</script></html>"""
    (ROOT/'report.html').write_text(document,encoding='utf8')
    dump(ROOT/'visual_review.json',dict(reviewed=True,frames=read(ROOT/'review_selection.json'),observations=notes,
        exhaustive_physical_ground_truth=False,target_assessment='around 30.4 seconds consistent with approximate user 30-second reference; physical phase still inferred'))
    dump(ROOT/'experiment_log.json',[
        dict(stage='R22 diagnosis',finding='first-row overlap was wrongly used as stable new-layer winding origin; single cycle is early',old_s=old),
        dict(stage='raw short-window trial',finding='near-edge nonuniform progress biases period; six-second target fit gave about 10.9s',file='inspection/period_diagnostics.json'),
        dict(stage='qualified count fit',finding='far current evidence plus steady-speed and multi-window checks',period_s=e['count_period']['period_s']),
        dict(stage='two-cycle fusion and image check',prediction_s=clock['t2'],candidate_s=e['candidate_s'],confirmation_s=e['confirmation_s']),
        dict(stage='full regression',frames=4819,videos=11,tests=45,all_R22_fields_unchanged=True)])
    links=[]
    for pair in re.findall(r'(?:src|href)="([^"]+)"|(?:src|href)=\x27([^\x27]+)\x27',document):
        link=next(x for x in pair if x)
        if '://' in link:continue
        assert (ROOT/unquote(link)).exists(),link;links.append(link)
    dump(ROOT/'report_link_audit.json',dict(passed=True,local_links=len(links)))
    snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for f in ROOT.glob('*.py'):shutil.copy2(f,snapshot/f.name)
    dump(ROOT/'release_manifest.json',dict(version='RES23_CPL1_MFT2',source_hashes={f.name:sha(f) for f in ROOT.glob('*.py')},
        baseline_R22_release_sha256=sha(R22/'release_manifest.json'),report_sha256=sha(ROOT/'report.html'),
        regression_audit_sha256=sha(ROOT/'regression_audit.json'),videos={v['id']:v['video_sha256'] for v in videos}))
    print('report ready',len(videos),'videos',len(links),'links')

if __name__=='__main__':main()
