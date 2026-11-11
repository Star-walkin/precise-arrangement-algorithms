import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from common21 import *
import html,time,collections,shutil
from html.parser import HTMLParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']='Microsoft YaHei';plt.rcParams['axes.unicode_minus']=False

def charts(audit):
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    id='dual_20260924_111143__right';rows=read(ROOT/'delivery'/id/'frames.json');old=read(R20/'delivery'/id/'frames.json')
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True)
    t=np.array([r['time_s'] for r in rows]);newn=[r['N'] if r['available'] else np.nan for r in rows];oldn=[r['N'] if r['available'] else np.nan for r in old]
    axes[0].plot(t,oldn,color='#b46455',label='R20 当前估计',lw=2);axes[0].plot(t,newn,color='#087f8c',label='R21 当前估计',lw=1.5)
    low=[r['N_interval'][0] if r['available'] else np.nan for r in rows];high=[r['N_interval'][1] if r['available'] else np.nan for r in rows]
    axes[0].fill_between(t,low,high,alpha=.15,color='#087f8c',label='模型敏感性范围（非置信区间）');axes[0].set_ylabel('中心线距离 / 当前节距');axes[0].set_ylim(-3,8);axes[0].legend(loc='upper right')
    for y,rr,color,label in [(1,old,'#b46455','R20'),(0,rows,'#087f8c','R21')]:
        tt=[r['time_s'] for r in rr if r['available']];axes[1].scatter(tt,[y]*len(tt),marker='|',s=80,color=color)
    axes[1].set_yticks([0,1],['R21 可估计帧','R20 可估计帧']);axes[1].set_xlabel('视频时间 / 秒');axes[1].set_xlim(0,20)
    for ax in axes:ax.grid(alpha=.2);ax.axvline(15.2,color='gray',ls=':',label='固定列标定运动段切换')
    fig.suptitle('111143 右相机：先用当前动态列估计，后续运动段继续使用固定 d');fig.tight_layout();fig.savefig(out/'111143_before_after.png',dpi=140);plt.close(fig)
    id='dual_20260924_111500__left';rows=read(ROOT/'delivery'/id/'frames.json');old=read(R20/'delivery'/id/'frames.json');ev=read(ROOT/'events'/f'{id}.json')
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True);t=[r['time_s'] for r in rows]
    axes[0].plot(t,[r['N'] if r['available'] else np.nan for r in rows],color='#087f8c',label='当前中心线距离（R20、R21 相同）');axes[0].axhline(1.25,color='gray',ls=':',label='近边进入阈值');axes[0].set_ylim(-1,4);axes[0].set_ylabel('当前节距单位');axes[0].legend()
    target={'near_edge_uncertain','visual_contact_candidate'}
    oldt=[r['time_s'] for r in old if r['contact']['state'] in target];newt=[r['time_s'] for r in rows if r['near_event']['state']=='near']
    axes[1].scatter(oldt,[1]*len(oldt),marker='|',s=100,color='#b46455');axes[1].scatter(newt,[0]*len(newt),marker='|',s=100,color='#087f8c')
    axes[1].set_yticks([0,1],['R21 连续近边事件','R20 单帧近边']);axes[1].set_xlabel('视频时间 / 秒')
    for ax in axes:
        ax.set_xlim(22,32);ax.grid(alpha=.2)
        for e in ev['events']:ax.axvspan(e['start_s'],e['end_s'],alpha=.07,color='#087f8c')
        ax.axvline(24.9,color='#8a5f9d',ls='--');ax.axvline(29.,color='#8a5f9d',ls='--')
    fig.suptitle('111500 左相机：近边 24.6 秒起，24.9 秒确认，29.0 秒开始离开\n紫虚线为用户约 24.9 秒近边、29 秒离开的参考，仅用于评价');fig.tight_layout();fig.savefig(out/'111500_events.png',dpi=140);plt.close(fig)

def main():
    audit=read(ROOT/'regression_audit.json');receipt=read(ROOT/'delivery_receipt.json');charts(audit)
    summaries={r['id']:r for r in receipt['videos']};videos={r['id']:r for r in audit['videos']};early=read(ROOT/'delivery'/'dual_20260924_111143__right'/'frames.json')[:152]
    earlygood=sum(r['available'] for r in early);oldtotal=sum(r['old_available'] for r in videos.values());newtotal=sum(r['new_available'] for r in videos.values());fixed=sum(r['fixed_frames_exactly_equal'] for r in videos.values())
    ref=read(ROOT/'human_reference.json');ref['layer_entry_definition']='第一根绳开始跨上新一层（用户补充定义）';dump(ROOT/'human_reference.json',ref)
    rows=[];cards=[]
    for v in plan():
        id=v['id'];s=summaries[id];a=videos[id];short=id.replace('dual_','');url=f'delivery/{id}/RES21_{id}_full.mp4';baseline=f'../{R20.name}/delivery/{id}/RES20_{id}_full.mp4'
        # Some frozen R20 names are read from their actual summary, never guessed.
        baseline=Path(read(R20/'delivery'/id/'summary.json')['video']).relative_to(ROOT.parent).as_posix();baseline='../'+baseline
        rows.append(f'<tr><td><a href="#{id}">{short}</a></td><td>{a["frames"]}</td><td>{a["old_available"]} → {a["new_available"]}</td><td>{len(a["lost_available_frames"])}</td><td>{a["fixed_frames_exactly_equal"]}</td></tr>')
        evs=[]
        for e in s['near_events']:
            ending=f'离开 {e["departure_s"]:.1f}s' if e['departure_s'] is not None else '证据中断，未确认离开' if e['termination']=='evidence_lost_not_departure' else '片段结束，未确认离开'
            evs.append(f'{e["start_s"]:.1f}–{e["end_s"]:.1f}s（进入确认 {e["entry_confirmed_s"]:.1f}s；{ending}）')
        eventtext='；'.join(evs) or '没有形成满足当前证据约束的近边事件。'
        cal=read(R20/'calibration'/f'{id}.json');dlabels=[]
        for c in cal['epochs']:
            dd=c.get('d_original_image');dlabels.append(f'{c["start"]/v["fps"]:.1f}–{c["end"]/v["fps"]:.1f}s：'+(f'固定 d={dd:.1f}px' if dd is not None else '固定 d 未标定；仅在当前动态几何通过检查时输出'))
        buttons=''
        if id==ref['id']:
            buttons=' '.join(f'<button onclick="seek(\'{id}\',{t})">{label}</button>' for t,label in [(24.6,'算法近边起点 24.6s'),(24.7,'人工跨层参考 24.7s'),(24.9,'近边确认 24.9s'),(29.0,'离开判定 29.0s')])
        if id.endswith('111143__right'):buttons=' '.join(f'<button onclick="seek(\'{id}\',{t})">{t}s</button>' for t in [1,3,6,9,12,14,16])
        cards.append(f'''<section id="{id}"><h2>{short}</h2><p>{eventtext}</p><div class="buttons">{buttons}</div>
<video id="video_{id}" controls preload="none" poster="delivery/{id}/review/{s['review_frames'][min(2,len(s['review_frames'])-1)]:06d}.jpg" src="{url}"></video>
<p><a href="{url}">打开完整 R21 视频</a> · <a href="{baseline}">R20 对照视频</a> · <a href="delivery/{id}/frames.json">逐帧记录</a> · <a href="row_overlap/{id}.json">首列重叠候选</a> · <a href="audit/{id}.json">回归记录</a></p>
<p class="muted">{'；'.join(dlabels)}。d 坐标已还原原图朝向；运动段不等同于实体层编号。</p></section>''')
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R21 · 动态计数与近边事件复测</title>
<style>body{{margin:0;background:#101921;color:#dce5eb;font:17px/1.75 'Microsoft YaHei',sans-serif}}main{{max-width:1240px;margin:auto;padding:38px 26px}}h1{{font-size:32px}}h2{{font-size:24px}}a{{color:#83d8d8}}section,.box{{background:#182631;padding:24px;margin:24px 0;border-radius:12px}}video,img{{width:100%;height:auto;border-radius:8px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;border-bottom:1px solid #354755;padding:11px}}.muted{{color:#adbdc9;font-size:15px}}.numbers{{font-size:22px;color:#93e3c2}}button{{background:#28565d;color:white;border:0;padding:9px 14px;border-radius:7px;cursor:pointer;margin:4px}}code{{background:#101921;padding:3px 7px}}.tag{{color:#edc282}}ul{{padding-left:24px}}</style></head><body><main>
<h1>RopeEdgeSuite R21：动态列计数与连续近边事件</h1><p>2026-11-10 · 11 个视频 / {audit['frames']} 帧 · R20 保留为对照基准</p>
<div class="box"><p class="numbers">当前可估计帧：{oldtotal} → {newtotal}；原有有效帧丢失 0；{fixed} 帧固定列结果逐项一致。</p>
<p>已对全部视频重新计算 F3 计数与距离敏感性、运行动态列质量检查、分析近边事件，并从原视频逐帧解码生成结果。沿用并逐文件校验 R20 已接受的活动绳、卷面、F3 黄点和分组缓存；这部分检测器未改动。输出视频再次完整解码核查。</p>
<p class="tag">技术回归通过不代表逐帧实体真值已验证。距离为当前绳节距单位的图像几何估计；实体接触、第一根绳跨上新层仍需区分。</p></div>
<section><h2>111143：为何前 15 秒无可靠计数</h2><p>前一运动段从近边开始，没有满足“至少远离 5 根绳”的固定 d 参考帧。R20 将固定 d 未标定和完全不可估计绑定，导致 0–15.2 秒的 152 帧都没有主计数输出。</p>
<p>R21 保留 ≥5 圈固定 d 标定约束，同时允许当前活动绳轴线、F3 曲线切向列、绳节距和卷面边界共同支持的动态列估计。至少需要 4 帧当前证据、连续性检查和可接受误差范围。短段鲁棒轨迹使用测量误差范围，避免逐帧差分放大噪声。</p>
<p>该段恢复 <b>{earlygood}/152 帧</b>；3、6、9、12 秒原始当前估计约 4.6、2.8、1.7、1.0 圈。视频主数字沿用 3 秒稳健平滑，可能滞后于当前原始值（例如 6 秒主数字约 3.6，当前原始值约 2.8）；近边事件使用当前测量，不使用平滑预测。其余帧因预热、弯折或当前轴线缺失、遮挡、列范围过宽等继续留空。白虚线表示当前动态列，不能称为已锁定 d。</p><img src="figures/111143_before_after.png" alt="111143 计数前后对比"></section>
<section><h2>111500：近边、开始离开与首根跨层分开评价</h2><p>R20 的单帧阈值将近边窗口切成多段；离开必须等距离越过绝对阈值，因而延后。投影距离最低点 28.1 秒只是运动分段依据，不能代表第一根绳跨层。</p>
<table><tr><th>事件</th><th>用户参考</th><th>R21 结果</th><th>含义</th></tr>
<tr><td>接近边缘</td><td>约 24.9s</td><td>24.6s 起，24.9s 确认</td><td>连续当前证据支持的视觉近边窗口</td></tr>
<tr><td>第一根绳开始跨层</td><td>约 24.7s</td><td>首列重叠候选 24.6s 起，25.0s 确认</td><td>时段相近，但投影重叠不足以确认实体换层</td></tr>
<tr><td>开始离开边缘</td><td>约 29.0s</td><td>29.0s</td><td>距离相对近边低位持续回升，而非等待绝对距离超过阈值</td></tr></table>
<p>最后一轮还修正了边界支撑条件：计算列短暂超出可见支撑点范围时，不再一律丢弃，而要求当前拟合的边界误差不超过一个绳节距、总距离范围不超过四个节距且无遮挡。首版过严条件会将离开延迟到 29.7 秒。相对 R21 首轮全量输出，该修正只改变两段目标视频中少量连续事件文字，其他 9 个视频的事件显示和视频文件逐字节不变；计数、F3、d 均不变。</p>
<p>首列检查要求最外侧已解析绳沟及后续至少三个完整节距、当前绳中心落入首列内区、当前边界支撑和无遮挡。它排除了缺失顶端绳沟的假首列。另行测试的直线姿态变化起点为 25.6 秒，不能用作您定义的首根跨层时刻；该试验仅保留作原因分析。</p>
<p>人工时间只用于评价。检测代码没有使用视频名或人工时间作为阈值。事件起点允许回溯到支持帧，同时记录确认时间；两者没有混写。事件窗口在给定标定后按当前及历史帧判断；固定 d 沿用 R20 离线标定，部分标定参考来自后续帧，因此本次整套复测不能作为严格在线运行验证。</p><img src="figures/111500_events.png" alt="111500 近边事件比较"><p><a href="inspection/111500_row_overlap_trial.json">首列重叠证据</a> · <a href="inspection/111500_pose_trial.json">姿态分段试验</a> · <a href="inspection/dual_20260924_111500__left_detail_sheet0.jpg">24.0 / 24.5 / 24.7 秒原图局部对照</a></p></section>
<section><h2>全量回归与视频入口</h2><p>计数可用性按当前测量 available 统计，不把短时预测计入。新增数字均来自原先固定 d 未标定的帧。其余视频原有固定 d 数值、误差范围、F3 完整相位状态及单帧贴边状态均保持一致。新增连续事件要求更多当前支撑，不能形成完整事件时，原单帧近边提示仍在画面下方保留。未标定、入卷点不可见、源视频过短时，仍保留相应限制。</p>
<table><tr><th>视频</th><th>帧数</th><th>可估计帧 R20 → R21</th><th>原有有效帧丢失</th><th>固定列完全一致帧</th></tr>{''.join(rows)}</table></section>
{''.join(cards)}
<section><h2>验证材料与复现</h2><p><a href="regression_audit.json">全量回归审计</a> · <a href="measurement_receipt.json">完整重算记录</a> · <a href="dynamic_b_receipt.json">动态估计第二轮试验</a> · <a href="human_reference.json">人工参考及定义</a> · <a href="visual_review.json">图像抽查记录</a> · <a href="release_manifest.json">源码与产物索引</a></p>
<p>原有 7 项计数约束测试和新增 8 项动态列、近边事件、遮挡及首列缺失测试均通过。复现顺序：<code>run_measure21.py --workers 6</code>、<code>refine_dynamic21.py</code>、<code>finish_final21.py --workers 4</code>、<code>audit_final21.py</code>、<code>build_report21.py</code>。使用工作区 .venv_lc1 的 Python；运行目录为本 R21 文件夹。</p>
<p class="muted">此轮人工核对两段目标视频的关键时刻，并对其余视频的近边、遮挡和动态输出作抽查；没有全帧实体接触或换层真值，不能宣称全部视频已经达到 100% 识别准确率。</p></section>
</main><script>function seek(id,t){{const v=document.getElementById('video_'+id);v.currentTime=t;v.scrollIntoView({{behavior:'smooth',block:'center'}});v.play();}}</script></body></html>'''
    (ROOT/'report.html').write_text(page,encoding='utf8')
    snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=True);sources={}
    for p in ROOT.glob('*.py'):shutil.copy2(p,snapshot/p.name);sources[p.name]=sha(p)
    manifest=dict(version='RopeEdgeSuite R21',source_hashes=sources,baseline_release_manifest_sha256=sha(R20/'release_manifest.json'),
        production=['measurement21b.py','final_input21.py','events21b.py','row_overlap21.py','finish_final21.py','render_final21.py'],
        provenance_note='full fixed-column recount uses count20.py from frozen R20; dynamic gates A and B retained as experiments',
        videos=[dict(id=s['id'],video=s['video'],sha256=s['video_sha256']) for s in receipt['videos']],
        calibration_hashes={v['id']:sha(R20/'calibration'/f"{v['id']}.json") for v in plan()},
        report_sha256=sha(ROOT/'report.html'),regression_audit_sha256=sha(ROOT/'regression_audit.json'))
    dump(ROOT/'release_manifest.json',manifest)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('href','src','poster') and v and not v.startswith('#'):self.links.append(v)
    parser=Links();parser.feed(page);missing=[v for v in parser.links if not (ROOT/v).exists()];assert not missing,missing
    dump(ROOT/'report_link_audit.json',dict(links=len(parser.links),missing=missing));print('report complete',oldtotal,newtotal,'fixed',fixed,'links',len(parser.links))

if __name__=='__main__':main()
