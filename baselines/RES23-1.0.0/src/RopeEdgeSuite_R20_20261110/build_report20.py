"""Create the local, self-contained video/result review index after completion."""
import html,datetime,collections
from common20 import *

CONTACT={'unavailable':'证据不足','separated':'分离','near_edge_approaching':'接近','near_edge_uncertain':'近边但不确定','projection_conflict':'投影冲突','visual_contact_candidate':'视觉贴边候选'}
STATE={'unidentified':'未能独立标定','conditional_geometry_estimate':'条件几何估计','observed_bend_calibrated':'完整弯折标定'}
def escaped(x):return html.escape(str(x))
def lineplot(rows):
    # Missing values break the trace. The plot never connects across unknowns.
    nums=[x['N'] for x in rows if x['available']];upper=max(6,min(55,float(np.percentile(nums,98))+2)) if nums else 6
    duration=max(.1,rows[-1]['time_s']);w,h=1020,220
    x=lambda t:50+940*t/duration
    y=lambda n:185-155*(np.clip(n,-3,upper)+3)/(upper+3)
    bits=[f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="剩余节距与时间">']
    for v in [0,5,round(upper)]:bits.append(f'<line x1="50" x2="990" y1="{y(v)}" y2="{y(v)}" stroke="#30414f"/><text x="6" y="{y(v)+5}" fill="#bbcad5">{v}</text>')
    for key,col in [('N','#e2b25f'),('filtered','#73d7be')]:
        path='';connected=False
        for r in rows:
            value=r['filtered']['N'] if key=='filtered' else r['N'];ok=r['filtered']['available'] if key=='filtered' else r['available']
            if not ok or value is None:connected=False;continue
            path+=(' L' if connected else ' M')+f'{x(r["time_s"]):.1f},{y(value):.1f}';connected=True
        bits.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="1.6"/>')
    for t in np.linspace(0,duration,7):bits.append(f'<text x="{x(t)-10}" y="210" fill="#bbcad5">{t:.0f}s</text>')
    bits.append('</svg>');return ''.join(bits)

def main():
    assert read(ROOT/'delivery_receipt.json')['status']=='complete';audit=read(ROOT/'final_audit.json')
    items=plan();summaries=[read(ROOT/'delivery'/x['id']/'summary.json') for x in items];cards=[];table=[];data=[]
    for item,s in zip(items,summaries):
        id=item['id'];rows=read(ROOT/'delivery'/id/'frames.json');n=s['frames'];cal=s['calibration'];ds=[]
        for p in cal['epochs']:
            value='—' if p['d'] is None else f'{p["d_original_image"]:.1f} px [{p["d_interval_original_image"][0]:.0f}, {p["d_interval_original_image"][1]:.0f}]'
            ds.append(f'<li>{p["start"]/10:.1f}–{p["end"]/10:.1f}s：{STATE[p["status"]]}；d（原图）{value}；通过 ≥5 圈复核的参考帧 {len(p["far_guard_accepted_frames"])}。</li>')
        windows=read(ROOT/'delivery'/id/'near_observation_windows.json');ww='；'.join(f'近边观察 {q["start_s"]:.1f}–{q["end_s"]:.1f}s' for q in windows) or '无连续至少 3 帧的近边观察窗口。'
        ww+=' 以上为当前证据连续的近边观察，不能视为实体接触确认；不跨无证据帧或运动阶段拼接。'
        source_note='源文件仅能解码不足 1 秒，缺少标定时间跨度。' if item.get('partial_source') else ('活动绳入卷区落到画面外；上游绳可见不代表计数端点可观测。' if id=='dual_20260923_113029__right' else '')
        table.append(f'<tr><td><a href="#{id}">{escaped(item["session"])} {"右" if item["camera"]=="right" else "左"}</a></td><td>{n}</td><td>{s["F3_called_frames"]}</td><td>{s["available_distance_frames"]}</td><td>{s["complete_F3_phase_frames"]}</td><td>{sum(p["d"] is not None for p in cal["epochs"])}/{len(cal["epochs"])}</td></tr>')
        v=f'delivery/{id}/RES20_{id}_full.mp4';poster=f'delivery/{id}/review/{s["review_frames"][len(s["review_frames"])//2]:06d}.jpg'
        cards.append(f'''<section id="{id}"><h2>{escaped(item['session'])} · {'右相机' if item['camera']=='right' else '左相机'}</h2>
<p>{source_note}</p><video controls preload="none" poster="{poster}" src="{v}"></video><p><a href="{v}">打开完整视频</a> · <a href="delivery/{id}/frames.json">逐帧数据</a> · <a href="calibration/{id}.json">d 标定与参考帧审计</a> · <a href="delivery/{id}/near_observation_windows.json">近边观察窗口</a> · <a href="delivery/{id}/contact_windows.json">逐类别原始窗口</a> · <a href="delivery/{id}/audit.json">技术核验</a></p>
<ul>{''.join(ds)}</ul><p>{escaped(ww)}</p><p class="small">黄：当前帧距离估计；绿：短时鲁棒拟合。曲线断开表示无输出；纵轴为绳节距单位。模型误差范围见视频与逐帧文件。</p>{lineplot(rows)}</section>''')
        data.append(dict(id=id,frames=n,F3_frames=s['F3_called_frames'],distance_available=s['available_distance_frames'],complete_phase=s['complete_F3_phase_frames'],calibrated_epochs=sum(p['d'] is not None for p in cal['epochs']),epochs=len(cal['epochs']),source_note=source_note))
    total=sum(x['frames'] for x in summaries);available=sum(x['available_distance_frames'] for x in summaries);phase=sum(x['complete_F3_phase_frames'] for x in summaries)
    head=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>R20 F3、d 与贴边复核</title><style>
body{{margin:0;background:#0d1720;color:#e9f0f4;font:17px/1.7 "Microsoft YaHei",sans-serif}}main{{max-width:1120px;margin:auto;padding:32px 24px 80px}}h1{{font-size:30px}}h2{{font-size:23px}}a{{color:#79d4e8}}section{{border-top:1px solid #3a4d5a;padding:28px 0}}video{{width:100%;background:#000}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:10px;border-bottom:1px solid #30414f;text-align:left}}.small{{color:#b3c5d0;font-size:14px}}.box{{padding:18px 24px;background:#192a37;border-radius:8px}}svg{{width:100%}}</style><main>
<h1>RopeEdgeSuite R20：完整 F3、逐片 d、贴边证据复核</h1><p>本轮全量处理 {len(items)} 个视频，共 {total} 帧。以用户认可的 R19 活动绳/卷面前端为基准，重新处理 ROI、F3 分组与计数、卷面侧活动绳轴、逐运动阶段 d 和贴边证据。另对当前帧可证实的黄色法兰碎片截断问题做显式修复，原基准文件保留。</p>
<div class="box"><p>产生当前距离数值 {available}/{total} 帧；其中完整 F3 相位链 {phase} 帧。数值包含用于暴露模型不一致的有符号诊断，不能把它们全部视为可靠物理距离。这些数字表示输出覆盖情况，不能解释为识别准确率。独立逐帧计数和实体接触真值尚未提供。</p><p>视频中的“圈”是活动绳中心线到可见卷面边界的当前绳节距。观测绳沟条数另行显示；“视觉贴边候选”与“实体接触确认”含义不同。<a href="final_statistics.json">查看按可观测性、几何估计与投影冲突分类的统计</a>。</p></div>
<h2>本轮修正与证据</h2><ul>
<li>修复可见边界横向范围被误用为整个 F3 卷面范围的问题。在当前卷面轮廓内恢复被错误裁掉的区域，并保留遮挡区排除。113745 的 37 秒帧恢复了测量列下方 31 条可用 F3 交点，111500 的 28.1 秒帧恢复了 30 条；这些条数不是“剩余根数”。</li>
<li>卷面侧活动绳轴通过当前绳体、方向、宽度、与已有支持的重叠验证身份；上游绳轴外推与当前局部测量分开，并保留两者分歧。</li>
<li>从未归类的真实 F3 黄点恢复漏掉的局部绳沟，要求测量列两侧支持、残差及顺序合理；计数和显示共用同一组曲线。</li>
<li>对剩余失败帧复核卷面宽度：ROI22 补回位于当前卷面内、却因边界支持范围门限被排除的区域。黄色法兰在反光下分裂时，按当前色块高度关系关联；只有当前宽范围凸曲线通过分区留出检验，才允许替换失配的历史形状。所有修复帧保留原始结果、当前证据和哈希；未修改最终 d 参考帧。</li>
<li>d 按视频和运动阶段重新估计，每个最终参考帧在固定 d 范围内重新检验当前 F3 距离下界 ≥5 圈。缺少完整弯折证据时使用明确标记的条件几何估计；范围来自模型敏感性分析。</li>
<li>遮挡后拟合的边界显示为蓝色虚线。贴边判定只使用当前图像证据；短时预测最大 0.5 秒，不能确认接触。远离/接近分段处理，换向后重新建立标定。</li>
<li>缺少足够远离边缘的参考帧时，固定 d 及其计数保持未标定；如果当前绳轴和局部切线几何可观测，则另外给出当前帧近边状态。此分支的白色虚线是诊断列，不是已标定的固定 d，不进入锁列计数输出。</li></ul>
<p>这是使用整段视频的离线验证，部分阶段的 d 会使用该阶段后续参考帧；当前输出不等同于已验证的实时启动方案。</p>
<h2>尚未通过的部分与原因</h2><ul>
<li>完整观测 F3 相位链 1782 帧；另有 2093 帧依赖当前 F3 节距和几何模型。后者不能当作每条绳沟都已正确识别。</li>
<li>162 帧仍缺少当前有效证据，涉及初始化、局部 F3 缺口、遮挡和边界支持不足；其中仍有算法检测能力的限制，不能一概归因于原视频。</li>
<li>12 帧存在活动绳投影与边界模型冲突，主要集中在换向与工具干预附近。保留有符号诊断并标明冲突，没有把负值改成零圈。</li>
<li>770 帧没有独立固定 d：113029 右相机 596 帧的入卷区域不可见；111143 前 152 帧缺少远离至少五圈的同阶段参考；三个残缺视频共 22 帧，时间跨度不足。</li>
<li>9 个运动阶段获得条件几何 d，共使用 552 个最终固定列下仍满足五圈约束的参考帧。完整 P1 弯折证据未达到直接物理标定要求，因此本版没有把这些 d 称为已验证的实体切点。</li>
<li>近边观察窗口已输出，但这批单目投影与遮挡画面未能提供实体接触确认。当前没有独立真值，不能声称全部帧已识别正确或给出准确率。</li></ul>
<p><a href="iteration_log.md">迭代记录</a> · <a href="measurement_contract_tests.json">针对性测试</a> · <a href="final_audit.json">全量文件与视频核验</a> · <a href="visual_review.json">原图视觉复核记录</a></p>
<table><tr><th>视频</th><th>帧数</th><th>当前 F3 调用</th><th>产生距离数值</th><th>完整相位链</th><th>有 d 的阶段</th></tr>{''.join(table)}</table>
<p class="small">技术核验状态：{'通过' if audit['passed'] else '存在需查看的问题'}。所有数值与对应源帧、F3 数据、几何结果和标定文件通过哈希关联；交付视频经过完整解码检查。</p>'''
    (ROOT/'report.html').write_text(head+''.join(cards)+'</main></html>',encoding='utf8');dump(ROOT/'summary_table.json',data)
    print('Report built',total,available,phase,flush=True)

if __name__=='__main__':main()
