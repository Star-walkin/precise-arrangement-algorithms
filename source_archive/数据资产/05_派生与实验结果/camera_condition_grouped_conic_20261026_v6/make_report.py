"""从C3实际运行记录生成交付说明，不修改预测、旧实验或评价口径。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import html
import json
import os
from urllib.parse import quote

HERE=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def esc(v):return html.escape(str(v))
def link(p,label):
    rel=quote(Path(os.path.relpath(p,HERE)).as_posix(),safe='/._-')
    return f'<a href="{rel}">{esc(label)}</a>'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--pilot',type=Path,required=True);ap.add_argument('--evaluation',type=Path,required=True)
    a=ap.parse_args();s=read(a.run/'summary.json');audit=read(a.run/'integrity_audit.json')
    ev=read(a.evaluation/'summary.json');old=read(HERE.parent/'camera_condition_local_conic_20261025_v5/evaluation_001/summary.json')
    rows=[]
    for title,m,c2 in [('全部',ev['overall'],old['overall'])]+[(k,v,old['by_camera'][k]) for k,v in ev['by_camera'].items()]+[(f'隐藏带{k}',v,old['by_band'][k]) for k,v in ev['by_band'].items()]:
        c=m['conic'];b=m['linear'];o=c2['conic']
        rows.append(f'<tr><td>{esc(title)}</td><td>{c["hidden_points"]:,}</td><td>{c["all_point_hit_rate"]:.2%}</td><td>{o["all_point_hit_rate"]:.2%}</td><td>{b["all_point_hit_rate"]:.2%}</td><td>{c["all_point_capped_loss"]:.5f} / {o["all_point_capped_loss"]:.5f} / {b["all_point_capped_loss"]:.5f}</td></tr>')
    logs=[json.loads(v) for v in (a.run/'run_log.jsonl').read_text(encoding='utf-8').splitlines()]
    empty=[v['output_file'] for v in logs if v.get('curve_count',0)==0]
    history=[]
    for f in sorted(HERE.glob('evaluation_*/summary.json')):
        e=read(f);c=e['overall']['conic']
        history.append(f'<li>{link(f,f.parent.name)}：隐藏黄点命中 {c["all_point_hit_rate"]:.2%}，截断误差 {c["all_point_capped_loss"]:.5f}，预设门槛 {"通过" if e["predeclared_continue_criterion_met"] else "未通过"}。</li>')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C3 黄点归类与整组圆锥拟合</title>
<style>body{{max-width:1100px;margin:auto;padding:32px 24px;background:#f3f6f9;color:#183244;font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif}}a{{color:#006c99}}h1{{line-height:1.4}}h2{{margin-top:32px}}.hero{{background:#ddebf4;padding:22px;border-radius:12px}}.button{{display:inline-block;background:#00759c;color:white;padding:10px 22px;text-decoration:none;border-radius:6px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border:1px solid #b7cedb;text-align:left}}th{{background:#e0ebf2}}code,pre{{background:#e1eaf1;padding:3px 6px;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;padding:15px}}.note{{background:#fff0d5;border-left:4px solid #bf8424;padding:12px 18px}}li{{margin:7px 0}}</style>
<h1>整组圆锥 C3：先将黄点归类，再用每组全部点拟合</h1>
<p>任务日期 2026-10-26 · 仅固定开放开发集 · 报告生成 UTC {datetime.now(timezone.utc).isoformat()}</p>
<div class="hero"><p><b>整组圆锥 C3 / RB-C3.0.0</b> 实现用户的两步要求：<b>第一步黄点归组；第二步每组全部黄点共同拟合一条圆锥曲线。</b>每个接受组只有一套系数，整个可见区及内部缺口使用同一模型。</p>
<a class="button" href="{quote(Path(os.path.relpath(a.run/'review.html',HERE)).as_posix(),safe='/._-')}">查看 {s['planned']} 张结果</a>　{link(a.pilot/'review.html','固定38张对照')}<p>旧基准 {link(HERE.parents[2]/'工程导航/05_实验版本档案/卷绳边界_亮带基准_B1_20261025/README.md','亮带基准 B1 / RB-B1.0.0')} 与 {link(HERE.parent/'camera_condition_local_conic_20261025_v5/结果说明.html','局部圆锥 C2')} 保持原样。</p></div>
<h2>1. 两步具体做了什么</h2>
<ol><li><b>分组。</b>读取B1黄点，使用局部趋势、跨列位置、相邻边界次序和间距，关联可能属于同一视觉边界的片段。仅以当前可见点分组，不读取旧紫线或C2的分组答案。每个原点按“原片段ID＋点序号”保存唯一归属；拒识组也保留。</li>
<li><b>整组拟合。</b>设归一化横坐标 u=(x−中心)/尺度，拟合 y=k+b·u+c·u²，c≥0。这里使用抛物线，它是圆锥曲线的一种；图像y轴向下，因此c&gt;0时两臂向图像下方延伸。每组全部黄点均以严格正权重参与同一个稳健最小二乘目标，邻行形状约束作为单独记录的正则项。异常点可降低权重，不能偷偷删为只剩端点。</li>
<li><b>统一绘制。</b>整组从最左黄点至最右黄点使用同一套系数，不再将“可见段曲线”与“缺口曲线”拼成一条线。活动绳和未知排除区留空；这些空白不是换用另一条模型。</li></ol>
<p>“全部点参与拟合”不等于曲线必须精确穿过每一个点；观测噪声和模型偏差会留下残差。c=0时退化为直线并在记录中标明。不能辨识的组记录拒绝原因，不编造可靠性。</p>
<h2>2. 怎样检查分组和曲线</h2><ul><li>图集切换<b>第一步：黄点归类</b>：同色同编号是一组。左右重复编号便于检查缺口两侧是否接到相邻边界；灰色组拟合被拒绝。</li><li>切换<b>第二步：整体曲线</b>：每组是一条整体模型。青色靠近黄点，紫色离观测较远，可能是缺口推断，也可能是拟合偏差。</li><li>切换<b>整体曲线＋原黄点</b>：检查模型是否偏离原始黄色观测。颜色本身不证明真实边界正确。</li><li>固定38图提供原图 / 分组 / 旧C2 / 新C3四列对照。每图JSON保存全部组、点引用、权重、残差、模型和拒绝原因。</li></ul>
<h2>3. 实际运行和结构检查</h2><ul><li>处理 {s['completed']}/{s['planned']} 张，左 {s['roles'].get('left',0)}、右 {s['roles'].get('right',0)}；运行异常 {s['failed']} 张。</li><li>有模型轮廓 {s['nonempty_curve_images']} 张；黄点组 {s['group_count']:,}，接受单模型组 {s['curve_count']:,}，入选拟合点 {s['accepted_group_point_count']:,}/{s['grouped_point_count']:,}。这些是输出数量，不是已确认绳根数或准确率。</li><li>独立结构/文件/像素审计 {audit['passed']}/{audit['planned']} 通过：原B1文件逐字节不变、点引用恰好分组一次、每接受组单模型、全部成员正权重、残差与模型一致、0/255掩码不越界。</li><li>空输出保留：{esc(', '.join(empty)) or '无'}。</li></ul>
<h2>4. 与直线和旧C2的同口径比较</h2><p>仍用同38张开放开发图、同两条预定隐藏带和同评分函数。先删除隐藏点再重新分组和拟合，保存预测后再读评分参考。总计38图×2带，不是76个独立工况。命中容差为1/4光学绳距；未预测点计入误差和全部分母。</p>
<p class="note">C3 本次固定遮点命中为 {ev['overall']['conic']['all_point_hit_rate']:.2%}，旧C2为 {old['overall']['conic']['all_point_hit_rate']:.2%}；C3截断误差 {ev['overall']['conic']['all_point_capped_loss']:.5f}，旧C2为 {old['overall']['conic']['all_point_capped_loss']:.5f}（越低越好）。C3满足本次整组单模型实现要求，并通过相对直线的门槛；缺口预测指标仍弱于C2。B1、C2和首轮失败均保留。</p>
<table><tr><th>范围</th><th>隐藏点</th><th>C3命中</th><th>C2命中</th><th>直线命中</th><th>截断误差 C3 / C2 / 直线</th></tr>{''.join(rows)}</table>
<p>预设继续门槛：<b>{'通过' if ev['predeclared_continue_criterion_met'] else '未通过'}</b>。它要求总体截断误差改善，以及左右相机、各隐藏带命中率不明显退化；并不要求每条带的误差都改善。隐藏带2的截断误差相对直线仍高 {ev['by_band']['2']['conic']['all_point_capped_loss']/ev['by_band']['2']['linear']['all_point_capped_loss']-1:.2%}，这项局部退化保留。{link(a.evaluation/'summary.json','逐图/分组完整分数')} · {link(HERE/'evaluation_protocol.json','固定评价协议')}。遮点参考来自已有B1黄点，不是真实过曝中心真值，也不确认物理同根身份。</p>
<h2>5. 迭代与复现</h2><ul>{''.join(history)}</ul>
<p>首轮完整保留。第二轮只修正短孤片段归类：若少量原黄点与唯一长组贴合，合并后再次用所有点拟合；其余不确定组仍拒识。未遮点的固定38张显示结果与首轮152张显示文件逐字节相同；改变发生在遮点后产生的短片段处理，不能把代理分数的改善描述成实际38图都变好。</p>
<p>已保留的可见退化包括左 dual_20260923_101643__pair0315（4个拒识组、167个点没有接受曲线）和 ruler_left_01（2个拒识组、88个点），分组图中以灰色显示。左右强眩光代表图则可见整体连接更平顺。{link(HERE/'independent_qa_pilot_001.json','首轮逐图视觉检查及退化')} · {link(HERE/'independent_qa_pilot_002.json','第二轮独立复核')}；所有目视结论均只针对实际检查图片。</p>
<p>{link(HERE/'conic_family.py','归组与全点拟合代码')} · {link(HERE/'run_conic.py','输出与可视化代码')} · {link(HERE/'audit_conic.py','独立审计代码')} · {link(a.run/'run_context.json','环境、命令和源码哈希')} · {link(a.run/'run_log.jsonl','逐图运行日志')} · {link(a.run/'integrity_audit.json','完整性与结构审计')} · {link(HERE/'protocol.json','本次实施协议')}</p>
<pre>&amp; D:\\precise_arrangement\\vision_prototype\\.venv\\Scripts\\python.exe &lt;本目录&gt;\\run_conic.py --mode all --output &lt;新的运行目录&gt; --workers 2</pre>
<p>目录：group_overlays＝分组图；smooth_overlays＝整体曲线图；overlays＝叠加黄点；boundaries/records＝逐组数据；boundaries/smooth_boundaries＝0/255完整模型mask；roi/effective_masked＝原色ROI且外部0。</p>
<p class="note">几何分组仍可能误把邻绳归到一起；整组单抛物线也可能无法贴合全部真实弯曲。请结合分组图和原图复核。实际过曝区、活动绳遮挡完整性、三维材料身份及t0/t1/t2没有因此得到验收。本次未连接设备、未改权限、未读取保留集。</p></html>'''
    (HERE/'结果说明.html').write_text(page,encoding='utf-8')
    print('Report saved')

if __name__=='__main__':main()
