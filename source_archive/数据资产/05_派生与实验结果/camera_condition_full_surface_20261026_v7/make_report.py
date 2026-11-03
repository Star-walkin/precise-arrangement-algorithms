"""由实际冻结运行生成C4交付说明，不将文件通过写成语义准确率。"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import html
import json
import os
from urllib.parse import quote
HERE=Path(__file__).resolve().parent
ARCHIVE=HERE.parents[2]/'工程导航/05_实验版本档案/卷绳边界_整链基准_C3_20261026'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def esc(x):return html.escape(str(x))
def href(p):return quote(Path(os.path.relpath(p,HERE)).as_posix(),safe='/._-')
def link(p,label):return f'<a href="{href(p)}">{esc(label)}</a>'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--pilot',type=Path,required=True);ap.add_argument('--evaluation',type=Path,required=True)
    a=ap.parse_args();s=read(a.run/'summary.json');audit=read(a.run/'independent_audit.json');ev=read(a.evaluation/'summary.json')
    log=[json.loads(v) for v in (a.run/'run_log.jsonl').read_text(encoding='utf-8').splitlines()]
    empty=[v['output_file'] for v in log if v.get('modeled_pixels',0)==0]
    additions=[v for v in log if v.get('union_added_bright_pixels',0)>0]
    rows=[]
    for name,m in [('全部',ev['bright']['overall'])]+[(k,v) for k,v in ev['bright']['by_camera'].items()]:
        new,old=m['v7'],m['c3']
        rows.append(f'<tr><td>{esc(name)}</td><td>{new["reference_points"]:,}</td><td>{new["all_point_hit_rate"]:.2%}</td><td>{old["all_point_hit_rate"]:.2%}</td><td>{100*(new["all_point_hit_rate"]-old["all_point_hit_rate"]):+.3f} 个百分点</td><td>{new["all_point_capped_loss"]:.5f} / {old["all_point_capped_loss"]:.5f}</td></tr>')
    history=[]
    for p in sorted(HERE.glob('evaluation_*/summary.json')):
        d=read(p)
        history.append(f'<li>{link(p,p.parent.name)}：亮区命中 {d["bright"]["overall"]["v7"]["all_point_hit_rate"]:.2%}；边侧隐藏命中 {d["edge"]["overall"]["v7"]["all_point_hit_rate"]:.2%}。同输入简单外推与C4相同：{d["edge"]["predictions_identical_to_simple_when_no_dark_candidates"]}。</li>')
    examples=[
        ('left_camera/dual_20260923_113305__pair0315__left.jpg','左相机强眩光：亮区扩到完整卷面工作区，暗区直接观测与几何线分别着色。'),
        ('right_camera/dual_20260923_101643__pair0315__right.jpg','右相机活动绳：新增区域重新屏蔽；弱短组限制远端外推，部分位置仍留白。'),
        ('left_camera/ruler_left_05.jpg','必须保留的失败：原SAM漏分。并集仅恢复已认可亮区，矩形上部和下方SAM斜面并不等于完整自然轮廓。')]
    cards=''.join(f'<figure><a href="{href(a.run/"qa_compare"/p)}"><img src="{href(a.run/"qa_compare"/p)}" alt="{esc(t)}"></a><figcaption>{esc(t)}</figcaption></figure>' for p,t in examples)
    edge=ev['edge']['overall']['v7'];base=ev['edge']['overall']['simple_c3']
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>整链基准C3与全卷面C4试验</title>
<style>body{{max-width:1150px;margin:auto;padding:30px 24px;font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif;background:#f3f6f8;color:#173344}}h1{{line-height:1.4}}h2{{margin-top:32px}}a{{color:#006d9d}}.hero{{padding:22px;background:#dfedf5;border-radius:10px}}.button{{display:inline-block;padding:10px 18px;margin:6px 10px 6px 0;background:#006f95;color:white;border-radius:6px;text-decoration:none}}.note{{border-left:4px solid #c18324;background:#fff0d6;padding:12px 18px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #bfd1dd;padding:8px;text-align:left}}th{{background:#e0ebf2}}figure{{margin:20px 0}}img{{max-width:100%;border:1px solid #a3bdce}}figcaption{{font-size:14px}}code,pre{{background:#e0e9f0;padding:4px 7px;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;padding:14px}}li{{margin:6px 0}}</style>
<h1>已冻结整链基准 C3；另存全卷面 C4 试验结果</h1>
<p>2026-10-26 · 离线开放开发集 · 报告UTC {datetime.now(timezone.utc).isoformat()}</p>
<div class="hero"><p><b>整链基准 C3 / RB-C3.0.0</b>：根据用户“400张效果尚可”的确认，将原图黄点识别、活动绳屏蔽、归类和整组圆锥拟合及所需模型/代码封存。{link(ARCHIVE/'README.md','基准说明与恢复入口')} · {link(ARCHIVE/'baseline_manifest.json','基准冻结清单')}。</p>
<p><b>全卷面 C4 / RB-C4.0.0</b> 是基于C3的新试验分支：扩大工作ROI，重新筛选暗区黄点及限制外推，等待人工审阅。</p>
<a class="button" href="{href(a.run/'review.html')}#roi">查看 {s['planned']} 张完整工作ROI</a><a class="button" href="{href(a.run/'review.html')}">查看 {s['planned']} 张边界结果</a><p>{link(a.pilot/'review.html','固定38张对照')} · {link(a.run/'summary.json','全量运行汇总')}</p></div>
<h2>1. ROI怎样扩大</h2>
<p>采用 <b>原SAM2 v2卷面掩码 ∪ 原B1亮带ROI</b>。这是同一条规则应用到全部400图。工作ROI内保留原图RGB，外部统一为0；0/255掩码另存。原SAM2掩码及黑底图也逐字节另存，便于对照。</p>
<p>并集是必要的输入修正：首轮 ruler_left_05 的SAM2漏掉原亮ROI的144,880像素，导致原黄点和14,345像素旧未知遮挡被裁。首轮独立审计37/38，失败原样保留。第二轮恢复既有亮区，<b>并未修好SAM对其余暗区的语义漏分</b>；该图矩形保留区和下方斜面仍是待修的ROI缺陷。</p>
<p>本次全量原SAM像素合计 {s['sam_raw_pixels']:,}，工作ROI {s['surface_pixels']:,}；{len(additions)}张通过并集增加既有亮区，共 {s['union_added_bright_pixels']:,} 像素。工作ROI中原亮区外面积占 {s['expanded_pixels']/max(1,s['surface_pixels']):.2%}。面积扩大不是切点覆盖率。</p>
<h2>2. 暗区怎样识别、归类和拟合</h2>
<ol><li><b>保留亮区种子。</b>C3已接受组的黄点和整组模型作为起点；没有可靠种子的整行不按等间距凭空复制。</li>
<li><b>重新屏蔽完整工作ROI内的活动绳候选。</b>原屏蔽掩码只覆盖亮区，不能直接套到新增区域。本次在原图上重跑已有语义候选＋历史活动绳剖面算法，亮区保留旧屏蔽，新区域保存支持和未知两层。未检出不代表不存在活动绳。</li>
<li><b>筛选暗区原图黄点。</b>局部对比度增强仅用于检测，保存的ROI仍是原像素。候选必须同时有原始灰度对比、双侧亮纹、跨列连续支持和唯一组归属；股纹、饱和点、孤点和竞争分组被拒绝并记录。</li>
<li><b>每组整条拟合。</b>全部入选亮点与暗点以正权重拟合一条抛物线（圆锥曲线的一种），y=k+b·u+c·u²，c≥0。若暗点拉偏亮区，整组回退到C3系数，并记录拒绝的暗点。不会拼接两套模型来隐藏偏差。</li>
<li><b>有限外推。</b>长亮段的额外外推上限为30个光学绳距且最多750px，还受ROI、斜率、遮挡和邻线次序限制。亮段跨度小于12个绳距时，只接纳附近连续暗点，额外外推每侧最多2个绳距，防止短组向远暗区跳到邻绳。冲突处明确留白，系数和删除列位置均保存。</li></ol>
<p class="note">青线是靠近观测点的模型，紫线是缺口或外推模型；黄色为原亮区点，橙黄为筛选后的暗区点。二者仍是视觉候选。平滑、等间距及颜色不能证明真实同根材料、物理切点或换层事件。</p>
<h2>3. 实际输出与核验</h2>
<ul><li>{s['completed']}/{s['planned']}张处理完成，左{s['roles'].get('left',0)}、右{s['roles'].get('right',0)}，运行异常{s['failed']}。</li>
<li>{s['nonempty_curve_images']}张有模型曲线；{s['images_with_dark_observations']}张有新增暗区观测；{s['images_with_extended_models']}张在原亮区外有模型输出。新增暗区观测像素{s['dark_observed_pixels']:,}；亮区外模型像素{s['modeled_outside_bright_pixels']:,}；推断像素{s['inferred_pixels']:,}。这些仅是输出量。</li>
<li>独立文件/像素/结构审计 {audit['passed']}/{audit['planned']} 通过，失败{audit['failed']}：原图对应、SAM参考哈希、并集公式、ROI原像素、0/255、遮挡保留、曲线位置/权重/残差与交叉检查。</li>
<li>空曲线输出保留：{esc(', '.join(empty)) or '无'}。拒识未从400分母删除。</li></ul>
<p>唯一空图 R_run_20260923_01__f001348__right 在B1阶段产生的24条短轨迹全部因长度不足被拒绝，C3没有观测点和可用分组，C4因此未继续生成边界。图中仍可见绳纹，这是入口识别失败，并非没有钢丝绳。原图与预览对照还发现，该图SAM将左侧手套／工具的高亮区域收入ROI，存在非卷面误纳入。{link(HERE/'full_visual_review_001.json','全量中非QA样本与空图的只读目视记录')}。</p>
<h2>4. 固定38图的同口径比较</h2>
<p>以下是与原亮区黄点的一致性，阈值为1/4光学绳距；分母固定122,068个既有点，含拒识/漏点。没有暗区逐根人工真值，不能解释为全卷面准确率。</p>
<table><tr><th>范围</th><th>固定点数</th><th>C4命中</th><th>C3命中</th><th>变化</th><th>截断误差 C4/C3</th></tr>{''.join(rows)}</table>
<p>边侧隐藏检验：分别移除原亮ROI左/右15%区域黄点，重新分组和拟合，保存预测后才评分，不读取RGB重新找回隐藏点。共38图×2侧，隐藏点{edge['reference_points']:,}。C4命中{edge['all_point_hit_rate']:.2%}，同输入简单C3外推{base['all_point_hit_rate']:.2%}；截断误差均为{edge['all_point_capped_loss']:.5f}。两者预测完全相同；该检验<b>没有证明暗点加入的净增益</b>。</p>
<p>指标门槛：{'通过' if ev['metric_gate_pass'] else '未通过'}。{link(a.evaluation/'summary.json','详细分组指标')} · {link(a.evaluation/'visual_review.json','逐图目视记录')}。首轮与第二轮均为开发实验，未使用独立保留集。</p>
<h2>5. 代表图与已知失败</h2>{cards}
<h2>6. 历史记录与文件位置</h2><ul>{''.join(history)}</ul>
<p>首轮SAM漏裁审计、原始评价和第二轮修订均保留。ROI构造变化见{link(HERE/'protocol_revision_002.json','第二轮ROI修订')}；弱短组约束见{link(HERE/'revision_notes.json','算法修订记录')}。</p>
<table><tr><th>目录</th><th>内容</th></tr><tr><td>roi/surface_masked</td><td>400张工作ROI原色PNG，外部0</td></tr><tr><td>overlays</td><td>400张全分辨率边界结果PNG</td></tr><tr><td>roi/sam_reference_masks、sam_reference_masked</td><td>原SAM2掩码/原色图，逐字节保存</td></tr><tr><td>boundaries/modeled、inferred、dark_observed</td><td>全部模型、推断部分、暗区观测的0/255mask</td></tr><tr><td>boundaries/records</td><td>逐图分组、点来源、单模型系数、权重、残差、拒识与遮挡诊断</td></tr><tr><td>qa_compare</td><td>固定38张四列对照</td></tr></table>
<p>{link(HERE/'full_surface_core.py','暗区归类与拟合代码')} · {link(HERE/'run_full_surface.py','完整运行器')} · {link(HERE/'independent_audit.py','独立审计')} · {link(a.run/'run_context.json','环境/命令/源与模型哈希')} · {link(a.run/'run_log.jsonl','逐图运行日志')} · {link(a.run/'independent_audit.json','全量审计')} · {link(HERE/'freeze_all_001.json','全量运行前冻结决定')}</p>
<pre>&amp; D:\\precise_arrangement\\vision_prototype\\.venv\\Scripts\\python.exe &lt;本目录&gt;\\run_full_surface.py --mode all --output &lt;新的目录&gt; --workers 2</pre>
<p class="note">仍未验证：暗侧真实逐根准确率、SAM完整语义覆盖、活动绳完整遮挡、真实切点包含关系、不同外径泛化与t0/t1/t2。C3是本次用户接受的开发基准；C4等待人工复核。未连接设备、未使用保留集、未改线上模型或运动权限。</p></html>'''
    target=HERE/'结果说明.html'
    if target.exists():raise FileExistsError(target)
    target.write_text(page,encoding='utf-8');print('Report saved')
if __name__=='__main__':main()
