"""汇总真实保存的运行与检验结果；不改算法、不改旧实验。"""
from pathlib import Path
import argparse
import html
import json
import os
from urllib.parse import quote

HERE=Path(__file__).resolve().parent
BASELINE=Path('D:/precise_arrangement/工程导航/05_实验版本档案/卷绳边界_亮带基准_B1_20261025')

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def href(path):return quote(Path(os.path.relpath(path,HERE)).as_posix(),safe='/._-')
def link(path,label):return '<a href="'+href(path)+'">'+html.escape(label)+'</a>'
def pct(x):return '—' if x is None else f'{100*x:.2f}%'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--evaluation',type=Path,required=True);ap.add_argument('--pilot',type=Path,required=True)
    args=ap.parse_args();run=args.run;ev=read(args.evaluation/'summary.json')
    s=read(run/'summary.json');audit=read(run/'integrity_audit.json')
    logs=[json.loads(x) for x in (run/'run_log.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[]
    for title,metrics in [('全部',ev['overall'])]+[(('左相机' if k=='left' else '右相机'),v) for k,v in ev['by_camera'].items()]+[(f'隐藏带 {k}',v) for k,v in ev['by_band'].items()]:
        c,b=metrics['conic'],metrics['linear']
        rows.append(f'<tr><td>{title}</td><td>{c["hidden_points"]:,}</td><td>{pct(c["all_point_hit_rate"])}</td><td>{pct(b["all_point_hit_rate"])}</td><td>{c["all_point_capped_loss"]:.4f}</td><td>{b["all_point_capped_loss"]:.4f}</td></tr>')
    empty=[r['output_file'] for r in logs if r.get('curve_count',0)==0]
    worst=sorted(ev['case_metrics'],key=lambda c:c['methods']['conic']['sum_capped_normalized_error']-c['methods']['linear']['sum_capped_normalized_error'],reverse=True)[:3]
    regressions=''.join(f"<li>{html.escape(c['source'])}，隐藏带{c['band_index']}：C2命中{c['methods']['conic']['hits_within_quarter_pitch']}/{c['hidden_points']}，直线{c['methods']['linear']['hits_within_quarter_pitch']}/{c['hidden_points']}。</li>" for c in worst)
    versions=[]
    for path in sorted(HERE.glob('evaluation_*/summary.json')):
        e=read(path);m=e['overall'];versions.append(f'<tr><td>{link(path,path.parent.name)}</td><td>{pct(m["conic"]["all_point_hit_rate"])}</td><td>{m["conic"]["all_point_capped_loss"]:.4f}</td><td>{"通过" if e["predeclared_continue_criterion_met"] else "未通过"}</td></tr>')
    actualtime=read(run/'run_context.json')['started_utc']
    explanation=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>亮带基准 B1 与圆锥曲线补全 C2</title>
<style>*{{box-sizing:border-box}}body{{max-width:1150px;margin:0 auto;padding:36px 25px 70px;background:#f4f7fa;color:#173043;font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif}}h1{{font-size:30px;line-height:1.4}}h2{{margin-top:35px}}a{{color:#0869aa}}.hero{{background:#e0eef5;padding:22px;border-radius:14px}}.button{{display:inline-block;background:#086a94;color:white;text-decoration:none;padding:10px 20px;border-radius:7px}}.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}.cards div{{background:white;border:1px solid #c9dce7;padding:14px 20px;border-radius:9px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #c3d4df;padding:9px;text-align:left}}th{{background:#e1ebf1}}code,pre{{background:#e2eaf0;padding:3px 6px;border-radius:3px;overflow-wrap:anywhere}}pre{{padding:16px;white-space:pre-wrap}}.limit{{border-left:4px solid #ba7c1b;background:#fff1d9;padding:13px 19px}}li{{margin:7px 0}}small{{color:#466476}}</style>
<h1>亮带基准 B1 与圆锥曲线补全 C2</h1>
<p>用户任务日期：2026-10-25 · 离线开发实验 · 新全量运行始于 UTC {html.escape(actualtime)}</p>
<div class="hero"><p><b>回溯基准：亮带基准 B1，版本号 RB-B1.0.0。</b>用户认可的 v3/all_001 已冻结源码、模型、配置和证据。{link(BASELINE/'README.md','基准档案与复现说明')} · {link(BASELINE/'baseline_manifest.json','基准哈希清单')}</p>
<p><b>本版：局部圆锥补全 C2（RB-C2.0.0）。</b>保留 B1 的黄色观测和 ROI，以已冻结的可见段拟合作为基础，仅在内部缺口做局部圆锥推断。中央过曝区域的紫线是模型预测。</p>
<a class="button" href="{href(run/'review.html')}">打开全部 400 张结果</a>　{link(args.pilot/'review.html','固定 38 张三列对照')}</div>
<div class="cards"><div>输出 <b>{s['completed']}/{s['planned']}</b><br>左 {s['roles'].get('left',0)} / 右 {s['roles'].get('right',0)}</div><div>运行失败 <b>{s['failed']}</b><br>有平滑曲线 {s['nonempty_curve_images']}</div><div>有缺口补线 <b>{s['images_with_gap']}</b><br>几何补线像素 {s['gap_pixels']:,}</div><div>文件/像素审计 <b>{audit['passed']}/{audit['planned']}</b><br>拟合近距冲突图 {audit['images_with_close_pairs']}</div></div>
<p class="limit">上述数字描述文件、几何一致性和输出量，不是逐根绳边界准确率。过曝处缺少可见纹理和独立逐像素真值；光滑、绳距相近与遮点检验通过，均不能证明活动绳同根身份或实际接触。</p>
<h2>1. 怎样看结果</h2><ul><li><b style="color:#087c8b">青色</b>：靠近原黄点的平滑拟合。它仍是模型曲线，不把拟合点当作实测点。</li><li><b style="color:#b01bac">紫色</b>：距原观测超过约 0.22 倍光学绳距的模型像素，包括内部缺口，也可能包含个别拟合偏差；“紫色”不自动等于“被确认的过曝区域”。</li><li><b style="color:#9a7500">黄色</b>：B1 原观测，可在图集勾选“叠加原黄点”。黄色坐标和掩码逐字节保留。</li><li>橙色活动绳候选、蓝色未知区沿用 B1 并禁止补线。绿色是显示包络，精确 ROI 见掩码；边框平滑只改变展示。</li></ul>
<h2>2. 如何按你的思路拟合</h2>
<ol><li><b>取真实观测。</b>只读取 B1 的黄点和光学绳距，剥离旧紫线、旧曲线分组和已拟合答案。原图像素只用于展示。</li><li><b>保留可见段基础。</b>复用冻结的 C1 第二轮几何模块，在可见支持范围内生成平滑基础，并从多条可靠黄点片段估计共同弯曲方向。它只是新方法的内部基础，不代表 C1 已通过验收。</li><li><b>只在内部缺口建模。</b>用两侧可见黄点、局部切线及邻行弯曲估计圆锥曲线。端点式为 y(x)=chord(L,R)+a(x−L)(x−R)，既约束两端位置，也利用相邻行提供曲率。补线仅在缺口区间生效，不将它扩展到整条已可见边界。</li><li><b>开口方向明确。</b>图像 y 轴向下，因此 a&gt;0 对应顶点在上、两臂向下，轴平行图像纵向；接近直线可退化为直线，明显反向弯曲拒绝。没有声称真实三维螺旋投影必然严格满足该方程。</li><li><b>检查配对与相邻距离。</b>两端位置、趋势、绳距和顺序共同约束匹配；不足时留空。对曲线交叉或过近处记录冲突处理，原黄点不删除。每段模型单独保存来源、系数和有效区间，一根视觉轨迹可能由多段构成，不假称全长是同一条精确圆锥曲线。</li><li><b>逐列输出。</b>每个 x 列输出拟合 y，限定在原 ROI 和观测支持跨度内。活动绳/未知区始终留空，不做范围外外推。</li></ol>
<p>{link(HERE/'conic_family.py','局部缺口拟合代码')} · {link(HERE/'base_conic.py','冻结可见段基础代码')} · {link(HERE/'run_conic.py','批处理与输出代码')} · {link(HERE/'audit_conic.py','独立像素检查代码')} · {link(HERE/'evaluation_conic.py','遮点评价代码')}</p>
<h2>3. 固定遮点检验</h2><p>固定 38 张开放开发图，左右各 19 张。分别隐藏 ROI 横向 25%–40% 或 60%–75% 的黄点，一次只遮一带；可见点重新切段并赋新编号，再分别用圆锥方法和独立直线方法预测。先存预测，再评分。共 76 个图带案例，但仍只有 38 张已使用开发图。</p>
<p>命中容差固定为 0.25 倍光学绳距；分母含全部隐藏点。截断误差为 min(|纵向误差|/绳距,1)，无预测按 1 计。相同 x 列集合匹配不确认具体同根身份。预设门槛：总体截断误差至少比直线降低 10%，总体、左右相机及两带命中率均不得下降超过 2 个百分点。</p>
<table><tr><th>范围</th><th>隐藏黄点分母</th><th>C2 命中</th><th>直线命中</th><th>C2 截断误差</th><th>直线截断误差</th></tr>{''.join(rows)}</table>
<p>本轮预设继续门槛：<b>{'通过' if ev['predeclared_continue_criterion_met'] else '未通过'}</b>。预测侧未匹配比例 C2 {pct(ev['overall']['conic']['reverse_unmatched_fraction'])} / 直线 {pct(ev['overall']['linear']['reverse_unmatched_fraction'])}，用于观察额外错线风险，但原黄点不完整，不能称为真实误报率。{link(args.evaluation/'summary.json','全部分组、逐图和失败明细')} · {link(HERE/'evaluation_protocol.json','预先固定的评价协议')}</p>
<p>总体通过仍包含局部退化；按累计截断误差的退化量排序，前三个案例为：</p><ul>{regressions}</ul>
<h2>4. 迭代与失败记录</h2><table><tr><th>运行</th><th>圆锥命中</th><th>圆锥截断误差</th><th>预设门槛</th></tr>{''.join(versions)}</table>
<p>此前 C1“整条边界统一圆锥拟合”分支做了三轮，均未通过主门槛；第三轮还使强眩光图的可见段退化，已结束该分支，未运行其全量400图。{link(HERE.parent/'camera_condition_conic_boundaries_20261025_v4/结果说明.html','C1失败与关闭记录')}。C2另设最多两轮局部缺口实验，固定输入、隐藏区域、分母、容差及通过门槛均沿用原协议，评分函数逐项核验相同。以上均属于同一开放开发集的追加迭代，不是独立盲测。只有选定源码冻结后才生成本版全量400图。</p>
<h2>5. 文件位置与复现</h2><table><tr><th>路径（相对于全量运行）</th><th>内容</th></tr><tr><td>smooth_overlays / overlays</td><td>平滑结果 / 叠加原黄点；左右相机分目录</td></tr><tr><td>boundaries/smooth_boundaries</td><td>全部拟合轮廓，原尺寸 4096×3000，前景255、背景0</td></tr><tr><td>boundaries/geometric_gap</td><td>离开观测的模型补线，0/255</td></tr><tr><td>boundaries/fit_near_observations</td><td>靠近观测的拟合部分；与补线不重叠，二者并集等于完整轮廓</td></tr><tr><td>boundaries/observed</td><td>原 B1 观测掩码，未改动</td></tr><tr><td>boundaries/records</td><td>每条曲线的系数、采样点、配对、回退和拒绝原因</td></tr><tr><td>roi/effective_masked</td><td>ROI 内原 RGB 保留，外部统一黑色0；与 B1 逐字节一致</td></tr><tr><td>qa_compare</td><td>固定38图的原图 / B1 / C2 三列对照</td></tr></table>
<p>{link(run/'run_context.json','环境/命令/源码哈希')} · {link(run/'run_log.jsonl','400图日志')} · {link(run/'integrity_audit.json','文件与像素审计')} · {link(HERE/'protocol.json','输入与实验协议')}</p>
<pre>&amp; D:\\precise_arrangement\\vision_prototype\\.venv\\Scripts\\python.exe &lt;本目录&gt;\\run_conic.py --mode all --output &lt;新的运行目录&gt; --workers 2</pre><p>拒绝覆盖已有运行目录。原图、B1、先前失败轮次和保留集均不改动；本次没有连接设备或改变运动权限。</p>
<h2>6. 仍保留的未知与失败</h2><ul><li>无平滑曲线图：{html.escape(', '.join(empty)) or '本运行没有空结果'}。仍计入固定总数，不删除。</li><li>分段拼接并不保证一阶导数严格连续。固定38图中713处可核对接缝的纵向差中位数2px、P95为5px、最大9px；20处超过1/4光学绳距，保留为需人工复核的细节。{link(args.pilot/'visible_and_join_diagnostic.json','接缝与可见段诊断')}</li><li>实际过曝区域没有独立真值，遮点检验只测与 B1 已检测黄点的几何一致性。</li><li>暗带、活动绳完整屏蔽、跨活动绳同根连接、三维接触、换层 t0/t1/t2 没有由本次得到验证。</li><li>相邻帧来自有限原录像；已知工字轮外径覆盖仍约700 mm，不能把400张当400个独立工况。</li></ul>
</html>'''
    (HERE/'结果说明.html').write_text(explanation,encoding='utf-8')
    print('Report written from saved evidence:',HERE/'结果说明.html')

if __name__=='__main__':main()
