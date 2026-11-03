"""根据已完成、已审计的运行生成中文交付报告；不运行分割算法。"""
from pathlib import Path
from collections import Counter
import hashlib
import html
import json
from datetime import datetime, timezone

BASE=Path(__file__).resolve().parent
RUN=BASE/'all_001'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def esc(v):
    return html.escape(str(v),quote=True)

def main():
    summary=read(RUN/'summary.json')
    audit=read(RUN/'integrity_audit.json')
    context=read(RUN/'run_context.json')
    frozen=read(BASE/'freeze_all_001.json')
    occlusion=read(RUN/'occlusion_analysis_001.json')['total']
    assert summary['completed']==400 and summary['failed']==0
    assert audit['passed']==400 and audit['failed']==0 and not audit['global_errors']
    assert context['source_sha256']==frozen['source_sha256']
    for p,s in frozen['source_sha256'].items():
        assert sha(Path(p))==s,p
    records=[json.loads(l) for l in (RUN/'run_log.jsonl').read_text(encoding='utf-8').splitlines()]
    role_rows=[]
    for role,label in [('left','左相机'),('right','右相机')]:
        rr=[r for r in records if r['camera_role']==role]
        role_rows.append(f"<tr><td>{label}</td><td>{len(rr)}</td><td>{sum(r['observed_pixels']>0 for r in rr)}</td>"
                         f"<td>{sum(r['v2_observed_points'] for r in rr):,}</td><td>{sum(r['v3_observed_points'] for r in rr):,}</td>"
                         f"<td>{sum(r['inferred_pixels'] for r in rr):,}</td></tr>")
    called=sum(r['occlusion_diagnostics']['history_module_called'] for r in records)
    valid=sum(r['occlusion_diagnostics']['history_valid'] for r in records)
    supported=sum(r['rope_pixels']>0 for r in records)
    unknown=sum(r['unknown_pixels']>0 for r in records)
    empty=[r['output_file'] for r in records if r['observed_pixels']==0]
    percentages=(summary['v3_observed_points']/summary['v2_observed_points']-1)*100
    group_rows=''.join(f"<tr><td>{esc(g['camera_role'])}</td><td>{esc(g['source_group'])}</td>"
        f"<td>{g['planned']}</td><td>{g['completed']}</td><td>{g['v2_points']:,}</td><td>{g['v3_points']:,}</td></tr>" for g in summary['groups'])
    source_rows=''.join(f"<tr><td><a href='{name}'>{name}</a></td><td>{description}</td></tr>" for name,description in [
        ('full_roi_observation.py','整个 ROI 独立扫描、动态对比门槛、暗谷片段跟踪'),
        ('curve_family_v3.py','共享曲率拟合、片段关联、端点约束、逐像素拒识'),
        ('free_rope_v3.py','历史活动绳检测器、原图斜带双侧对比、未知区回退'),
        ('run_v3.py','固定清单、文件哈希、掩码契约、输出及日志'),
        ('audit_v3.py','独立检查原图与输出文件、像素关系和点列'),
        ('build_review_v3.py','400 图离线筛选与固定 38 图并排对照')])
    examples=[('left_camera/dual_20260923_113305__pair0315__left.jpg','左相机强眩光：两侧可见暗沟增加，中心白带留白'),
              ('right_camera/dual_20260924_111500__pair0315__right.jpg','右相机：旧版漏掉的多排可见暗沟现在有候选'),
              ('right_camera/ruler_right_02.jpg','活动绳限制：橙色仅局部支持，蓝色为未知排除区，白色上缘仍可能漏屏蔽')]
    example_html=''.join(f"<figure><a href='all_001/qa_compare/{p}'><img loading='lazy' src='all_001/qa_compare/{p}'></a><figcaption>{esc(t)}。从左到右：原图 / v2 / v3。</figcaption></figure>" for p,t in examples)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>卷面边界 v3：结果与实际改进范围</title><style>
body{{font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif;color:#193040;background:#f2f5f7;margin:0}}main{{max-width:1120px;margin:auto;padding:28px}}
h1{{font-size:30px}}h2{{margin-top:32px}}a{{color:#075a9e}}.box{{background:white;padding:20px;border:1px solid #c8d6df;border-radius:10px;margin:14px 0}}
.warn{{border-left:5px solid #b16b15}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #cad6de;padding:8px;text-align:left}}
th{{background:#e2ebf2}}code,pre{{background:#e6edf2;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;padding:14px}}img{{width:100%;height:auto}}figure{{margin:22px 0}}figcaption{{color:#495d6b;font-size:14px}}
details{{margin:16px 0}}summary{{cursor:pointer;font-weight:bold}}.small{{font-size:14px;color:#4b606f}}</style><main>
<h1>卷面钢丝绳视觉边界 v3</h1><p>2026-10-25 · 同一组开放开发图 · 左右各 200 张 · 推荐运行 <b>all_001</b></p>
<div class="box"><a href="all_001/review.html"><b>打开 400 张结果审阅页</b></a> · <a href="all_001/qa_compare">固定 38 张原图 / v2 / v3 对照</a> · <a href="all_001/summary.json">运行汇总</a> · <a href="all_001/integrity_audit.json">独立像素审计</a></div>
<h2>这次具体改了什么</h2><ol>
<li><b>黄色：在整个 ROI 重新找暗沟。</b>每个采样列独立计算原图亮度的局部极小值，结合两侧亮度差、局部动态阈值和绳距抑制相邻细纹，再把平缓、连续的点组成片段。旧黄色坐标和旧线号不再作为入口；仍复用旧版每图光学绳距作为尺度提示。</li>
<li><b>紫色：多个片段共同约束走势。</b>拟合 yᵢ(x)=aᵢ+b·u+c·u² 的共享曲线族。每条片段有自己的位置 aᵢ，共用斜率与弯曲趋势 b、c；缺口关联还检查两端位置、切线和至少两条邻绳。普通缺口必须通过原图暗谷与曝光检查，未知遮挡不补。</li>
<li><b>活动绳：真正调用旧检测器，再检查卷面交叉处。</b>自动语义候选提供搜索范围，历史 <code>vision_geometry_lab/rope_detector.py::detect_rope</code> 测自由段方向和光学宽度。历史亮条剖面不适合亮卷面上的暗绳时，再按该方向展开原图，检查连续斜带的双侧对比。局部支持标橙，未验证的旧候选标蓝并排除。</li>
<li><b>绿框显示：</b>叠图只显示平滑外包络，避免活动绳附近掩码孔洞边缘挤成一团。显示外包络不参与计算；精确范围请看 reference ROI mask。</li></ol>
<div class="box warn"><b>仍未解决：</b>完整活动绳遮挡范围、跨遮挡的可靠逐绳接续、绳股纹假阳性和每根绳的真实身份。曲线模型是局部共享二次曲线族，并非完整椭圆拟合；输出也不是逐根真值或设备控制结果。</div>
<h2>实际运行结果</h2><table><tr><th>相机</th><th>完成图数</th><th>非空黄点图</th><th>v2 黄点</th><th>v3 黄点</th><th>v3 紫色像素</th></tr>{''.join(role_rows)}</table>
<p>黄点总数 {summary['v2_observed_points']:,} → {summary['v3_observed_points']:,}，原始计数增加 {percentages:.1f}%；{summary['images_with_more_observed_points']} 张图增加。<b>这不是准确率或召回率提升</b>：采样步长略有不同（v2 约 0.32p，v3 约 0.30p），点密度与候选覆盖都会改变计数；尚无独立逐根边界真值。</p>
<p>紫色仅 {summary['inferred_pixels']} 像素，分布在 {sum(r['inferred_pixels']>0 for r in records)} 张图；其中 {summary['inferred_under_occlusion_pixels']} 像素位于橙色候选内。少量跨遮挡像素只是几何提案，<b>没有验证为正确接续</b>。紫线减少体现保守拒识，不能当成正确率提高。</p>
<p>历史活动绳检测调用 {called} 张、返回有效自由段 {valid} 张；{supported} 张存在局部原图支持区，{unknown} 张仍有未知排除区。这些计数均不是活动绳完整检出率。与 v2 相比，实际新增排除 {occlusion['newly_excluded_pixels']:,} 像素，分布在 {occlusion['images_with_newly_excluded']} 张，旧排除范围未缩小；多数颜色变化只是支持证据的重新分类。<a href="all_001/occlusion_analysis_001.json">完整对照</a>单独记录。</p>
<p>非空边界 399/400；空输出保留：<code>{esc(empty[0])}</code>。该图所有候选片段均未达到连续长度要求，未强行补线。</p>
<p><a href="all_001/postrun_visual_diagnostics.json">冻结后的 3 个重点诊断</a>保留了唯一跨橙色区域紫线、最长局部补线、唯一空输出的原图局部对照；只诊断，不用这些全量结果继续调参。</p>
<details><summary>按来源组查看（相邻帧不是独立工况）</summary><table><tr><th>相机</th><th>来源组</th><th>计划</th><th>完成</th><th>v2 黄点</th><th>v3 黄点</th></tr>{group_rows}</table></details>
<h2>看图时重点检查</h2><p>黄色是原图可见暗沟候选；紫色是推断；橙色是活动绳局部图像支持；蓝色是不确定的排除区。橙色变多不一定扩大了遮挡范围，可能只是把原蓝区重新分类。以下例子同时展示覆盖改善和剩余失败。</p>{example_html}
<h2>文件和像素约定</h2><ul>
<li><code>all_001/roi/reference_masks</code>：旧亮带 ROI 文件逐字节复制。</li>
<li><code>all_001/roi/effective_masks</code>：参考 ROI 减去橙色、蓝色排除区；<code>effective_masked</code> 内保持原图 RGB，外部统一为 0（黑色）。</li>
<li><code>all_001/boundaries/observed</code> / <code>inferred</code>：分别保存黄色候选和紫色推断，所有 mask 统一 0/255。</li>
<li><code>incoming_rope_occlusion</code> / <code>unknown_exclusion</code> / <code>inferred_under_occlusion</code>：局部绳支持、未知区和遮挡内紫线分别保存；<code>records</code> 存逐片段点列、拟合和拒绝原因。</li></ul>
<p>黄色输出为半径 1 像素的采样点；紫色为单像素点列。缩略图显示密度不等于每根绳的真实边界完整度。</p>
<h2>可复核证据和代码</h2><table><tr><th>文件</th><th>用途</th></tr>{source_rows}</table>
<p>固定 38 图第一次整合 <a href="integration_pilot_001/visual_review_pilot_001.json">保留了眩光误补线问题</a>；第二次整合收紧端点、像素支持和曝光门槛后通过 38/38 像素审计，才冻结执行 400 图。没有用全量结果回头调参。模块先导调试目录均保留，正式可追溯运行以 integration_pilot_001/002 和 all_001 为准。</p>
<p>全量独立检查：{audit['passed']}/{audit['planned']} 通过，{audit['failed']} 失败，{audit['file_manifest_rows']} 个输出文件有哈希清单。检查包括原图哈希、旧 ROI 字节、0/255、黄紫遮挡关系、RGB 保真、点列与位图双向一致。<b>文件检查通过不等于分割准确率通过。</b></p>
<p><a href="protocol.json">预定范围与门槛</a> · <a href="freeze_all_001.json">全量冻结记录</a> · <a href="all_001/run_context.json">命令、环境、源码和模型哈希</a> · <a href="all_001/run_log.jsonl">逐图日志</a> · <a href="all_001/pilot_full_consistency.json">固定 38 图与全量一致性</a></p>
<pre>&amp; 'D:\\precise_arrangement\\vision_prototype\\.venv\\Scripts\\python.exe' '{esc(BASE/'run_v3.py')}' --mode all --workers 2 --output '新的独立输出目录'</pre>
<p class="small">来源仍为 camera_condition_images_20261009_v2 固定清单。400 图为已开放开发素材，非新增盲测，不能当 400 个独立工况；已确认录像工字轮外径约 700 mm，未证明多种外径适用。本次没有连接设备、改变控制权限或使用保留集；原始素材及旧实验文件未覆盖。</p>
</main></html>'''
    (BASE/'结果说明.html').write_text(page,encoding='utf-8')
    print('Report written; audited files:',audit['file_manifest_rows'])

if __name__=='__main__':main()
