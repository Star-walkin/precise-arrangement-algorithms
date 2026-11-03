"""全量完成后形成可回溯交付包，图片结果与失败实验均不覆盖。"""
from pathlib import Path
from datetime import datetime,timezone
from html import escape
from urllib.parse import urlparse,unquote
import json,re
from experiment_io import HERE,ROOT,sha,read_json,dump,sources

def main():
    for name in ['pilot_visual_review_001.json','method_comparison.json','full_visual_review.json']:
        if not (HERE/name).exists():raise ValueError('Required independent report missing: '+name)
    if (HERE/'index_update_receipt.json').exists():raise ValueError('Already packaged; do not append indices twice')
    run=HERE/'all_001';s=read_json(run/'summary.json');a=read_json(run/'independent_audit.json')
    if s['completed']!=400 or s['failed'] or a['passed']!=400 or a['failed'] or a['global_errors']:
        raise ValueError('Complete run and structural audit required')
    if sources()!=read_json(HERE/'freeze_all.json')['source_sha256']:raise ValueError('Source changed')
    # 在两个运行都结束后逐字节比较49个重复输入的图像和记录；不以此当真值。
    pilot=HERE/'pilot_003_global_gamma';comparisons=[]
    for item in read_json(pilot/'run_context.json')['selection']:
        p=Path(item);role=p.parent.name;stem=p.stem
        for folder in ['enhanced_roi','overlays','enhanced_overlays','previews','enhanced_previews',
                       'masks/c3_observed','masks/c3_smooth','masks/free_observed','masks/connection_observed','masks/connection_inferred','records']:
            suffix='.json' if folder=='records' else '.jpg' if folder in ['previews','enhanced_previews'] else '.png'
            rel=Path(folder)/role/(stem+suffix)
            comparisons.append({'path':rel.as_posix(),'identical':sha(run/rel)==sha(pilot/rel)})
    dump(run/'pilot_repeat_check.json',{'files_checked':len(comparisons),'mismatches':[c for c in comparisons if not c['identical']],
        'meaning':'Deterministic repeat check, not accuracy','results':comparisons})
    if any(not c['identical'] for c in comparisons):raise ValueError('Pilot/full mismatch')
    cases=[json.loads(l) for l in (run/'run_log.jsonl').read_text(encoding='utf-8').splitlines()]
    selected=[r for r in cases if r['free_status']=='observed']
    dump(run/'detected_images.json',{'criterion':'Frozen A1 observed free span; not manual presence labels',
        'total_input_denominator':400,'count':len(selected),'files':[r['output_file'] for r in selected]})
    rows=[]
    for d in sorted(HERE.glob('pilot_*')):
        if not d.is_dir() or not (d/'summary.json').exists():continue
        q=read_json(d/'summary.json')
        rows.append(f'<tr><td><a href="{d.name}/review.html">{d.name}</a></td><td>{q["new_c3_points"]:,}</td><td>{q["new_c3_curves"]}</td><td>{escape(json.dumps(q["connection_status"]))}</td></tr>')
    selected_count=len(selected)
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>A2 活动绳实验结果</title>
<style>body{{max-width:1140px;margin:32px auto;padding:0 22px;font:17px/1.8 system-ui,'Microsoft YaHei';color:#20323d;background:#f2f6f8}}h1,h2{{line-height:1.4}}section{{padding:22px;background:white;margin:20px 0;border-radius:10px}}a{{color:#0069a3}}.warn{{background:#fff0d5;border-left:6px solid #c27600;padding:18px}}.buttons a{{display:inline-block;padding:12px 18px;background:#145b76;color:white;margin:5px;border-radius:6px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #cad8dd;padding:8px;overflow-wrap:anywhere}}code{{overflow-wrap:anywhere}}.small{{font-size:14px}}li{{margin:6px 0}}</style>
<h1>A2 · 活动绳自由段精修与暗区增强</h1><p>实验版本 <b>IR-A2.0.0</b> · 2026-10-26 · 离线开放开发400图，左/右各200图。</p>
<div class="warn"><b>已实现并输出，接续识别尚未验收通过。</b> 本轮能精修可见自由段、提亮暗区并重跑C3，但开发对照发现过平行偏移和邻旧圈误接，最终暗区路径仍未确认。曲线拟合与局部双边响应都不能证明同根身份或真实接触。本版保留为实验结果，C3与A1原基准不被替换。</div>
<div class="buttons"><a href="all_001/review.html#detected">查看{selected_count}张已检出自由绳</a><a href="all_001/review.html">查看全部400张</a><a href="all_001/selection.csv">逐图筛选与状态清单</a></div>
<section><h2>本轮做了什么</h2><ol>
<li><b>自由段双边精修：</b>在原图卷面外，分别寻找两侧半峰交点和相反梯度峰，再以鲁棒共同拟合得到中心线、倾角和光学宽度。数据不足时保留A1结果并记录原因。宽度单位是像素，未标定为物理绳径。</li>
<li><b>全局增强：</b>对照原图、全局 Gamma 0.70、带限增益的低频照明场补偿。照明场在暗侧放大紫绿块噪声，完整400图采用较温和的 Gamma 0.70。它提亮灰度，不恢复被欠曝光丢失的信息。</li>
<li><b>重新跑C3：</b>在自由绳附近卷面走廊内重新搜索黄点，用原始图像复测，再调用原C3归类与每组全点拟合。C3此处实际是抛物线模型。旧C3曲线仅用于数量对照，没有把旧曲线当新检测结果。</li>
<li><b>双边接续候选：</b>相邻C3沟槽间距接近自由绳光学宽度时，生成有限、切向连续的过渡曲线；必须与直行假设竞争，且过渡本身有原图双边支持。拒绝候选和原因完整保留。</li></ol></section>
<section><h2>实际输出</h2><table><tr><th>项目</th><th>数量 / 说明</th></tr>
<tr><td>输入 / 完成 / 异常</td><td>400 / {s['completed']} / {s['failed']}</td></tr>
<tr><td>自由段状态</td><td>{escape(json.dumps(s['free_status'],ensure_ascii=False))}</td></tr>
<tr><td>双边精修</td><td>{escape(json.dumps(s['refinement_status'],ensure_ascii=False))}</td></tr>
<tr><td>接续状态</td><td>{escape(json.dumps(s['connection_status'],ensure_ascii=False))}</td></tr>
<tr><td>新C3观测点 / 曲线</td><td>{s['new_c3_points']:,} / {s['new_c3_curves']:,}，仅为输出量</td></tr>
<tr><td>独立文件/像素审计</td><td>{a['passed']}/400通过；49张重复运行539个图像/记录逐字节一致。这些检查不是视觉准确率。</td></tr></table>
<p>自动筛选沿用A1已检出的自由段，{selected_count}张不是人工确认的活动绳总数。其余图仍保留在400的分母内。未获得可用自由段的图有增强ROI，但局部接续不强制输出。ROI沿用C4已有卷面掩码，本次不重做SAM分割；原有边界漏面或包含遮挡物的风险仍在。</p>
<ul><li><code>all_001/enhanced_roi/</code>：400张原分辨率增强卷面ROI，外部统一0；左右相机分目录。</li>
<li><code>all_001/overlays/</code>与<code>enhanced_overlays/</code>：分别400张原图/增强图识别叠图。</li>
<li><code>all_001/records/</code>：400条JSON，含倾角、光学宽度、原图观测、新C3模型、候选/拒绝原因和输入哈希。</li>
<li><code>all_001/masks/</code>：自由段、C3观测/拟合、接续观测/推断掩码，0/255二值。</li></ul></section>
<section><h2>怎样看图</h2><p>绿线＝卷面外自由绳双边；黄点＝新C3局部原图复核点；青边/橙线＝候选位置存在局部双边响应；紫虚线＝几何推断；暗蓝细线＝新C3卷面拟合。局部边缘响应可能来自旧圈，并不等于整条路径正确。</p>
<p><b>STRAIGHT_SUPPORTED</b>：直行候选有局部支持，不能确认整个暗侧尾段；<b>CURVE_CANDIDATE</b>：图像分数接受的弯曲候选，仍需查看误接；<b>UNKNOWN / AMBIGUOUS</b>：无法确定，不计作正确。红十字若出现，只是图像过渡假设，不是真实接触点。</p></section>
<section><h2>三轮开发与保留失败</h2><p>固定49张＝原QA38张与A1全部11张单选案例的并集。不同处理都用同一清单、统一参数；独立审图先固定25张。均为已开放开发资料，不是盲测。</p>
<table><tr><th>运行</th><th>新黄点</th><th>新曲线</th><th>接续状态（非准确率）</th></tr>{''.join(rows)}</table>
<p>首轮发现“旧圈起点就必须平行于直绳”的几何门过严；第二轮改为有限转角与曲率约束，并把外推起点投影回整段拟合直线。第二轮仍有从证据缺口跳入旧圈的候选；第三轮增加过渡本身连续观测门。门槛阻止部分候选，但没有证明剩余路线正确。按预设预算停止继续堆评分，保留三轮结果。</p>
<p>首轮固定25图独立审阅中，A1的11张单选案例有8张仍见明显平行错位、2张仅局部投影接近、1张暗侧不可判；例如右侧 <code>R_run_stop_run_20260922_01__f000451</code>。这些是首轮定性诊断，不冒充终轮400图真值评分。</p>
<p>终轮单独复核 f000451：在修正外推起点后的 Gamma 原分辨率叠图中，另一位审阅者未发现可见观测范围贯穿整段的明显平移；暗侧推断尾段仍无法验证。两次判读的版本与观察范围不同，只作为局部定性诊断，不据此推出208张直行结果的改善率或整绳准确率。终轮没有接受弯曲候选。亮度变高、黄点增多或由弯线改为直线均不作为整条路径修复成功。</p></section>
<section><h2>代码与可追溯记录</h2><p><a href="free_refinement.py">自由段精修</a> · <a href="illumination_c3.py">增强与新C3观测</a> · <a href="paired_connection.py">接续候选</a> · <a href="run_experiment.py">批处理</a></p>
<p><a href="protocol.json">固定协议</a> · <a href="freeze_all.json">全量冻结</a> · <a href="evaluation_scope.json">独立审图范围</a> · <a href="pilot_visual_review_001.json">首轮25图诊断</a> · <a href="full_visual_review.json">终轮视觉诊断</a> · <a href="method_comparison.json">方法与接续门槛对照</a> · <a href="freeze_visual_notes.json">定稿前观察记录</a> · <a href="all_001/summary.json">全量统计</a> · <a href="all_001/independent_audit.json">独立审计</a> · <a href="all_001/run_context.json">命令/环境/源码快照</a> · <a href="all_001/output_manifest.json">输出哈希清单</a></p>
<p class="small">首次全量审计误把15张“仅方向”状态当作完全未知，得到385/400；<a href="all_001/audit_diagnostic_002.json">首次诊断</a>原样保留。修正的是审计状态契约，检测图像和记录未修改，随后全400重审通过。</p>
<p class="small">来源及图像逐项哈希见固定原400 manifest与逐图record；原图、C3、C4、A1均不覆盖。无设备连接、无运动权限修改、无保留集读取、无线上替换。单帧几何不确认同根、接触或t0/t1/t2。</p></section></html>'''
    (HERE/'结果说明.html').write_text(body,encoding='utf-8')
    # 所有导航仅追加新证据；先确认用户/其他任务未改动索引。
    before=read_json(HERE/'index_before.json')
    for name,expected in before.items():
        if sha(Path(name))!=expected:raise ValueError('Concurrent index edit: '+name)
    rels={ROOT/'数据资产/README.md':'05_派生与实验结果/',HERE.parent/'README.md':'',ROOT/'工程导航/05_实验版本档案/README.md':'../../数据资产/05_派生与实验结果/'}
    changes=[]
    for path,prefix in rels.items():
        link=prefix+HERE.name+'/结果说明.html'
        text=f'\n\n## 活动绳增强实验 A2 / IR-A2.0.0（2026-10-26）\n\n[结果与400图查看入口]({link})。沿用同一开放400图，新增原图自由段双边精修、全局Gamma 0.70增强后重跑C3及带直行竞争的接续候选。400/400完成、异常0，独立结构审计400/400通过；自动自由段检出{selected_count}张，精修接受{s["refinement_status"].get("accepted",0)}张，其余保留A1状态。实验存在平行偏移/邻旧圈误接，接续身份和接触点未验收通过，不替换C3或A1基准。固定49图三轮开发对照、25图独立诊断与失败保留；点数/曲线数/审计通过不等于准确率。原图与旧冻结结果不改，无保留集或设备访问。\n'
        with path.open('a',encoding='utf-8',newline='') as f:f.write(text)
        changes.append({'path':str(path),'before':before[str(path)],'after':sha(path)})
    dump(HERE/'index_update_receipt.json',{'date':'2026-10-26','files':changes})
    # 清单生成在运行和审计之后；本身不纳入自身哈希。
    files=[{'path':p.relative_to(run).as_posix(),'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(run.rglob('*')) if p.is_file() and p.name!='output_manifest.json']
    dump(run/'output_manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'files':files,'count':len(files)})
    missing=[];checked=0
    for page in [HERE/'结果说明.html',run/'review.html']:
        for ref in re.findall(r'(?:href|src)="([^"]+)"',page.read_text(encoding='utf-8')):
            u=urlparse(ref)
            if u.scheme or not u.path:continue
            checked+=1
            if not (page.parent/unquote(u.path)).exists():missing.append({'page':str(page),'ref':ref})
    dump(HERE/'package_validation.json',{'local_links_checked':checked,'missing':missing,
        'output_manifest_sha256':sha(run/'output_manifest.json'),'files':len(files),'pilot_repeat_identical':True,
        'complete':not missing,'hardware_access':False,'reserved_access':False})
    if missing:raise ValueError('Broken output links')
    dump(HERE/'release_manifest.json',{'version':'IR-A2.0.0','kind':'experimental_not_new_baseline',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'root_files':[{'path':p.name,'sha256':sha(p)} for p in sorted(HERE.iterdir())
                      if p.is_file() and p.suffix in ['.py','.json','.html'] and p.name!='release_manifest.json'],
        'full_output_manifest':'all_001/output_manifest.json','full_output_manifest_sha256':sha(run/'output_manifest.json')})
    print(json.dumps({'files':len(files),'links':checked,'manifest_sha256':sha(run/'output_manifest.json')},ensure_ascii=False))

if __name__=='__main__':main()
