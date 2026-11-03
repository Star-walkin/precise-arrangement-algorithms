"""全量审计后生成中文报告与输出哈希清单；不改变任何预测。"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from html import escape
from html.parser import HTMLParser
from urllib.parse import quote, urlsplit, unquote
import argparse
import json
import os
import stat
from experiment_io import HERE, ROOT, read_json, sha, dump, source_inventory

def link(path, label):
    href=quote(Path(os.path.relpath(path,HERE)).as_posix(),safe='/._-#')
    return '<a href="'+href+'">'+escape(label)+'</a>'

def ordinary_files(folder):
    result=[];todo=[folder]
    while todo:
        current=todo.pop()
        for p in current.iterdir():
            attributes=getattr(p.lstat(),'st_file_attributes',0)
            if p.is_symlink() or attributes & getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0x400):
                raise ValueError('Reparse point not allowed in new output: '+str(p))
            if p.is_dir():todo.append(p)
            elif p.is_file():result.append(p)
    return sorted(result)

def check_links(pages):
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            self.links.extend(v for k,v in attrs if k in ['href','src'] and v)
    count=0
    for page in pages:
        parser=Parser();parser.feed(page.read_text(encoding='utf-8'))
        for raw in parser.links:
            if raw.startswith('#'):continue
            u=urlsplit(raw)
            if u.scheme or u.netloc:continue
            path=page.parent/Path(unquote(u.path))
            if not path.is_file():raise ValueError('Broken local link: '+str(path))
            count+=1
    return {'pages':[str(p) for p in pages],'checked_local_links':count,'broken_links':0}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args();run=args.run.absolute()
    if run.parent!=HERE:raise ValueError('Run must be direct experiment child')
    targets=[HERE/'结果说明.html',HERE/'delivery_manifest.json',run/'output_manifest.json',HERE/'index_change_log.json']
    if any(p.exists() for p in targets):raise FileExistsError('Delivery already exists; never overwrite evidence')
    freeze=read_json(HERE/'freeze_all.json');pilot=Path(freeze['selected_pilot'])
    s=read_json(run/'summary.json');a=read_json(run/'independent_audit.json');ctx=read_json(run/'run_context.json')
    assert s['planned']==s['completed']==400 and s['failed']==0 and s['source_unchanged'] and s['exit_code']==0
    assert a['passed'] and a['expected_images']==400 and a['error_count']==0
    source,models=source_inventory()
    assert source==ctx['source_sha256']==freeze['source_sha256']
    assert models==ctx['model_sha256']==freeze['model_sha256']
    for p,h in freeze['evidence_sha256'].items():
        if sha(p)!=h:raise ValueError('Frozen evidence changed: '+p)
    # 将选定pilot的每图预测/绘图与全量同图逐字节比较，排除只对QA使用不同逻辑。
    compared=0;mismatches=[]
    for p in ordinary_files(pilot):
        rel=p.relative_to(pilot)
        if rel.parts[0] not in ['records','overlays','previews','baseline_previews','masks','candidate_masks']:
            continue
        other=run/rel;compared+=1
        if not other.is_file() or sha(p)!=sha(other):mismatches.append(rel.as_posix())
    if mismatches:raise ValueError('Pilot/full mismatch: '+str(mismatches[:10]))
    logs=[json.loads(line) for line in (run/'run_log.jsonl').read_text(encoding='utf-8').splitlines()]
    # 自动单选仍可能错接旧圈，必须在交付中展示其独立目视结论。
    single_review_path=run/'single_hypothesis_visual_review.json'
    singles={r['output_file'] for r in logs if r['connection_status']=='SINGLE_IMAGE_HYPOTHESIS'}
    single_review=read_json(single_review_path)
    assert {c['output_file'] for c in single_review['cases']}==singles
    single_table=''.join('<tr><td>'+link(run/'previews'/Path(c['output_file']).parent.name/(Path(c['output_file']).stem+'.jpg'),Path(c['output_file']).stem)+'</td><td>'+escape(c['judgement'])+'</td><td>'+escape(c['observations'])+'</td></tr>' for c in single_review['cases'])
    # 原始预测页面不修改，另建带事后复核提示的查看入口。
    checked=run/'review_checked.html'
    if checked.exists():raise FileExistsError(checked)
    checked_html=(run/'review.html').read_text(encoding='utf-8')
    notice='<p class="note"><b>事后逐图复核：</b>11张自动单选中，8张明确错接旧圈，2张仅有局部投影支持，1张图像不足；接触识别尚未达标。下方标注保留原预测，新增文字提示仅是独立复核。<a href="review.html">原始预测页</a> · <a href="single_hypothesis_visual_review.json">11张完整复核记录</a></p>'
    checked_html=checked_html.replace('</header>',notice+'</header>',1)
    for case in single_review['cases']:
        title='<h3>'+escape(Path(case['output_file']).stem)+'</h3>'
        badge='<p class="note"><b>独立复核：'+escape(case['judgement'])+'</b> '+escape(case['observations'])+'</p>'
        if checked_html.count(title)!=1:raise ValueError('Review case heading mismatch')
        checked_html=checked_html.replace(title,title+badge,1)
    checked.write_text(checked_html,encoding='utf-8')
    groups={}
    for row in logs:
        key=(row['camera_role'],row['source_group'])
        groups.setdefault(key,Counter())[row['free_status']]+=1
    group_table=''.join('<tr><td>'+escape(camera)+'</td><td>'+escape(group)+'</td><td>'+str(sum(c.values()))+'</td><td>'+str(c['observed'])+'</td><td>'+str(c['direction_only'])+'</td><td>'+str(c['unknown'])+'</td></tr>' for (camera,group),c in sorted(groups.items()))
    iterations=[]
    for p in sorted(HERE.glob('pilot_*/summary.json')):
        d=read_json(p)
        iterations.append('<tr><td>'+link(p.parent/'review.html',p.parent.name)+'</td><td>'+str(d['completed'])+'/'+str(d['planned'])+'</td><td>'+str(d['free_status'].get('observed',0))+'</td><td>'+str(d['connection_status'].get('SINGLE_IMAGE_HYPOTHESIS',0))+'</td><td>'+str(d['connection_status'].get('MULTI_CANDIDATE',0))+'</td><td>'+str(d['connection_status'].get('UNKNOWN',0))+'</td></tr>')
    reviews=[p for p in ordinary_files(HERE) if p.name.startswith('visual_review') and p.suffix=='.json' and p.parent.parent==HERE]
    review_links=' · '.join(link(p,str(p.relative_to(HERE))) for p in reviews)
    free=s['free_status'];conn=s['connection_status']
    third_note = ('第三轮修复了另一个具体漏检：强语义候选同时跨过自由区与卷面，整段原图精测被重复纹理干扰而失败。'
                  '对无效候选仅在同一语义分量的卷面外部分重试，未降低原有门槛。两张左标尺图的可见斜绳重新检出；'
                  '该变化及固定14图回归见选定轮次目视记录。' if pilot.name == 'pilot_003' else '')
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>活动绳 A1 · 400图试验报告</title><style>body{{max-width:1180px;margin:auto;padding:30px;font:16px/1.8 system-ui,'Microsoft YaHei';background:#f3f6f8;color:#193342}}a{{color:#006f9c}}h1{{line-height:1.4}}h2{{margin-top:32px}}.hero{{background:#dcecf5;padding:20px;border-radius:10px}}.warning{{border-left:4px solid #b97919;background:#fff1d7;padding:14px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #becdd6;padding:8px}}th{{background:#e0eaf0}}code{{overflow-wrap:anywhere}}li{{margin:6px 0}}</style>
<h1>活动绳直段与转弯候选 A1 · IR-A1.0.0</h1><p>2026-10-26 · 固定400张开放开发图片 · 全部离线</p>
<div class="hero"><p><b>{free.get('observed',0)}张</b>自动检出自由直段；{free.get('direction_only',0)}张仅保留方向或宽度不稳定候选；{free.get('unknown',0)}张未可靠检出。左右相机各200，400张均保存结果与状态。</p><p>{link(run/'review_checked.html#detected','查看自动检出活动绳的结果')} · {link(checked,'查看全部400张及漏检/不确定')} · {link(run/'selection.csv','400图筛选清单')}</p></div>
<p class="warning"><b>本轮结论：自由直段有明确改善，卷面内接续与接触点识别尚未达标。</b>11张自动单选中，8张明确错接旧圈，2张仅有局部投影支持，1张图像不足。此版保留为可回溯实验结果，不能把平滑曲线或自动单选当作正确接触点；11张均来自同一右相机录像，并非11个独立工况。</p>
<p class="warning">筛选结果是算法建议。原400图没有逐帧活动绳、宽度、弯曲位置或物理接触真值；历史开放标签同帧匹配为0。以下数量不是准确率。弯曲转折、相切圆锥与图像投影不能独立证明三维接触、同根身份或换层。</p>
<h2>1. 本次实现</h2><ol><li><b>独立检测自由直段。</b>复用原自动语义候选和历史自由绳剖面算法，在未增强原图上测双侧半峰边界；结合卷面外支撑和有限延长的空间关系剔除明显机架候选。未提供人工入口或接触点。</li>
<li><b>测量宽度与方向。</b>输出原图坐标、中心、两侧光学边界、宽度分布及方向敏感性。像素宽度不是毫米绳径，经验范围不是经过校准的统计置信区间。</li>
<li><b>直段接圆锥。</b>用C3已接受卷面边界构成邻带竞争，并将自由直段和相切抛物线组成有限平滑候选。曲率及过渡位置受几何约束；原图双侧证据另行检验。C3曲线仅作几何先验，不作为新绳实测点。</li>
<li><b>保留竞争和拒识。</b>最多6条候选，同池比较简单几何G0与加入原图证据的G2；只在候选实际分开的区域计算区分证据。不可见部分与实测部分分开，候选点/区间不升级为物理接触确认。</li></ol>
<p>绿色：自由段光学边界；黄色：原图剖面点；橙色：单选路径的局部原图支持；彩色虚线：不同几何候选；灰色断线：证据不可用；红色端帽/十字：图像转弯候选区间或候选位置。每张可打开原图、全分辨率结果、直线延长对照和完整JSON。</p>
<p class="warning"><b>红点不是已确认接触点。</b>UNKNOWN 图也会展示排名靠前的几何候选，其中可能包含被拒绝的路径，便于查看失败原因。是否选中请以图上连接状态及 JSON 的 selected_candidate_id 为准；未知时选中掩码全零。</p>
<h2>2. 全量输出与检查</h2><ul><li>运行 {s['completed']}/{s['planned']} 完成，异常{s['failed']}；结构审计400/400通过。</li><li>接续状态：单一图像假设{conn.get('SINGLE_IMAGE_HYPOTHESIS',0)}张、多候选{conn.get('MULTI_CANDIDATE',0)}张、未知{conn.get('UNKNOWN',0)}张。单一假设仍非独立验收的正确接触点。</li><li>选定38图试验与全量的{compared}个逐图文件字节完全一致。</li><li>原图、C3基准、C4输入掩码和旧失败记录保持不变；源码、模型及各输入哈希见运行上下文和输出清单。</li></ul>
<p>{link(run/'summary.json','全量汇总')} · {link(run/'independent_audit.json','独立结构审计')} · {link(run/'run_context.json','环境、命令、源码与模型哈希')} · {link(run/'run_log.jsonl','逐图日志')} · {link(run/'output_manifest.json','完整输出哈希清单')}</p>
<h2>3. 开发轮次与质量边界</h2><table><tr><th>轮次</th><th>处理</th><th>直段</th><th>单一假设</th><th>多候选</th><th>未知</th></tr>{''.join(iterations)}</table>
<p>首轮明确发现按钮盒水平边缘误检，以及beam轨迹跳到邻圈形成折段。第二轮在同38图中排除7处已发现的机架/按钮盒误检，其余27张的自由段中心、角度和宽度逐值不变；候选路径改为相切圆锥后不再出现抽查中首轮的折返。原轮次保留；绘图改为区间端帽，避免把两端直接连成红色三角形。该比较是开发诊断，不是400张准确率。</p>
<p>{third_note}</p>
<p>最终选择：{escape(freeze['decision'])}。{link(Path(freeze['selected_review']),'选定轮次原图目视记录')}。{review_links}</p>
<p class="warning">尚未验收：重叠区真实同根识别、切点准确位置、非观察区域光学宽度及不同规格泛化。SAM卷面还可能漏分暗区或误纳入手套/工具；短可见自由段、低对比与邻圈重复纹理仍可能导致拒识或错候选。不会用零输出、低残差或平滑外观代替这些验证。</p>
<h2>自动单选的逐图复核</h2><p>以下{len(singles)}张是算法自动选出的图像假设，原预测原样保留。表格是预测保存后的独立开发诊断，未回填算法，不等于接触点真值；发现的错接不能计为识别成功。{link(single_review_path,'完整逐图复核记录')}</p><table><tr><th>图像（点击查看结果）</th><th>目视判断</th><th>具体观察</th></tr>{single_table}</table>
<h2>4. 按相机和采集来源统计</h2><table><tr><th>相机</th><th>来源</th><th>图片</th><th>直段</th><th>不稳定候选</th><th>未检出</th></tr>{group_table}</table><p>相邻帧不是独立工况；本批没有独立盲测含义。</p>
<h2>5. 文件与复现入口</h2><ul><li><code>overlays/left_camera、right_camera</code>：400张4096×3000结果PNG。</li><li><code>previews、baseline_previews</code>：局部结果和同输入直线延长对照。</li><li><code>records</code>：自由段剖面、宽度、方向、最多6条候选、逐点证据状态、转弯区间及拒识理由。</li><li><code>masks、candidate_masks</code>：0/255的原图支持与推断mask分开保存；未知时没有伪造选中mask。</li></ul>
<p>{link(HERE/'free_segment.py','自由段识别代码')} · {link(HERE/'connection_core.py','圆锥接续代码')} · {link(HERE/'run_experiment.py','完整运行入口')} · {link(HERE/'protocol.json','预定协议')} · {link(HERE/'evaluation_scope.json','标签匹配与评价范围')}</p>
<p>复现使用既有 <code>D:\\precise_arrangement\\vision_prototype\\.venv\\Scripts\\python.exe</code> 执行本目录 <code>run_experiment.py --mode all --output &lt;新目录绝对路径&gt; --workers 2</code>。新目录须在本实验目录下；不能覆盖历史结果。</p><p>未连接相机/PLC/灯光/机构，未读取保留集，未改变线上模型、动作锁或权限。C3继续作为卷面边界基准。</p></html>'''
    (HERE/'结果说明.html').write_text(report,encoding='utf-8')
    before=read_json(HERE/'index_before.json')
    for p,h in before.items():
        if sha(p)!=h:raise ValueError('Index changed concurrently: '+p)
    text=f'''## 2026-10-26：活动绳直段与转弯候选 A1

新增 [活动绳 A1 / IR-A1.0.0]({(HERE/'结果说明.html').as_posix()})，使用同400张开放开发图片和冻结C3卷面曲线，离线筛选自由直段、测光学宽度/方向并生成直线接相切圆锥候选。[400图结果]({(run/'review.html').as_posix()})保留全部拒识和失败。全量400/400运行及结构审计通过；直段检出{free.get('observed',0)}张、不稳定候选{free.get('direction_only',0)}张、未可靠检出{free.get('unknown',0)}张；接续单一图像假设{conn.get('SINGLE_IMAGE_HYPOTHESIS',0)}、多候选{conn.get('MULTI_CANDIDATE',0)}、未知{conn.get('UNKNOWN',0)}。这些均为输出数量，没有逐帧真值可据此宣称准确率或物理接触确认。逐张复核自动单选已发现明确错接旧圈；重叠区接续与接触点识别未达标，此版仅为实验结果。首轮按钮盒误检、折段候选与后续修正单独保留；不把拒识算正确。未触碰保留集、设备和线上权限；C3基准不变。

'''
    text=text.replace((run/'review.html').as_posix(),checked.as_posix())
    changes=[]
    for p,h in before.items():
        path=Path(p);old=path.read_text(encoding='utf-8');first,rest=old.split('\n',1)
        path.write_text(first+'\n\n'+text+rest.lstrip('\n'),encoding='utf-8')
        changes.append({'path':p,'before_sha256':h,'after_sha256':sha(path)})
    dump(HERE/'index_change_log.json',changes)
    files=ordinary_files(run)
    inventory=[{'path':p.relative_to(run).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    dump(run/'output_manifest.json',{'run':str(run),'version':'IR-A1.0.0','image_count':400,
                                    'files':inventory,'self_excluded':True})
    dump(HERE/'delivery_link_audit.json',check_links([HERE/'结果说明.html',run/'review.html',checked,pilot/'review.html']))
    root_files=[p for p in HERE.iterdir() if p.is_file()]
    dump(HERE/'delivery_manifest.json',{'version':'IR-A1.0.0','created_utc':datetime.now(timezone.utc).isoformat(),
             'run':str(run),'selected_pilot':str(pilot),'pilot_full_files_compared':compared,'mismatches':[],
             'output_manifest_sha256':sha(run/'output_manifest.json'),
             'root_files':{p.name:sha(p) for p in root_files},'equipment_access':False,'reserved_inputs':False})
    print(json.dumps({'images':400,'hashed_files':len(files),'pilot_full_files_compared':compared,
                      'report':str(HERE/'结果说明.html'),'exit_code':0},ensure_ascii=False))

if __name__=='__main__':main()
