"""将已保存C4输出组织成完整ROI/边界两种浏览入口，不改预测。"""
from pathlib import Path
import argparse
import html
import json
import os
from urllib.parse import quote
HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'camera_condition_images_20261009_v2'
def esc(v):return html.escape(str(v),quote=True)
def href(run,p):return quote(Path(os.path.relpath(p,run)).as_posix(),safe='/._-')
def link(run,p,label):
    if not p.is_file():raise FileNotFoundError(p)
    return f'<a href="{href(run,p)}" target="_blank">{esc(label)}</a>'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    rows=[json.loads(x) for x in (run/'run_log.jsonl').read_text(encoding='utf-8').splitlines()]
    cards=[]
    for r in rows:
        rel=Path(r['output_file']);role=rel.parent.name;stem=rel.stem;name=rel.name
        if r['status']!='completed':
            cards.append(f'<article data-role="{esc(r["camera_role"])}" data-search="{esc(stem)}" data-qa="0"><h2>{esc(stem)}</h2><p>运行失败：{esc(r.get("error"))}</p></article>')
            continue
        roi=run/'roi/surface_masked'/role/name
        overlay=run/'overlays'/role/name
        preview=run/'previews'/role/(stem+'.jpg')
        smooth=run/'smooth_previews'/role/(stem+'.jpg')
        groups=run/'group_overlays'/role/(stem+'.jpg')
        more=[(DATA/rel,'原图'),(roi,'完整ROI原色PNG'),(overlay,'全分辨率边界PNG'),
              (run/'roi/surface_masks'/role/name,'完整SAM ROI掩码'),
              (run/'roi/bright_reference_masks'/role/name,'旧亮ROI'),
              (run/'roi/effective_masks'/role/name,'排除遮挡后ROI'),
              (run/'boundaries/dark_observed'/role/name,'暗区观测点'),
              (run/'boundaries/modeled'/role/name,'完整曲线mask'),
              (run/'boundaries/inferred'/role/name,'推断部分mask'),
              (run/'boundaries/simple_c3_extension'/role/name,'直接延长C3对照'),
              (run/'boundaries/records'/role/(stem+'.json'),'每组点、拟合系数与诊断')]
        if (run/'roi/sam_reference_masks'/role/name).is_file():
            more += [(run/'roi/sam_reference_masks'/role/name,'原SAM2参考掩码'),
                     (run/'roi/sam_reference_masked'/role/name,'原SAM2参考原色图')]
        qa=link(run,run/'qa_compare'/role/(stem+'.jpg'),'ROI / 原C3 / 简单延长 / C4 四列对照') if r['qa_selected'] else ''
        cards.append(f'''<article data-role="{esc(r['camera_role'])}" data-search="{esc(stem+' '+r['source_group'])}" data-qa="{int(r['qa_selected'])}">
<a class="zoom" href="{href(run,overlay)}" target="_blank"><img loading="lazy" src="{href(run,preview)}" data-evidence="{href(run,preview)}" data-curve="{href(run,smooth)}" data-group="{href(run,groups)}" data-roi="{href(run,roi)}" data-full="{href(run,overlay)}" alt="{esc(stem)}"></a>
<section><h2>{esc(stem)}</h2><p>{'左相机' if r['camera_role']=='left' else '右相机'} · {esc(r['source_group'])}</p>
<p>完整ROI {r['surface_pixels']:,} px · 原亮ROI外占 {r['expanded_pixels']/max(1,r['surface_pixels']):.1%}<br>新暗区观测像素 {r['dark_observed_pixels']:,} · 原亮ROI外曲线 {r['modeled_outside_bright_pixels']:,} px<br>推断曲线 {r['inferred_pixels']:,} px（数量不是准确率）</p>
<p>{qa}</p><details><summary>原图、掩码与每组模型</summary>{' · '.join(link(run,p,t) for p,t in more)}</details></section></article>''')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整卷面ROI与C4边界</title>
<style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#101b23;color:#e5eff5;font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif}}header,main,footer{{padding:22px 4vw}}h1{{margin:0;font-size:27px}}a{{color:#8ed3ff}}header p{{max-width:1200px}}.stats{{display:flex;gap:12px;flex-wrap:wrap}}.stats span{{padding:8px 14px;background:#253a49;border-radius:7px}}nav{{position:sticky;top:0;background:#223743f5;padding:12px 4vw;z-index:2;display:flex;gap:14px;flex-wrap:wrap}}select,input{{background:#304b5e;color:white;padding:6px;border:1px solid #6d8797}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(370px,1fr));gap:18px}}article{{background:#203442;border:1px solid #3c586a;border-radius:8px;overflow:hidden}}article[hidden]{{display:none}}article img{{width:100%;height:330px;object-fit:contain;background:#050b10}}section{{padding:12px}}h2{{font-size:14px;overflow-wrap:anywhere}}section p,details{{font-size:12px}}summary{{cursor:pointer}}.note{{border-left:4px solid #c98e38;padding-left:12px}}#count{{margin-left:auto}}</style>
<header><h1>全卷面 C4 · 完整SAM2表面ROI与钢丝绳边界</h1>
<p>基准：用户已确认展示效果尚可的整组圆锥 <b>C3 / RB-C3.0.0</b>。工作ROI采用<b>SAM2 v2卷面与原亮带ROI的并集</b>，避免旧SAM漏分裁掉已接受区域；保留原色、外部统一为0，原SAM2另存供对照。重新运行全卷面活动绳屏蔽，筛选暗区黄点并归入已有视觉边界组，每组使用单条圆锥曲线。缺乏观测处以模型推断显示。</p>
<p><b style="color:#ffe552">黄色：原亮区观测</b>　<b style="color:#ffb400">橙黄：筛选后的暗区观测</b>　<b style="color:#4de1e8">青色：靠近观测的模型</b>　<b style="color:#ff55e5">紫色：缺口或外推模型</b>。橙色半透明为活动绳候选，蓝色为未知排除；绿边为SAM表面，灰边为旧亮ROI。</p>
<p class="note">SAM2区域不是确认标签；扩展ROI不证明已覆盖真实切点。暗区可能误分组或外推偏离，紫线不代表直接识别；没有亮区种子的行不会凭间距人为生成。图中未画出的部分与拒识会保留记录。</p>
<div class="stats"><span>固定输入 {summary['planned']}</span><span>完成 {summary['completed']}</span><span>运行异常 {summary['failed']}</span><span>有曲线 {summary['nonempty_curve_images']}</span><span>有新增暗区观测 {summary['images_with_dark_observations']}</span></div>
<p>{link(run,HERE/'protocol.json','初始实施协议')} · {link(run,HERE/'protocol_revision_002.json','第二轮ROI修正')} · {link(run,run/'summary.json','运行汇总')} · {link(run,run/'run_context.json','代码与输入哈希')} · <a href="../结果说明.html">结果、对照与限制</a></p></header>
<nav><label>显示 <select id="view"><option value="evidence">完整卷面边界＋黄点</option><option value="curve">完整卷面平滑曲线</option><option value="roi">完整卷面ROI原色图</option><option value="group">黄点分组</option></select></label><label>相机 <select id="role"><option value="all">全部</option><option value="left">左</option><option value="right">右</option></select></label><label><input id="qa" type="checkbox">仅固定38图对照</label><input id="search" type="search" placeholder="文件名或来源组"><span id="count"></span></nav>
<main>{''.join(cards)}</main><footer>全分辨率ROI PNG与边界叠图分别保存。二值mask统一0/255。未访问保留集、未连接设备、未修改在线模型或运动权限。</footer>
<script>const cards=[...document.querySelectorAll('article')],view=document.querySelector('#view'),role=document.querySelector('#role'),qa=document.querySelector('#qa'),search=document.querySelector('#search');function filter(){{let n=0;for(const c of cards){{const ok=(role.value==='all'||c.dataset.role===role.value)&&(!qa.checked||c.dataset.qa==='1')&&c.dataset.search.toLowerCase().includes(search.value.toLowerCase());c.hidden=!ok;if(ok)n++;}}document.querySelector('#count').textContent=n+' / '+cards.length;}}function changeView(){{for(const im of document.querySelectorAll('img[data-full]')){{im.src=im.dataset[view.value];im.parentElement.href=view.value==='roi'?im.dataset.roi:view.value==='group'?im.dataset.group:im.dataset.full;}}}}for(const c of [role,qa,search])c.addEventListener('input',filter);view.addEventListener('change',changeView);if(location.hash==='#roi'){{view.value='roi';changeView();}}filter();</script></html>'''
    (run/'review.html').write_text(page,encoding='utf-8')
    print('Review generated',len(cards))
if __name__=='__main__':main()
