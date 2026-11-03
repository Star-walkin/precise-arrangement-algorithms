"""从已保存的 C2 文件生成离线审阅页，不重算算法。"""
import argparse
import csv
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

BASE=Path(__file__).resolve().parent
DATA=BASE.parent/'camera_condition_images_20261009_v2'
B1=BASE.parent/'camera_condition_bright_rope_boundaries_20261025_v3/all_001'

def esc(x):return html.escape(str(x),quote=True)
def href(root,path):return quote(Path(os.path.relpath(path,root)).as_posix(),safe='/._-')
def link(root,path,label):
    if not path.exists():raise FileNotFoundError(path)
    return f'<a href="{href(root,path)}" target="_blank">{esc(label)}</a>'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();run=args.run
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    rows=[json.loads(x) for x in (run/'run_log.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=summary['planned']:raise ValueError('Missing log rows')
    cards=[]
    for row in rows:
        rel=Path(row['output_file']);role=rel.parent.name;name=rel.name;stem=rel.stem
        if row['status']!='completed':
            cards.append(f'<article data-role="{esc(row["camera_role"])}" data-search="{esc(stem)}" data-qa="0" data-gap="0"><h2>{esc(stem)}</h2><p>运行失败：{esc(row.get("error"))}</p></article>');continue
        smooth=run/'smooth_overlays'/role/(stem+'.jpg');evidence=run/'overlays'/role/(stem+'.jpg')
        files=[(DATA/rel,'原图'),(B1/'overlays'/role/(stem+'.jpg'),'B1基准叠图'),
               (run/'roi/effective_masked'/role/name,'黑背景原色ROI'),
               (run/'roi/reference_masks'/role/name,'精确ROI'),
               (run/'boundaries/observed'/role/name,'原B1黄点'),
               (run/'boundaries/smooth_boundaries'/role/name,'完整平滑曲线mask'),
               (run/'boundaries/geometric_gap'/role/name,'缺口推断mask'),
               (run/'boundaries/fit_near_observations'/role/name,'近观测拟合mask'),
               (run/'boundaries/records'/role/(stem+'.json'),'系数与诊断JSON')]
        filelinks=' · '.join(link(run,p,t) for p,t in files)
        compare=link(run,run/'qa_compare'/role/(stem+'.jpg'),'原图 / B1 / C2 三列对照') if row['qa_selected'] else ''
        visible=(f"{row['yellow_near_curve_fraction']:.1%}" if row.get('yellow_near_curve_fraction') is not None else '无')
        fit=json.loads((run/'boundaries/records'/role/(stem+'.json')).read_text(encoding='utf-8'))['fit']
        connected=sum(c.get('status')=='gap_endpoint_conic_geometric_inference' for c in fit.get('curves',[]))
        cards.append(f'''<article data-role="{esc(row['camera_role'])}" data-search="{esc(stem+' '+row['source_group'])}" data-qa="{int(row['qa_selected'])}" data-gap="{int(row['gap_pixels']>0)}">
<a class="zoom" href="{href(run,smooth)}" target="_blank"><img loading="lazy" src="{href(run,smooth)}" data-smooth="{href(run,smooth)}" data-evidence="{href(run,evidence)}" alt="{esc(stem)}"></a>
<section><h2>{esc(stem)}</h2><p>{'左相机' if row['camera_role']=='left' else '右相机'} · {esc(row['source_group'])}</p>
<p>曲线片段 {row['curve_count']} · 双端缺口段 {connected}<br>远离观测的模型像素 {row['gap_pixels']:,}<br>黄点靠近拟合曲线比例 {visible}（拟合一致性，非准确率）</p>
<p>{compare}</p><details><summary>掩码、原图与逐曲线数据</summary>{filelinks}</details></section></article>''')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C2 圆锥曲线补全审阅</title>
<style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#121c25;color:#e8f1f6;font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif}}header,main,footer{{padding:22px 4vw}}h1{{margin:0 0 10px;font-size:28px}}a{{color:#84ceff}}header p{{max-width:1100px}}.stats{{display:flex;gap:14px;flex-wrap:wrap}}.stats span{{background:#253847;padding:8px 14px;border-radius:7px}}nav{{position:sticky;top:0;background:#20313eef;padding:12px 4vw;z-index:2;display:flex;flex-wrap:wrap;gap:16px}}input,select{{background:#30485a;color:white;padding:5px;border:1px solid #627d90}}#count{{margin-left:auto}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px}}article{{background:#20313f;border:1px solid #385369;border-radius:9px;overflow:hidden}}article[hidden]{{display:none}}article img{{width:100%;height:280px;object-fit:contain;background:#07121a}}section{{padding:12px}}h2{{font-size:14px;overflow-wrap:anywhere}}section p{{font-size:12px;color:#b7cbd8;margin:7px 0}}details{{font-size:12px;line-height:2}}summary{{cursor:pointer}}footer{{color:#b7cbd8}}</style>
<header><h1>圆锥曲线补全 C2 · B1 黄点保持不变</h1>
<p>以用户认可的 <b>亮带基准 B1 / RB-B1.0.0</b> 为输入。每条平滑线都是几何模型；跨片段连接要求两端黄点和相邻曲线支持。活动绳和未知排除区保持空白。仅处理可见支持的横向跨度，单端不足时不强行外推。</p>
<p><b style="color:#4ce0ed">青色：靠近黄点的平滑拟合</b>　<b style="color:#ff57e8">紫色：模型线离观测较远的部分</b>　<b style="color:#ffe552">黄色：原 B1 观测（勾选显示）</b>。紫色既可能是缺口补线，也可能是局部拟合偏差，不能仅按颜色认定真实过曝补全。绿框仅为显示包络，精确范围见 ROI mask。</p>
<div class="stats"><span>固定输入 {summary['planned']}</span><span>完成 {summary['completed']}</span><span>运行失败 {summary['failed']}</span><span>有平滑曲线 {summary['nonempty_curve_images']}</span><span>有缺口推断 {summary['images_with_gap']}</span></div>
<p>{link(run,BASE/'protocol.json','实验协议')} · {link(run,run/'summary.json','运行汇总')} · {link(run,run/'run_context.json','代码与输入哈希')} · <a href="../结果说明.html">结果与限制说明</a></p></header>
<nav><label>相机 <select id="role"><option value="all">全部</option><option value="left">左相机</option><option value="right">右相机</option></select></label><label><input id="qa" type="checkbox">仅固定38图对照</label><label><input id="gap" type="checkbox">有缺口推断</label><label><input id="evidence" type="checkbox">叠加原黄点</label><input id="search" type="search" placeholder="文件名或来源组"><span id="count"></span></nav>
<main>{''.join(cards)}</main><footer>结果保存为0/255掩码；有效ROI内原色不变，外部统一黑色。过曝区没有逐像素真值，平滑和遮点一致性不等于真实边界准确率。没有设备接入或保留集使用。</footer>
<script>const cards=[...document.querySelectorAll('article')],role=document.querySelector('#role'),qa=document.querySelector('#qa'),gap=document.querySelector('#gap'),search=document.querySelector('#search'),evidence=document.querySelector('#evidence');function filter(){{let n=0;for(const c of cards){{const ok=(role.value==='all'||c.dataset.role===role.value)&&(!qa.checked||c.dataset.qa==='1')&&(!gap.checked||c.dataset.gap==='1')&&c.dataset.search.toLowerCase().includes(search.value.toLowerCase());c.hidden=!ok;if(ok)n++;}}document.querySelector('#count').textContent=n+' / '+cards.length;}}for(const c of [role,qa,gap,search])c.addEventListener('input',filter);evidence.addEventListener('change',()=>{{for(const im of document.querySelectorAll('img[data-smooth]')){{im.src=evidence.checked?im.dataset.evidence:im.dataset.smooth;im.parentElement.href=im.src;}}}});filter();</script></html>'''
    (run/'review.html').write_text(page,encoding='utf-8')
    print('Review generated:',len(cards),'cards')

if __name__=='__main__':main()
