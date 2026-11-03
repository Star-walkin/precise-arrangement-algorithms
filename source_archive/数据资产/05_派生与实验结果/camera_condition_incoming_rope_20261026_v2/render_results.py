"""A2 原图/增强图一致坐标绘图，区分观测边缘与几何推断。"""
from pathlib import Path
from html import escape
from urllib.parse import quote
import os
import json
import cv2
import numpy as np
from experiment_io import DATA,A1,read_json,write_image

def line_segments(canvas,path,states,color,thickness=2):
    p=np.asarray(path,float)
    if p.ndim!=2 or len(p)<2:return
    for i in range(len(p)-1):
        if not np.isfinite(p[i:i+2]).all() or np.linalg.norm(p[i+1]-p[i])>40:continue
        state=states[i] if states else 'observed'
        if state=='unavailable':continue
        if state=='inferred' and (i//3)%2:continue
        col=color if state=='observed' else (200,80,200)
        cv2.line(canvas,tuple(np.rint(p[i]).astype(int)),tuple(np.rint(p[i+1]).astype(int)),col,thickness,cv2.LINE_AA)

def render(image,enhanced,surface,free,c3,connection):
    canvases=[image.copy(),enhanced.copy()]
    for canvas in canvases:
        contours=cv2.findContours(surface,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(canvas,contours,-1,(130,150,130),1)
        for curve in c3.get('fit',{}).get('curves',[]):
            points=curve.get('sampled_points_xy',[])
            line_segments(canvas,points,['observed']*len(points),(110,90,40),1)
        for line in c3.get('observed_lines',[]):
            pp=line.get('observed_points_xy',line.get('points',line.get('points_xy',[])))
            if len(pp):
                for q in np.asarray(pp)[::3]:
                    cv2.circle(canvas,tuple(np.rint(q[:2]).astype(int)),2,(0,220,255),-1)
        for key in ['boundary_left_xy','boundary_right_xy']:
            p=free.get(key,[]);line_segments(canvas,p,['observed']*len(p),(30,255,60),3)
        for q in free.get('observed_points_xy',[])[::3]:
            cv2.circle(canvas,tuple(np.rint(q).astype(int)),2,(0,255,255),-1)
        selected=connection.get('selected_candidate_id')
        candidates=connection.get('candidates',[])
        display=[c for c in candidates if c['candidate_id']==selected]
        # 未选中时不把直行对照涂成活动绳结果；全部竞争候选保存在JSON。
        for c in display:
            states=c['point_states']
            for key in ['boundary_left_xy','boundary_right_xy']:
                line_segments(canvas,c[key],states,(255,220,30),2)
            line_segments(canvas,c['path_xy'],states,(0,160,255),2)
            if c.get('transition_center_xy') is not None:
                cv2.drawMarker(canvas,tuple(np.rint(c['transition_center_xy']).astype(int)),(20,30,255),cv2.MARKER_CROSS,20,2)
    y,x=np.where(surface>0)
    extra=np.asarray(free.get('observed_points_xy',[]),float)
    if len(extra):x=np.r_[x,extra[:,0]];y=np.r_[y,extra[:,1]]
    crop=[max(0,int(x.min())-65),max(0,int(y.min())-90),min(image.shape[1],int(x.max())+66),min(image.shape[0],int(y.max())+66)] if len(x) else [0,0,image.shape[1],image.shape[0]]
    def preview(a,label):
        x0,y0,x1,y1=crop;tile=a[y0:y1,x0:x1]
        scale=min(1.,1450/tile.shape[1]);tile=cv2.resize(tile,(round(tile.shape[1]*scale),round(tile.shape[0]*scale)))
        out=np.zeros((tile.shape[0]+62,tile.shape[1],3),np.uint8);out[62:]=tile
        width=free.get('width_px');angle=free.get('angle_deg')
        text=f'IR-A2 | {label} | free={free.get("status")} | {connection.get("status")}'
        cv2.putText(out,text,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.50,(230,230,230),1,cv2.LINE_AA)
        detail=f'optical width={width:.2f}px, angle={angle:.2f}deg' if width is not None and angle is not None else 'optical width unavailable'
        cv2.putText(out,detail+' | green=free edges, cyan=local edge support, purple=inferred',(10,47),cv2.FONT_HERSHEY_SIMPLEX,.43,(220,220,220),1,cv2.LINE_AA)
        return out
    return canvases[0],canvases[1],preview(canvases[0],'original'),preview(canvases[1],'normalized'),crop

def url(p,base):return quote(Path(os.path.relpath(p,base)).as_posix(),safe='/._-#')

def build_review(run):
    run=Path(run);summary=read_json(run/'summary.json')
    rows=[json.loads(l) for l in (run/'run_log.jsonl').read_text(encoding='utf-8').splitlines()]
    cards=[]
    for r in rows:
        rel=Path(r['output_file']);role=rel.parent.name;stem=rel.stem
        a=run/'previews'/role/(stem+'.jpg');b=run/'enhanced_previews'/role/(stem+'.jpg')
        cards.append(f'''<article data-camera="{r['camera_role']}" data-free="{r.get('free_status','error')}" data-state="{r.get('connection_status','ERROR')}" data-name="{escape(stem.lower())}"><h3>{escape(stem)}</h3><p>{escape(r.get('connection_status','ERROR'))} · 自由段 {escape(r.get('free_status','error'))} · 精修 {escape(r.get('refinement_status',''))} · 新C3观测 {r.get('c3_point_count',0)}</p><div class="pair"><a href="{url(run/'overlays'/role/rel.name,run)}"><img loading="lazy" src="{url(a,run)}" alt="原图结果"></a><a href="{url(run/'enhanced_overlays'/role/rel.name,run)}"><img loading="lazy" src="{url(b,run)}" alt="照明归一化结果"></a></div><p><a href="{url(DATA/rel,run)}">干净原图</a> · <a href="{url(run/'enhanced_roi'/role/rel.name,run)}">增强ROI（外部0）</a> · <a href="{url(A1/'all_001/previews'/role/(stem+'.jpg'),run)}">A1对照</a> · <a href="{url(run/'records'/role/(stem+'.json'),run)}">完整观测/候选</a></p></article>''')
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A2 照明归一化与双边接续</title><style>body{{font:16px/1.7 system-ui,'Microsoft YaHei';background:#111b23;color:#e9eff2;margin:0}}header,main{{padding:20px 3%}}a{{color:#72d1ff}}article{{border:1px solid #34515f;padding:15px;margin:20px 0;background:#182a34}}h3{{overflow-wrap:anywhere}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}img{{width:100%}}nav{{position:sticky;top:0;background:#264353;padding:12px;z-index:2}}select,input{{padding:8px;margin:4px}}.note{{color:#ffcd82}}[hidden]{{display:none!important}}@media(max-width:1000px){{.pair{{grid-template-columns:1fr}}}}</style><header><h1>A2 · 照明归一化与双边接续</h1><p>{summary['mode']} · {summary['variant']} · {summary['completed']}/{summary['planned']}图完成，异常{summary['failed']}。</p><p>左侧为原图，右侧为归一化图，同一检测坐标。绿：自由绳双边；黄：新C3原图复核点；青/橙：有局部双边支持的延续；紫虚线：推断。</p><p class="note">STRAIGHT_SUPPORTED仅表示可见直段延续；CURVE_CANDIDATE仍是图像候选，不确认物理接触。增强不能恢复原图没有的信息。所有未知保留，数量不是准确率。</p><p><a href="summary.json">统计</a> · <a href="selection.csv">全部筛选清单</a></p></header><main><nav><select id="camera"><option value="all">左右相机</option><option value="left">左</option><option value="right">右</option></select><select id="free"><option value="all">全部自由段状态</option><option value="observed">检出</option><option value="direction_only">仅方向</option><option value="unknown">未知</option></select><select id="state"><option value="all">全部接续</option><option value="STRAIGHT_SUPPORTED">可见直行段</option><option value="CURVE_CANDIDATE">弯曲图像候选</option><option value="AMBIGUOUS">歧义</option><option value="UNKNOWN">未知</option></select><input id="search" placeholder="文件名"><span id="count"></span></nav>{''.join(cards)}</main><script>function filter(){{let n=0;for(const a of document.querySelectorAll('article')){{let ok=['camera','free','state'].every(k=>document.getElementById(k).value==='all'||document.getElementById(k).value===a.dataset[k]);ok=ok&&a.dataset.name.includes(document.getElementById('search').value.toLowerCase());a.hidden=!ok;if(ok)n++;}}document.getElementById('count').textContent='显示 '+n+' 张';}}for(const id of ['camera','free','state','search'])document.getElementById(id).addEventListener('input',filter);if(location.hash==='#detected')document.getElementById('free').value='observed';filter();</script></html>'''
    (run/'review.html').write_text(html,encoding='utf-8')
