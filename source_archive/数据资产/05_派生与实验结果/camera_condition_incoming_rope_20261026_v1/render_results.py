"""绘图只表达已有检测/推断，不参与算法评分；原始图像保持不变。"""
from pathlib import Path
from html import escape
from urllib.parse import quote
import argparse
import csv
import json
import os
import cv2
import numpy as np
from experiment_io import DATA, HERE, read_json, read_image, write_image

def points(value):
    if value is None: return np.empty((0, 2), float)
    a = np.asarray(value, dtype=float).reshape(-1, 2)
    return a[np.isfinite(a).all(axis=1)]

def poly(image, value, color, thick=2, dashed=False):
    p = points(value)
    if len(p) < 2: return
    p = np.rint(p).astype(np.int32)
    if not dashed:
        cv2.polylines(image, [p], False, color, thick, cv2.LINE_AA)
    else:
        for i in range(len(p) - 1):
            if (i // 3) % 2 == 0:
                cv2.line(image, tuple(p[i]), tuple(p[i+1]), color, thick, cv2.LINE_AA)

def crop_preview(image, surface, free, width=1450):
    yy, xx = np.nonzero(surface)
    candidates = [np.column_stack((xx, yy))] if len(xx) else []
    q = points(free.get('observed_segment_xy'))
    if len(q): candidates.append(q)
    if candidates:
        a = np.vstack(candidates)
        x0, y0 = np.maximum(np.floor(a.min(axis=0)).astype(int) - 110, 0)
        x1, y1 = np.minimum(np.ceil(a.max(axis=0)).astype(int) + 111,
                           [image.shape[1], image.shape[0]])
    else: x0, y0, x1, y1 = 0, 0, image.shape[1], image.shape[0]
    out = image[y0:y1, x0:x1]
    scale = min(1., width / max(1, out.shape[1]), 1200. / max(1, out.shape[0]))
    if scale < 1: out = cv2.resize(out, (round(out.shape[1]*scale), round(out.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    return out, [int(x0), int(y0), int(x1), int(y1)]

def render(image, surface, c3_fit, free, connection, title='IR-A1'):
    out = image.copy()
    contours = cv2.findContours(surface, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(out, contours, -1, (90, 160, 90), 2)
    # 卷面基准画得较淡，避免盖住真正需要观察的活动绳。
    layer = out.copy()
    for curve in c3_fit.get('curves', []):
        poly(layer, curve.get('sampled_points_xy'), (160, 150, 120), 1)
    out = cv2.addWeighted(out, .72, layer, .28, 0)
    poly(out, free.get('observed_segment_xy'), (0, 220, 255), 2)
    for k in ['boundary_left_xy', 'boundary_right_xy']:
        poly(out, free.get(k), (0, 255, 80), 3)
    for xy in points(free.get('observed_points_xy')):
        cv2.circle(out, tuple(np.rint(xy).astype(int)), 3, (0, 255, 255), -1)
    selected = connection.get('selected_candidate_id')
    candidates = connection.get('candidates', [])
    ranked = sorted(candidates, key=lambda c: c.get('score_g2',{}).get('cost',1e9))
    displayed = [c for c in candidates if c.get('candidate_id') == selected] if selected else ranked[:3]
    for i, c in enumerate(displayed):
        path = points(c.get('path_xy'))
        color = (255, 40, 220) if selected else [(255, 110, 230), (210, 160, 100), (180, 120, 255)][i % 3]
        states = c.get('point_states', [])
        if len(states) == len(path):
            for j in range(len(path) - 1):
                if 'unavailable' in (states[j], states[j+1]):
                    if (j//3)%2 == 0: poly(out, path[j:j+2], (150,150,150), 1)
                elif states[j] == states[j+1] == 'observed':
                    poly(out, path[j:j+2], (0, 190, 255) if selected else color, 3)
                elif (j//3)%2 == 0:
                    poly(out, path[j:j+2], color, 3 if selected else 2)
        else: poly(out,path,color,2,dashed=True)
        # 只有候选给出区间才标圈；圈不是已确认物理接触点。
        interval = points(c.get('transition_interval_xy'))
        if len(interval):
            for p in [interval[0], interval[-1]]:
                cv2.circle(out, tuple(np.rint(p).astype(int)), 7, (20, 80, 255), 2)
        center = points([c['transition_center_xy']]) if c.get('transition_center_xy') is not None else []
        for p in center:
            cv2.drawMarker(out,tuple(np.rint(p).astype(int)),(20,80,255),cv2.MARKER_CROSS,18,2)
        if len(path):
            p = tuple(np.rint(path[-1]).astype(int))
            cv2.putText(out, str(c.get('candidate_id', i)), p, cv2.FONT_HERSHEY_SIMPLEX, .75, color, 2, cv2.LINE_AA)
    preview, box = crop_preview(out, surface, free)
    banner = np.zeros((76, preview.shape[1], 3), np.uint8)
    line1 = title + ' | free=' + str(free.get('status')) + ' | connection=' + str(connection.get('status'))
    width = free.get('width_px'); angle = free.get('angle_deg')
    line2 = f'Optical width: {width:.2f}px' if width is not None else 'Optical width: unavailable'
    line2 += f' | angle: {angle:.2f}deg' if angle is not None else ''
    line2 += ' | green=free edges; orange=image support; magenta=inference; red=transition candidate'
    for j, t in enumerate([line1, line2]):
        cv2.putText(banner, t, (10, 25+32*j), cv2.FONT_HERSHEY_SIMPLEX, .45, (230, 230, 230), 1, cv2.LINE_AA)
    return out, np.vstack([banner, preview]), box

def url(path, root):
    return quote(Path(os.path.relpath(path, root)).as_posix(), safe='/._-')

def build_review(run):
    run = Path(run)
    summary = read_json(run / 'summary.json')
    rows = [json.loads(s) for s in (run/'run_log.jsonl').read_text(encoding='utf-8').splitlines()]
    cards = []
    names = {'observed': '检测到自由直段', 'direction_only': '直段候选／宽度不稳', 'unknown': '未可靠检出',
             'SINGLE_IMAGE_HYPOTHESIS': '单一图像接法候选', 'MULTI_CANDIDATE': '多种接法', 'UNKNOWN': '接续未知'}
    for row in rows:
        rel = Path(row['output_file']); camera = row['camera_role']; stem = rel.stem
        record = run/'records'/rel.parent.name/(stem+'.json')
        preview = run/'previews'/rel.parent.name/(stem+'.jpg')
        overlay = run/'overlays'/rel.parent.name/rel.name
        fs = row.get('free_status', 'error'); cs = row.get('connection_status', 'ERROR')
        status = names.get(fs, fs) + ' · ' + names.get(cs, cs)
        width = row.get('width_px'); angle = row.get('angle_deg')
        detail = ('光学宽度 %.2f px' % width) if width is not None else '宽度不可用'
        if angle is not None: detail += ' · 方向 %.2f°' % angle
        image_html = f'<img loading="lazy" src="{url(preview,run)}" alt="{escape(stem)}">' if preview.exists() else '<p>处理失败，见记录</p>'
        cards.append(f'''<article data-camera="{camera}" data-free="{fs}" data-connection="{cs}" data-qa="{int(row.get('qa_selected',False))}" data-search="{escape(stem.lower())}"><h3>{escape(stem)}</h3><p class="state">{escape(status)} · {escape(detail)}</p><a href="{url(overlay,run)}">{image_html}</a><p><a href="{url(DATA/rel,run)}">未画线原图</a> · <a href="{url(overlay,run)}">4096×3000结果</a> · <a href="{url(run/'baseline_previews'/rel.parent.name/(stem+'.jpg'),run)}">直线延长对照</a> · <a href="{url(record,run)}">候选与测量记录</a></p></article>''')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>活动绳 A1 · 直段与转弯候选</title><style>body{{font:16px/1.7 system-ui,'Microsoft YaHei';background:#101820;color:#e1e9ee;margin:0}}header{{padding:22px 4%;background:#182631}}main{{max-width:1600px;margin:auto;padding:20px}}a{{color:#65c9f4}}h1{{margin:0}}.note{{color:#ffcf80}}.toolbar{{position:sticky;top:0;background:#20323e;padding:12px;z-index:1}}select,input{{padding:7px;margin:4px;border-radius:4px}}article{{background:#172731;border:1px solid #355063;margin:20px 0;padding:16px;border-radius:8px}}img{{max-width:100%;display:block;margin:auto}}h3{{overflow-wrap:anywhere}}.state{{color:#9fe2ce}}[hidden]{{display:none!important}}</style><header><h1>活动绳直段与转弯候选 A1</h1><p>共{summary['planned']}张：已处理{summary['completed']}，异常{summary['failed']}。所有图片保留在分母中，筛选结果是自动建议。</p><p>绿线：自由段光学双边；黄点：剖面观测；橙线：局部图像支持；紫色虚线：几何推断；红色圈／线段：直转弯候选区间。</p><p class="note">宽度是图像像素；未检出不等于没有活动绳。平滑接续、双边响应或直线转弯均不能独立证明同根身份和三维物理接触。</p><p><a href="summary.json">运行汇总</a> · <a href="selection.csv">400图筛选清单</a> · <a href="run_context.json">参数与源码哈希</a></p></header><main><div class="toolbar"><select id="camera"><option value="all">左右相机</option><option value="left">左相机</option><option value="right">右相机</option></select><select id="free"><option value="all">全部筛选结果</option><option value="observed">检测到自由直段</option><option value="direction_only">直段候选／宽度不稳</option><option value="unknown">未可靠检出</option></select><select id="connection"><option value="all">全部接续状态</option><option value="SINGLE_IMAGE_HYPOTHESIS">单一图像候选</option><option value="MULTI_CANDIDATE">多种接法</option><option value="UNKNOWN">接续未知</option></select><select id="qa"><option value="all">全部图片</option><option value="1">固定QA38</option></select><input id="search" placeholder="文件名筛选"><span id="count"></span></div>{''.join(cards)}</main><script>const keys=['camera','free','connection','qa','search'];function update(){{let n=0;for(const a of document.querySelectorAll('article')){{let show=true;for(const k of keys){{const v=document.getElementById(k).value.toLowerCase();if(k==='search')show=show&&a.dataset.search.includes(v);else if(v!=='all')show=show&&a.dataset[k].toLowerCase()===v;}}a.hidden=!show;if(show)n++;}}document.getElementById('count').textContent='显示 '+n+' 张';}}for(const k of keys)document.getElementById(k).addEventListener('input',update);if(location.hash==='#detected')document.getElementById('free').value='observed';update();</script></html>'''
    (run/'review.html').write_text(page, encoding='utf-8')
    return len(cards)

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    print('Review cards:', build_review(ap.parse_args().run))
