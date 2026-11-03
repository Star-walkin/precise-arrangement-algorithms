"""Offline V28 report renderer: original pixels, candidates, and separate GT layer."""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from vision_geometry_lab.centerline_path_metrics_v6 import clip_scoring_segments

COLORS = ("#ffb347", "#31c9ff", "#ef81ff", "#ffe16b", "#ff5a73", "#aeb5ff", "#ffffff", "#ff92ba")
BG, FG, MUTED = "#101923", "#f2f6fa", "#abbccc"
STATUS = {"CORRECT": "可见接法匹配", "WRONG": "可见接法偏离", "UNKNOWN": "无法判断",
          "MATCHES_ANNOTATED_RIBBON": "匹配人工可见绳带", "DEPARTS_ANNOTATED_RIBBON": "偏离人工可见绳带",
          "SINGLE_CANDIDATE": "单一路径输出", "MULTI_CANDIDATE": "多候选 / 拒识"}
REASONS = {
    "empty_candidate_pool": "没有可用候选", "excessive_cost": "总代价过高",
    "near_tied_candidates": "前两名近似并列", "perturbation_changes_ranking": "扰动后换路",
    "insufficient_distinguishing_region": "关键分歧区不足", "insufficient_usable_original_evidence": "原图可用证据不足",
    "insufficient_bilateral_connection_support": "双侧边界支持不足",
    "lowest_fixed_cost_within_gates": "固定评分及门槛下最低代价",
}


def _read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def _font(size):
    path = Path("C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _pct(value):
    return "—" if value is None else f"{100*value:.1f}%"


def _number(value):
    return "—" if value is None else f"{value:.3f}"


def _choice_label(choice):
    if choice["rejected"]:
        return "拒识"
    visible = choice["selected_visible_connection_status"]
    geometry = choice["selected_visible_geometry_status"]
    if visible == "UNKNOWN" and geometry == "DEPARTS_ANNOTATED_RIBBON":
        return "分支身份未知；几何已偏离"
    if visible == "UNKNOWN" and geometry == "MATCHES_ANNOTATED_RIBBON":
        return "分支身份未知；几何贴合"
    return STATUS.get(visible, "无法判断")


def _box(points, image_size, padding=70, min_size=200):
    w, h = image_size
    p = np.asarray(points, float).reshape(-1, 2)
    if not len(p):
        return (0, 0, w, h)
    low, high = p.min(axis=0)-padding, p.max(axis=0)+padding
    center = (low+high)/2
    size = np.maximum(high-low, min_size)
    low, high = center-size/2, center+size/2
    return (max(0, int(low[0])), max(0, int(low[1])), min(w, int(np.ceil(high[0]))), min(h, int(np.ceil(high[1]))))


def _panel(source, crop, title, subtitle="", size=(720, 625)):
    canvas = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 13), title, font=_font(23), fill=FG)
    draw.text((16, 48), subtitle, font=_font(16), fill=MUTED)
    view = source.crop(crop)
    scale = min((size[0]-24)/view.width, (size[1]-115)/view.height)
    dimensions = (max(1, round(view.width*scale)), max(1, round(view.height*scale)))
    view = view.resize(dimensions, Image.Resampling.LANCZOS)
    offset = ((size[0]-view.width)//2, 82+(size[1]-115-view.height)//2)
    canvas.paste(view, offset)
    def transform(points):
        return (np.asarray(points, float)-np.asarray(crop[:2]))*scale + np.asarray(offset)
    return canvas, transform


def _line(draw, points, color, width=3, supported=None, unavailable=None, force_dash=False):
    p = np.asarray(points, float)
    if len(p) < 2:
        return
    supports = np.ones(len(p), bool) if supported is None else np.asarray(supported, bool)
    missing = np.zeros(len(p), bool) if unavailable is None else np.asarray(unavailable, bool)
    if len(supports) != len(p) or len(missing) != len(p):
        raise ValueError("Path point/support lengths differ")
    arc = 0.
    for i, (a, b) in enumerate(zip(p[:-1], p[1:])):
        length = float(np.linalg.norm(b-a))
        if length < 1e-7:
            continue
        style = "solid" if supports[i] and supports[i+1] and not force_dash else "dash"
        local_color = color
        if missing[i] or missing[i+1]:
            style, local_color = "dot", "#a2aab3"
        if style == "solid":
            draw.line([tuple(a), tuple(b)], fill=local_color, width=width)
        else:
            period, on = (12., 7.) if style == "dash" else (7., 2.)
            steps = max(1, int(np.ceil(length/2)))
            for j in range(steps):
                lo, hi = j/steps, (j+1)/steps
                if (arc+(lo+hi)*.5*length) % period < on:
                    draw.line([tuple(a+lo*(b-a)), tuple(a+hi*(b-a))], fill=local_color, width=width)
        arc += length


def _candidate(panel, transform, candidate, color, width=3, boundary=False, rejected=False):
    draw = ImageDraw.Draw(panel)
    _line(draw, transform(candidate["points_xy"]), color, width,
          candidate.get("supported"), candidate.get("unavailable"), rejected)
    if boundary:
        for key in ("left_xy", "right_xy"):
            if candidate.get(key):
                # Missing original boundaries remain absent, never bridge None
                # entries or draw guessed coordinates as a measured edge.
                for start, end in _valid_edge_runs(candidate[key]):
                    supported = candidate.get("supported")
                    unavailable = candidate.get("unavailable")
                    _line(draw, transform(candidate[key][start:end]), color, 1,
                          supported[start:end] if supported is not None else None,
                          unavailable[start:end] if unavailable is not None else None, rejected)


def _valid_edge_runs(points):
    """Contiguous existing finite boundary vertices; a missing point is a gap."""
    result, start = [], None
    for index, point in enumerate(list(points)+[None]):
        valid = (point is not None and isinstance(point, (list, tuple, np.ndarray)) and
                 np.asarray(point).shape == (2,) and np.isfinite(np.asarray(point, float)).all())
        if valid and start is None:
            start = index
        elif not valid and start is not None:
            if index-start >= 2:
                result.append((start, index))
            start = None
    return result


def _footer(panel, text):
    draw = ImageDraw.Draw(panel)
    draw.text((16, panel.height-28), text, fill=MUTED, font=_font(15))


def _entry(panel, transform, pool):
    if pool.get("free_seed"):
        draw = ImageDraw.Draw(panel)
        points = transform(pool["free_seed"])
        _line(draw, points, "#69d7ff", 3)
        p = points[-1]
        draw.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5), fill="#ffb347", outline="white", width=1)


def _main_panels(source, structure, pool, scores, evaluation, mode):
    candidates = pool["candidates"]
    crop = structure.get("roi_xyxy")
    if crop is None:
        crop = (0, 0, source.width, source.height)
    crop = tuple(int(v) for v in crop)
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        crop = (0, 0, source.width, source.height)
    panels = []
    panel, tr = _panel(source, crop, "干净原图 · 自动搜索区", "先独立观察，再比较候选；未缩放原图另有链接")
    _footer(panel, f"原图坐标 ROI = {list(crop)}")
    panels.append(panel)
    panel, tr = _panel(source, crop, "局部卷面结构", "细线为双边；实线=原图支持，虚线=几何推断")
    for i, row in enumerate(structure.get("structures", [])):
        _candidate(panel, tr, row, COLORS[i % len(COLORS)], width=2, boundary=True)
    _entry(panel, tr, pool)
    _footer(panel, "局部结构仍可能是旧绳、暗沟或股纹，不是人工身份确认")
    panels.append(panel)
    panel, tr = _panel(source, crop, "冻结的全部候选", f"{len(candidates)} 条；O=旧圈接续，X=可跨旧结构的新段")
    for i, row in enumerate(candidates):
        _candidate(panel, tr, row, COLORS[i % len(COLORS)], width=3)
    _entry(panel, tr, pool)
    _footer(panel, " / ".join(c["id"] for c in candidates) or "无候选；保留在总分母")
    panels.append(panel)
    by_id = {c["id"]: (i, c) for i, c in enumerate(candidates)}
    for group in ("G0", "G1", "G2"):
        ranking = scores[group]
        selected, best = ranking.get("selected_id"), ranking.get("best_id")
        choice = evaluation["choices"][group]
        title = f"{group} · " + (f"选择 {selected}" if selected else "拒识 / 无法判断")
        outcome = _choice_label(choice)
        panel, tr = _panel(source, crop, title, f"{mode} | {outcome} | 分差 {_number(ranking.get('margin'))} | 稳定 {_pct(ranking.get('stability'))}")
        chosen = selected or best
        if chosen in by_id:
            i, row = by_id[chosen]
            _candidate(panel, tr, row, COLORS[i % len(COLORS)], width=4, boundary=bool(selected), rejected=not selected)
        _entry(panel, tr, pool)
        _footer(panel, "拒识时仅用虚线示意第一名，未输出确定接法" if not selected else "自动双边支持不等于人工确认；灰色点线为不可用区")
        panels.append(panel)
    canvas = Image.new("RGB", (3*720, 2*625), BG)
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % 3)*720, (i//3)*625))
    return canvas


def _evaluation_panels(source, pool, scores, evaluation):
    reference = evaluation["evaluation_only_reference"]
    segments = evaluation.get("visible_branch_segments", [])
    points = [p for line in segments for p in line]
    if not points:
        points = reference["centerline"]
    crop = _box(points, source.size, padding=65, min_size=320)
    panels = []
    panel, tr = _panel(source, crop, "关键弯段 · 干净原图", "该放大区由人工参考事后确定，未用于自动搜索", size=(800, 650))
    _footer(panel, f"仅评价坐标 ROI = {list(crop)}")
    panels.append(panel)
    panel, tr = _panel(source, crop, "关键弯段 · 候选对照", "只画已冻结的候选，不据真值修正路径", size=(800, 650))
    for i, row in enumerate(pool["candidates"]):
        _candidate(panel, tr, row, COLORS[i % len(COLORS)], width=3, boundary=False)
    _footer(panel, "实线支持 / 虚线推断 / 灰点线不可用；无候选也如实展示")
    panels.append(panel)
    panel, tr = _panel(source, crop, "评价层 · 人工可信参考", "绿色仅为人工参考；所有算法均未读此层", size=(800, 650))
    draw = ImageDraw.Draw(panel)
    for key in ("edge_flange", "edge_surface"):
        for line in clip_scoring_segments([reference[key]], ignore_polygons=reference.get("ignore_polygons", [])):
            _line(draw, tr(line), "#5de693", 2)
    for segment in evaluation.get("trusted_bend_segments", []):
        _line(draw, tr(segment), "#39ff75", 4)
    departure = evaluation.get("GT_departure")
    if departure:
        p = tr([departure["point_xy"]])[0]
        draw.ellipse((p[0]-6,p[1]-6,p[0]+6,p[1]+6), outline="#ffffff", width=2)
    # Do not display annotated-but-unknown tails or use them as negative space.
    for polygon in reference.get("ignore_polygons", []):
        poly = tr(polygon)
        draw.line([tuple(p) for p in np.vstack([poly, poly[:1]])], fill="#808080", width=2)
    _footer(panel, "白圈=历史弯段诊断点；不是物理切点、接触或新层起点")
    panels.append(panel)
    canvas = Image.new("RGB", (2400, 650), BG)
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i*800, 0))
    return canvas


def _score_table(pool, scores, evaluation):
    costs = {g: {r["id"]: r for r in scores[g].get("ordered", [])} for g in ("G0", "G1", "G2")}
    ref = {r["id"]: r for r in evaluation["candidate_scores"]}
    rows = []
    for i, c in enumerate(pool["candidates"]):
        ev = ref[c["id"]]
        g2 = costs["G2"].get(c["id"], {}).get("components", {})
        cells = [f'<b style="color:{COLORS[i % len(COLORS)]}">{html.escape(c["id"])}</b>',
                 "可跨旧结构" if c.get("kind") == "new_segment_may_cross_old_rows" else "旧结构接续"]
        cells.extend(_number(costs[g].get(c["id"], {}).get("cost")) for g in ("G0", "G1", "G2"))
        cells += [_pct(g2.get("available_fraction")), _pct(g2.get("bilateral_fraction")),
                  _pct(ev["bend_geometry_5px"]["covered_fraction"]),
                  STATUS.get(ev["visible_ribbon_geometry_status"], "无法判断")]
        rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in cells) + "</tr>")
    return '<table><thead><tr>' + ''.join(f'<th>{t}</th>' for t in (
        "候选", "类型", "G0代价↓", "G1代价↓", "G2代价↓", "分歧区可用", "分歧区双边", "弯段5px覆盖", "可见绳带几何")) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'


def render_run(output_dir):
    """Create once; refuse overwrite. Requires runner-saved real artifacts."""
    output_dir = Path(output_dir)
    index_path = output_dir / "index.html"
    if index_path.exists():
        raise FileExistsError(index_path)
    manifest = _read(output_dir / "input_manifest.json")
    if isinstance(manifest, dict):
        manifest = manifest.get("inputs", manifest.get("samples", manifest.get("images")))
    if not isinstance(manifest, list):
        raise ValueError("Expected image list in input_manifest.json")
    records = _read(output_dir / "records.json")
    if isinstance(records, dict):
        records = records.get("records", records.get("rows"))
    evaluation = _read(output_dir / "evaluation.json")
    references = {(r["sample_id"], r["mode"]): r for r in evaluation["rows"]}
    inputs = {r["sample_id"]: r for r in manifest}
    assets = output_dir / "report_assets"
    assets.mkdir(exist_ok=False)
    images = {}
    for sid, row in inputs.items():
        target = assets / (sid + "__clean.png")
        shutil.copyfile(row["image_path"], target)
        images[sid] = target
    summary_html = []
    for mode, summary in evaluation["summary"].items():
        rows = []
        for group, row in summary["scores"].items():
            geometry_wrong = sum(r["mode"] == mode and not r["choices"][group]["rejected"] and
                                 r["choices"][group]["selected_visible_geometry_status"] == "DEPARTS_ANNOTATED_RIBBON"
                                 for r in evaluation["rows"])
            rows.append(f'<tr><td>{mode} / {group}</td><td>{row["correct"]}</td><td>{row["wrong"]}</td><td>{geometry_wrong}</td><td>{row["rejected"]}</td><td>{row["unevaluable_output"]}</td><td>{row["output_images"]}/8</td><td>{_pct(row["bend_coverage_5px"])}</td></tr>')
        summary_html.append(f'<p><b>{mode}</b>：总图数 {summary["total_images"]}；自由段入口失败 {summary["automatic_entry_failures"]}；可见接法可评价 {summary["visible_connection_evaluable_images"]}；候选包含可信弯段几何 {summary["candidate_pool_contains_trusted_bend_geometry_images"]}。整绳身份可评价仍为 0。</p>')
        summary_html.append('<table><tr><th>模式 / 评分</th><th>分支选对</th><th>分支选错</th><th>输出几何已偏离</th><th>拒识</th><th>输出但分支不可评价</th><th>有效输出</th><th>关键弯段覆盖</th></tr>'+''.join(rows)+'</table>')
        summary_html.append('<p>“几何已偏离”表示输出落到人工可信双边绳带之外，即使严格分支身份仍不可评价，也明确保留该错误；这一列是几何诊断，不与其余状态列相加。</p>')
        repairs = summary.get("G1_to_G2_repairs", [])
        degradation = summary.get("G1_to_G2_degradations", [])
        summary_html.append('<p>G1→G2 修复：'+html.escape('、'.join(repairs) or '未证明')+'；退化：'+html.escape('、'.join(degradation) or '无可评价退化')+'。</p>')
    cards, rows_html = [], []
    for record in records:
        sid, mode = record["sample_id"], record["mode"]
        ev = references[sid, mode]
        pool, scores, structure = (_read(record[key]) for key in ("pool_path", "scores_path", "structure_path"))
        filename = sid + "__" + mode
        with Image.open(images[sid]) as original:
            source = original.convert("RGB")
            main = _main_panels(source, structure, pool, scores, ev, mode)
            main.save(assets / (filename + "__comparison.png"))
            _evaluation_panels(source, pool, scores, ev).save(assets / (filename + "__evaluation.png"))
        link = "report_assets/"+filename+"__comparison.png"
        eval_link = "report_assets/"+filename+"__evaluation.png"
        clean_link = "report_assets/"+sid+"__clean.png"
        labels = []
        reason_lines = []
        for group in ("G0", "G1", "G2"):
            choice, score = ev["choices"][group], scores[group]
            label = _choice_label(choice)
            labels.append(label + (f'（{choice["selected_id"]}）' if choice["selected_id"] else ''))
            reasons = score.get("reason", [])
            if isinstance(reasons, str):
                reasons = [reasons]
            reason_lines.append(f'<li><b>{group}</b>：{html.escape(label)}；'+html.escape('；'.join(REASONS.get(r, r) for r in reasons))+f'；分差 {_number(score.get("margin"))}，扰动稳定 {_pct(score.get("stability"))}。</li>')
        rows_html.append(f'<tr><td><a href="#{filename}">{html.escape(sid)}</a></td><td>{mode}</td>'+''.join('<td>'+html.escape(t)+'</td>' for t in labels)+f'<td>{_pct(ev["best_pool_bend_coverage_5px"])}</td></tr>')
        entry_available = ev["automatic_entry_success"]
        cards.append(f'''<section id="{filename}"><h2>{mode} · {html.escape(sid)}</h2>
<p>自由段入口：{"可用" if entry_available else "失败，保留在总分母"}；局部结构 {len(structure.get("structures", []))} 条；冻结候选 {len(pool["candidates"])} 条；候选上限截断：{"是" if pool.get("truncated") else "否"}。</p>
<p><a href="{clean_link}" target="_blank">打开未画线的全分辨率原图</a> · <a href="{link}" target="_blank">放大六面板对照</a></p>
<a href="{link}" target="_blank"><img loading="lazy" src="{link}" alt="{mode} {sid} 原图、结构、全部候选及三种评分"/></a>
<ul>{''.join(reason_lines)}</ul>{_score_table(pool,scores,ev)}
<details><summary>展开事后评价层：关键弯段原图 / 候选 / 绿色人工可信参考</summary><p>只用于评价和人工核对。绿色参考未进入 AUTO 输入、ROI、候选、评分。白圈不是物理切点。</p><a href="{eval_link}" target="_blank"><img loading="lazy" src="{eval_link}" alt="独立人工评价层"/></a></details>
</section>''')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V28 局部卷面与多候选接续</title>
<style>body{background:#101923;color:#e9f0f6;font:16px/1.6 "Microsoft YaHei",sans-serif;margin:0 auto;padding:28px;max-width:1500px}h1{font-size:30px}h2{font-size:22px;word-break:break-all}a{color:#77cfff}section{margin:32px 0;padding:22px;background:#192532;border:1px solid #344555;border-radius:12px}p{max-width:1200px}table{width:100%;border-collapse:collapse;font-size:14px;margin:16px 0;display:table}th,td{padding:9px;border:1px solid #344555;text-align:left}th{background:#223448}img{width:100%;height:auto;border:1px solid #3a4d60;border-radius:8px}summary{cursor:pointer;color:#81d9aa;font-weight:bold;padding:12px}details{background:#111b25;border:1px solid #375145;border-radius:7px;margin-top:20px}.notice{background:#283242;border-left:5px solid #f5c767;padding:16px}li{margin:5px 0}@media(max-width:800px){body{padding:10px}section{padding:12px}table{font-size:12px;display:block;overflow:auto}}</style>
<h1>V28：哪些接法匹配，哪些偏离，哪些仍无法判断</h1>
<p class="notice">“选对 / 选错”仅指独立人工可信双边绳带上的可见接法。整绳材料身份、物理接触、爬层、t1/t2 均未确认。几何补全只产生候选；实线图像支持也不是人工身份确认。</p>'''
    page += ''.join(summary_html)
    page += '<h2>逐图结果（全部固定开发图）</h2><table><tr><th>图像</th><th>输入</th><th>G0</th><th>G1</th><th>G2</th><th>池中最佳弯段覆盖</th></tr>'+''.join(rows_html)+'</table>'
    page += '<p>AUTO 是主要结果。CONDITIONAL 仅使用已有独立自由段提示，单独评价；历史开发 8 图不是盲测，不是 8 次独立换层事件。邻近帧不能增加独立工况数。</p>'
    page += ''.join(cards)
    page += '''<section><h2>怎样阅读</h2><p>O 表示与局部旧结构接续，X 表示允许跨过旧结构的新段假设。两者都是二维候选，不代表真实高度关系。彩色实线是算法测得的原图双侧支持，彩色虚线是推断，灰点线是不可用区。拒识面板的最佳路径统一用虚线示意，不能当成选中的接法。</p><p>G0 只比较简易距离和方向，G1 加入宽度、曲率与局部排列，G2 再加入原图双边、法向剖面和中断证据。三种评分读取同一个已保存的候选池。代价越小越优；排名稳定性、分差均不是身份正确概率。</p><p>几何覆盖沿用可信弯段 5 px 口径，不能代替身份。可见接法评价另检查人工双侧绳带、关键分歧截面和竞争路径；缺少可靠分支证据或候选未到弯段时为未知。全体拒识不能宣布算法胜出；表中同时列有效输出率。</p><p>候选池未包含可信弯段时，应先改进局部结构与候选生成。只有 G2 把同图 G1 的可评价错误变为正确，才能称原图证据修复；单纯拒识或画线更完整均不算修复。</p></section></html>'''
    index_path.write_text(page, encoding="utf-8")
    return dict(index_path=str(index_path), cases=len(records), clean_images=len(inputs),
                comparison_images=len(records), evaluation_images=len(records))
