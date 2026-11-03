"""将已有 v3 运行结果整理为一个可离线打开的中文审阅页。

只读取固定 manifest、运行统计和保存的图片/记录，不导入算法、不重算分割。
执行 `python build_review_v3.py --run <v3运行目录>`，生成或重建该目录的
`review.html`。页面引用已保存的原图、叠加图和掩码，不复制原图。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
DATASET = ASSETS / "camera_condition_images_20261009_v2"
MANIFEST = DATASET / "manifest.csv"
QA_PLAN = ASSETS / "camera_condition_bright_rope_boundaries_20261009_v1" / "qa_plan.csv"
PROTOCOL = HERE / "protocol.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def href(run: Path, path: Path) -> str:
    """从 review.html 所在目录生成可点击的相对链接。"""
    rel = Path(os.path.relpath(path, run)).as_posix()
    return html.escape(quote(rel, safe="/._-"), quote=True)


def checked_link(run: Path, path: Path, label: str) -> str:
    if not path.is_file():
        return f"<span class='missing'>{esc(label)}：缺失</span>"
    return f"<a href='{href(run, path)}' target='_blank' rel='noopener'>{esc(label)}</a>"


def paths_for(run: Path, row: dict) -> dict[str, Path]:
    rel = Path(row["output_file"])
    role, name = rel.parent.name, rel.name
    stem = name.replace(".png", "")
    return {
        "source": DATASET / rel,
        "overlay": run / "overlays" / role / f"{stem}.jpg",
        "comparison": run / "qa_compare" / role / f"{stem}.jpg",
        "effective_rgb": run / "roi" / "effective_masked" / role / name,
        "reference_roi": run / "roi" / "reference_masks" / role / name,
        "effective_roi": run / "roi" / "effective_masks" / role / name,
        "observed": run / "boundaries" / "observed" / role / name,
        "inferred": run / "boundaries" / "inferred" / role / name,
        "supported_rope": run / "boundaries" / "incoming_rope_occlusion" / role / name,
        "unknown": run / "boundaries" / "unknown_exclusion" / role / name,
        "inferred_under_occlusion": run / "boundaries" / "inferred_under_occlusion" / role / name,
        "record": run / "boundaries" / "records" / role / f"{stem}.json",
    }


def load_plan(run: Path) -> tuple[list[dict], set[str], dict]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if digest(MANIFEST) != protocol["input_manifest_sha256"]:
        raise ValueError("固定 400 图 manifest 哈希与 v3 协议不一致")
    if digest(QA_PLAN) != protocol["fixed_qa_plan_sha256"]:
        raise ValueError("固定 38 图 QA 清单哈希与 v3 协议不一致")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    with QA_PLAN.open("r", encoding="utf-8-sig", newline="") as stream:
        qa_rows = list(csv.DictReader(stream))
    qa = {row["output_file"] for row in qa_rows}
    if len(all_rows) != 400 or len(qa_rows) != 38 or len(qa) != 38:
        raise ValueError("固定图片数量不符合 400/38 协议")
    if summary.get("mode") == "all":
        rows = all_rows
    elif summary.get("mode") == "pilot":
        rows = [row for row in all_rows if row["output_file"] in qa]
    else:
        raise ValueError("运行 summary 不是 pilot/all 模式")
    if len(rows) != summary.get("planned"):
        raise ValueError("运行 planned 数量与固定输入清单不一致")
    return rows, qa, summary


def card(run: Path, row: dict, qa: set[str], log: dict[str, dict]) -> str:
    files = paths_for(run, row)
    role = row["camera_role"]
    is_qa = row["output_file"] in qa
    status = log.get(row["output_file"], {}).get("status", "未记录")
    record = None
    if files["record"].is_file():
        try:
            record = json.loads(files["record"].read_text(encoding="utf-8"))["record"]
        except (ValueError, KeyError, TypeError):
            status = "记录不可读"
    observed_points = record.get("v3_observed_points", 0) if record else 0
    inferred_pixels = record.get("inferred_pixels", 0) if record else 0
    rope_pixels = record.get("rope_pixels", 0) if record else 0
    unknown_pixels = record.get("unknown_pixels", 0) if record else 0
    under_pixels = record.get("inferred_under_occlusion_pixels", 0) if record else 0
    has_purple = bool(inferred_pixels)
    has_occlusion = bool(rope_pixels or unknown_pixels)
    links = [
        ("原图", "source"), ("有效 ROI 内原色图", "effective_rgb"),
        ("原亮带 ROI mask", "reference_roi"), ("有效 ROI mask", "effective_roi"),
        ("黄色观测 mask", "observed"), ("紫色推断 mask", "inferred"),
        ("橙色活动绳 mask", "supported_rope"), ("蓝色未知遮挡 mask", "unknown"),
        ("紫色∩橙色 mask", "inferred_under_occlusion"), ("逐线记录 JSON", "record"),
    ]
    link_html = " · ".join(checked_link(run, files[key], label) for label, key in links)
    comparison = ("<p class='compare'>" + checked_link(run, files["comparison"], "打开固定 QA 并排对照") + "</p>"
                  if is_qa else "")
    overlay = files["overlay"]
    if overlay.is_file():
        thumb = (f"<a class='thumb' href='{href(run, overlay)}' target='_blank' rel='noopener'>"
                 f"<img loading='lazy' decoding='async' src='{href(run, overlay)}' "
                 f"alt='{esc(Path(row['output_file']).name)} 的 v3 叠加图'></a>")
    else:
        thumb = "<div class='thumb absent'>叠加图缺失，检查运行日志</div>"
    name = Path(row["output_file"]).name
    layer = row.get("winding_layer", "未标明")
    diameter = row.get("spool_outer_diameter_mm", "未标明")
    return (
        f"<article class='card' data-role='{esc(role)}' data-qa='{int(is_qa)}' "
        f"data-purple='{int(has_purple)}' data-occlusion='{int(has_occlusion)}' "
        f"data-search='{esc(name + ' ' + row['source_group'] + ' ' + layer)}'>"
        f"{thumb}<div class='body'><h2>{esc(name)}</h2>"
        f"<p class='tags'><span>{'左相机' if role == 'left' else '右相机'}</span>"
        f"{'<span>固定 QA</span>' if is_qa else ''}"
        f"{'<span>有紫线</span>' if has_purple else ''}"
        f"{'<span>有遮挡</span>' if has_occlusion else ''}</p>"
        f"<p class='meta'>来源组：{esc(row['source_group'])}<br>层厚记录：{esc(layer)}；"
        f"工字轮外径记录：{esc(diameter)} mm<br>运行状态：{esc(status)}</p>"
        f"<p class='counts'>黄点 {int(observed_points):,}；紫色 {int(inferred_pixels):,} px；"
        f"橙色 {int(rope_pixels):,} px；蓝色 {int(unknown_pixels):,} px；"
        f"遮挡下紫色 {int(under_pixels):,} px</p>"
        f"{comparison}<details><summary>查看原图、ROI、各类 mask 与记录</summary>"
        f"<div class='filelinks'>{link_html}</div></details></div></article>"
    )


def render_page(run: Path, rows: list[dict], qa: set[str], summary: dict) -> str:
    log: dict[str, dict] = {}
    log_path = run / "run_log.jsonl"
    if log_path.is_file():
        with log_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if "output_file" in record:
                    log[record["output_file"]] = record
    cards = "\n".join(card(run, row, qa, log) for row in rows)
    mode_label = "固定 400 图全量" if summary["mode"] == "all" else "固定 38 图试跑"
    stats = (
        f"<div class='stats'>"
        f"<div><strong>{int(summary.get('planned', 0)):,}</strong><span>固定输入</span></div>"
        f"<div><strong>{int(summary.get('completed', 0)):,}</strong><span>完成</span></div>"
        f"<div><strong>{int(summary.get('failed', 0)):,}</strong><span>失败</span></div>"
        f"<div><strong>{int(summary.get('v2_observed_points', 0)):,}</strong><span>v2 黄点数</span></div>"
        f"<div><strong>{int(summary.get('v3_observed_points', 0)):,}</strong><span>v3 黄点数</span></div>"
        f"<div><strong>{int(summary.get('inferred_pixels', 0)):,}</strong><span>紫色像素</span></div>"
        f"<div><strong>{int(summary.get('inferred_under_occlusion_pixels', 0)):,}</strong>"
        f"<span>橙色遮挡下紫色像素</span></div></div>"
    )
    return f"""<!doctype html>
<html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>卷面钢丝绳视觉边界 v3 · 离线审阅</title>
<style>
:root{{color-scheme:dark;--bg:#11171d;--surface:#1e2a34;--line:#365061;--text:#e8f0f5;--muted:#a9bdca;--link:#8ccfff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,'Microsoft YaHei',sans-serif}}
header{{padding:24px max(20px,4vw) 18px;border-bottom:1px solid var(--line)}}h1{{margin:0 0 8px;font-size:27px}}
p{{margin:8px 0}}.intro{{max-width:1050px;color:var(--muted)}}a{{color:var(--link)}}
.legend{{display:flex;flex-wrap:wrap;gap:9px;margin:16px 0}}.legend span,.tags span{{padding:3px 9px;border-radius:999px;background:#2c3d4a}}
.legend b{{font-size:17px}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:9px;max-width:1150px;margin-top:14px}}
.stats>div{{background:#20303b;border:1px solid var(--line);border-radius:9px;padding:9px 12px}}
.stats strong,.stats span{{display:block}}.stats strong{{font-size:23px}}.stats span{{color:var(--muted);font-size:12px}}
.controls{{position:sticky;top:0;z-index:3;background:#17232dee;border-bottom:1px solid var(--line);padding:12px max(20px,4vw);display:flex;flex-wrap:wrap;gap:12px 20px;align-items:center;backdrop-filter:blur(8px)}}
select,input[type=search]{{background:#253744;color:var(--text);border:1px solid #537081;border-radius:6px;padding:7px}}
label{{white-space:nowrap}}input[type=checkbox]{{accent-color:#f5ce3f}}#visibleCount{{margin-left:auto;color:var(--muted)}}
main{{padding:20px max(20px,4vw) 50px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:15px}}
.card{{min-width:0;background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
.card[hidden]{{display:none}}.thumb{{display:block;height:235px;background:#071015}}.thumb img{{width:100%;height:100%;object-fit:contain}}
.thumb.absent{{display:flex;align-items:center;justify-content:center;color:#f0aaa0}}
.body{{padding:12px}}h2{{font-size:14px;line-height:1.35;overflow-wrap:anywhere;margin:0}}
.tags{{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}}.tags span{{font-size:11px;color:#cfdeea}}
.meta,.counts{{color:var(--muted);font-size:12px}}.compare{{font-size:13px;margin:9px 0}}
summary{{cursor:pointer;color:var(--link);font-size:13px}}.filelinks{{margin-top:7px;font-size:12px;line-height:2}}
.missing{{color:#f0aaa0}}footer{{padding:25px max(20px,4vw);color:var(--muted);border-top:1px solid var(--line)}}
</style></head><body>
<header><h1>卷面钢丝绳视觉边界 v3 · 离线审阅</h1>
<p class='intro'>{esc(mode_label)}，左右相机原朝向。点击缩略图看叠加大图；固定 QA 图片可打开原图、v2、v3 并排对照。页面直接链接已保存文件，可离线使用。</p>
<div class='legend'><span><b style='color:#ffe354'>黄</b> 原图可见暗沟槽候选</span>
<span><b style='color:#d879f8'>紫</b> 曲线几何推断</span>
<span><b style='color:#ff9e45'>橙</b> 历史检测和原图剖面支持的活动绳遮挡候选</span>
<span><b style='color:#69b5ff'>蓝</b> 语义候选未充分验证，需排除</span>
<span><b style='color:#55e36f'>绿</b> 仅为显示外包络；精确 ROI 请打开 mask</span></div>
<p class='intro'>黄点数量、与 v2 的邻近比例和紫色像素数只描述输出覆盖，<strong>不是准确率</strong>。紫线及遮挡下紫线不是直接看到的边界，也不证明同根材料或换层事件。暗带、曝光饱和和不可信遮挡应允许留白。</p>
{stats}</header>
<div class='controls'><label>相机 <select id='role'><option value='all'>左右全部</option><option value='left'>左相机</option><option value='right'>右相机</option></select></label>
<label><input type='checkbox' id='qa'> 仅固定 QA 38 图</label>
<label><input type='checkbox' id='purple'> 仅有紫色推断</label>
<label><input type='checkbox' id='occlusion'> 仅有橙/蓝遮挡</label>
<label>名称/来源组 <input type='search' id='search' placeholder='输入文件名或来源组'></label>
<strong id='visibleCount'></strong></div>
<main><div class='grid' id='grid'>{cards}</div></main>
<footer>来源：固定开放开发图片集与该 v3 运行目录。原图、旧 ROI 和旧输出未在此页生成器中修改。图集用于人工检查，未进行逐根真值评分。</footer>
<script>
const cards=Array.from(document.querySelectorAll('.card'));
const role=document.getElementById('role'),qa=document.getElementById('qa');
const purple=document.getElementById('purple'),occlusion=document.getElementById('occlusion');
const search=document.getElementById('search'),visible=document.getElementById('visibleCount');
function applyFilters(){{
  const term=search.value.trim().toLowerCase();let count=0;
  for(const c of cards){{
    const show=(role.value==='all'||c.dataset.role===role.value)&&
      (!qa.checked||c.dataset.qa==='1')&&(!purple.checked||c.dataset.purple==='1')&&
      (!occlusion.checked||c.dataset.occlusion==='1')&&
      (!term||c.dataset.search.toLowerCase().includes(term));
    c.hidden=!show;if(show)count++;
  }}
  visible.textContent='显示 '+count+' / '+cards.length+' 张';
}}
for(const control of [role,qa,purple,occlusion,search]){{
  control.addEventListener(control===search?'input':'change',applyFilters);
}}
applyFilters();
</script></body></html>\n"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_dir():
        raise NotADirectoryError(run)
    rows, qa, summary = load_plan(run)
    page = render_page(run, rows, qa, summary)
    destination = run / "review.html"
    temporary = run / "review.html.tmp"
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(destination)
    print(f"离线审阅页：{destination}；输入 {len(rows)} 张，固定 QA {sum(r['output_file'] in qa for r in rows)} 张")


if __name__ == "__main__":
    main()
