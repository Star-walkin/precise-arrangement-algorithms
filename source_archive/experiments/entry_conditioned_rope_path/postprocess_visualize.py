"""Immutable-response P0/P1 panels after the independent scoring save gate.

Scientific images use fixed original gray0..255 and response0..1. Solid supported
segments, dashed bridges and ambiguous/candidate paths never become body masks.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import visualize as v
from compare_evaluate import read, save, sha
from postprocess_evaluate import SOURCE, BRANCHES, METHODS, MODES, evidence_labels
from vision_geometry_lab.experiments.v28_local_spool_connection import data_eval

COLORS = {'entry_condition': '#fff400', 'heatmap_supported': '#38ee89',
          'image_supported': '#18c9ff', 'bridge': '#ffb030', 'ambiguous': '#ff4273'}
COMBOS = [(a, p) for a in BRANCHES for p in METHODS]


def draw_path(panel, points, color, width=2, dashed=False):
    """Accumulate dash phase over the complete polyline, including 4px edges."""
    if not dashed:
        v.line(panel, points, color, width)
        return
    phase = 0.0
    on, period = 6.0, 11.0  # display pixels, not original rope width
    for a, b in zip(np.asarray(points, float)[:-1], np.asarray(points, float)[1:]):
        length = float(np.linalg.norm(b-a)*panel['scale'])
        if length <= 1e-9:
            continue
        position = 0.0
        while position < length-1e-9:
            phase %= period
            boundary = on if phase < on else period
            take = min(length-position, boundary-phase)
            if take <= 1e-9:
                phase = boundary % period
                continue
            if phase < on:
                v.line(panel, [a+(b-a)*position/length,
                               a+(b-a)*(position+take)/length], color, width)
            position += take
            phase = (phase+take) % period


def state_runs(candidate, method):
    points = candidate['points_xy']; labels = evidence_labels(candidate, method)
    start = 0
    for stop in range(1, len(labels)+1):
        if stop == len(labels) or labels[stop] != labels[start]:
            yield labels[start], points[start:stop+1]
            start = stop


def seed_overlay(panel, seed):
    if not len(seed):
        return
    v.line(panel, seed, COLORS['entry_condition'], 3)
    v.marker(panel, seed[0], 'white', 4)
    if len(seed) > 1:
        a, b = np.asarray(seed[-2:]); d = b-a; d /= max(1e-6, np.linalg.norm(d))
        n = np.array([-d[1], d[0]])
        v.line(panel, [b-16*d+8*n, b, b-16*d-8*n], COLORS['entry_condition'], 3)


def path_overlay(panel, pred, method, seed):
    chosen = pred.get('selected_candidate_id')
    # Alternatives stay separate, never a union mask or a silently selected c0.
    for j, candidate in enumerate(pred['candidates']):
        if candidate['id'] != chosen:
            draw_path(panel, candidate['points_xy'], ['#b379ec', '#ff8dd0', '#8398ff'][j], 1, True)
    selected = next((c for c in pred['candidates'] if c['id'] == chosen), None)
    if selected:
        for state, points in state_runs(selected, method):
            draw_path(panel, points, COLORS[state], 3, state in {'bridge', 'ambiguous'})
        v.marker(panel, selected['points_xy'][-1], 'white', 4)
    else:
        v.text(panel['im'], (15, 40), '拒识；细虚线仅为未选候选' if pred['candidates'] else '无候选 / 入口不可用', 15, '#ff4474')
    seed_overlay(panel, seed)


def annotate_reference(panel, scored):
    for line in scored['source_reference']['trusted_clipped_centerline']:
        draw_path(panel, line, 'white', 1, True)
    seed_overlay(panel, scored['source_reference']['actual_input_seed_xy'])


def legend(board, y, size=17):
    x = 14
    for state, label in [('entry_condition', '入口条件→'), ('heatmap_supported', '热图支持 实线'),
                         ('image_supported', '原图支持 实线 / P0旧联合代理'), ('bridge', '桥接 虚线'),
                         ('ambiguous', '歧义 虚线')]:
        v.text(board, (x, y), label, size, COLORS[state]); x += len(label)*(size+1)


def put(board, panels, y, width, gap=10):
    for k, panel in enumerate(panels):
        board.paste(panel['im'], (10+k*(width+gap), y))


def load_case(run, branch, method, mode, sid):
    scored = read(run/'evaluation'/branch/method/mode/(sid+'.json'))
    paths = scored['files']['paths']
    if sha(paths) != scored['files']['paths_sha256']:
        raise ValueError('Prediction changed since independent evaluation')
    return {'score': scored, 'prediction': read(paths)}


def metric_text(scored, view='supported_only'):
    cov = scored['selected'][view]['all']['covered_fraction']
    stage = scored['selected'][view]['stages']['attached']['covered_fraction']
    def f(x): return 'unknown' if x is None else f'{100*x:.1f}%'
    return f"5px部分参考覆盖 {f(cov)} / 后段 {f(stage)} / {scored['selection_status']}"


def render(run, item, scope, mode, dest):
    sid = item['id']
    if sha(item['image_path']) != item['image_sha256']:
        raise ValueError('Original image hash mismatch')
    gray = cv2.imdecode(np.fromfile(item['image_path'], np.uint8), cv2.IMREAD_GRAYSCALE)
    cases = {combo: load_case(run, *combo, mode, sid) for combo in COMBOS}
    arrays = {}
    for branch in BRANCHES:
        a, b = cases[(branch, 'P0')]['score'], cases[(branch, 'P1')]['score']
        assert a['files']['response_sha256'] == b['files']['response_sha256']
        arrays[branch] = np.load(a['files']['response'], mmap_mode='r', allow_pickle=False)
    first = cases[('A', 'P0')]['score']; seed = first['source_reference']['actual_input_seed_xy']
    scene = read(item['scene_path']); bbox = item['bbox']; w, h = 600, 455
    board = Image.new('RGB', (2450, 1125), '#e6edf4')
    v.text(board, (14, 10), sid+' | '+mode+' | 冻结750步响应：P0旧连接 / P1提段与短桥', 23)
    v.text(board, (14, 46), '灰度0–255；A/B原始热图固定0–1，P0/P1共用相同float；白虚线为部分参考，未标区unknown', 19)
    legend(board, 78)
    original = v.panel(gray, bbox, '原始完整搜索ROI；种子不是新增识别', (w, h)); seed_overlay(original, seed)
    ref = v.panel(gray, bbox, '独立评分范围：部分中心 / 场景界面非终点身份', (w, h))
    annotate_reference(ref, first); v.scene_lines(ref, scene)
    panels = [original, ref]
    for branch in BRANCHES:
        panels.append(v.panel(gray, bbox, branch+' 原始网络响应（未作后处理修改）', (w, h), arrays[branch], True))
    put(board, panels, 115, w)
    panels = []
    for branch, method in COMBOS:
        c = cases[(branch, method)]
        p = v.panel(gray, bbox, branch+'/'+method+' 中心路径示意；无真实主体宽标签', (w, h))
        path_overlay(p, c['prediction'], method, seed)
        v.text(p['im'], (10, h-80), metric_text(c['score']), 13)
        panels.append(p)
    put(board, panels, 580, w)
    v.text(board, (14, 1045), '支持=算法的局部证据类别，不能确认同根；桥接仅为端点推断，不是检测或遮挡恢复。P0支持采用旧联合代理。', 19)
    v.text(board, (14, 1077), '所有拒识保留；未选候选为细紫虚线。完整边缘身份无独立真值，几何覆盖不是全弧段成功率。', 19)
    folder = dest/mode/sid; folder.mkdir(parents=True, exist_ok=False)
    board.save(folder/'full_comparison.png')

    # Original frozen scope, then a scene-only far-interface window: never
    # choose windows from P1 outputs or their errors.
    windows = [{'id': z['id'], 'bbox': z['bbox']} for z in scope['zooms'][:3]]
    edge = v.zooms_for(scene, item)[-1]
    windows.append({'id': 'edge_visual_proxy', 'bbox': edge['box']})
    w, h = 560, 600
    local = Image.new('RGB', (2860, 2620), '#e6edf4')
    v.text(local, (14, 10), sid+' | '+mode+' | 同坐标：自由 / 过渡 / 后段 / 远侧界面', 23)
    v.text(local, (14, 47), '每行原图+四组合；热图固定0–1且未修改；路径为证据状态，不把桥算可见绳。白细虚线=部分参考', 19)
    legend(local, 78)
    for row, window in enumerate(windows):
        p = v.panel(gray, window['bbox'], window['id']+' 原图 / 评分参考', (w, h))
        annotate_reference(p, first); panels = [p]
        for branch, method in COMBOS:
            c = cases[(branch, method)]
            p = v.panel(gray, window['bbox'], branch+'/'+method+' 同一原响应+保存路径', (w, h), arrays[branch], True)
            path_overlay(p, c['prediction'], method, seed); panels.append(p)
        put(local, panels, 120+row*615, w)
    v.text(local, (14, 2580), '阶段为原场景视觉代理；投影贴面不证明实际接触。相邻历史绳圈归属仍需独立身份证据。', 19)
    local.save(folder/'fixed_local_comparison.png')
    save(folder/'render_inputs.json', {'image_path': item['image_path'], 'image_sha256': item['image_sha256'],
        'windows': windows, 'window_policy': 'first3 old frozen scope windows plus unchanged scene far-interface proxy',
        'cases': {a+'/'+p: cases[(a,p)]['score']['files'] for a,p in COMBOS},
        'display': {'gray': [0,255], 'raw_response': [0,1]}, 'complete_body_mask_available': False})
    return [folder/'full_comparison.png', folder/'fixed_local_comparison.png']


def main(run):
    summary = read(run/'evaluation/summary.json')
    assert summary['status'] == 'completed' and summary['formal_saved_pairs'] == 64
    freeze = read(run/'evaluation_protocol.json')
    assert sha(SOURCE/'evaluation_scope/scope.json') == freeze['fixed_visual_scope_sha256']
    # References enter only through completed evaluator output. No label loader.
    scope = read(SOURCE/'evaluation_scope/scope.json')
    scopes = {i['sample_id']: i for i in scope['items']}
    index = {i['id']: i for i in read(SOURCE/'data/index.json')['items']}
    dest = run/'visualizations'; dest.mkdir(exist_ok=False); files = []
    cv2.setNumThreads(1)
    for mode in MODES:
        for sid in data_eval.SAMPLE_IDS:
            files.extend(render(run, index[sid], scopes[sid], mode, dest))
            print(mode+' '+sid, flush=True)
    for file in files:
        with Image.open(file) as im: im.verify()
    save(run/'visualization_receipt.json', {'status': 'completed', 'exit_code': 0,
        'source_sha256': sha(__file__), 'command': sys.argv, 'png_count': len(files),
        'verified_decode': True, 'files': [str(f) for f in files],
        'evaluation_summary_sha256': sha(run/'evaluation/summary.json'),
        'response_scale': [0,1], 'gray_scale': [0,255], 'formal_images': 8,
        'formal_groups': 3, 'modes': list(MODES), 'no_body_mask_claim': True,
        'dash_rule': '6 display pixels on / 5 off; phase accumulated across all edges in each continuous evidence run; never reset at 4px vertices',
        'pair0075_formal_inclusion': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    args = p.parse_args(); main(args.run.resolve())
