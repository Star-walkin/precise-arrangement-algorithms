"""Independent post-replay V19 scoring/figures; never a physical event detector.

Both complete frozen predictions and upstream bindings are validated before the
only manual-truth read. The first emitted candidate is scored, not the closest
one. Missing candidates have unknown errors, never zero errors. Every candidate
is retained. t0 and t2 are not implemented/evaluated; comparison to manual t2
only measures the source-video time margin, not a safe actuator deadline.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LAB = Path(__file__).resolve().parent
BASE = LAB / 'outputs/v19_front_candidates'
TRUTH_PATH = LAB.parent / 'vision_prototype/validation_ground_truth/dual_reversal_events.json'
SESSIONS = {'left': 'dual_20260924_111500', 'right': 'dual_20260924_111143'}
FONT_PATH = Path('C:/Windows/Fonts/msyh.ttc')
UNKNOWN, LOW, MID, HIGH, WARMUP = range(5)
COLORS = np.array([(75, 80, 88), (65, 175, 120), (225, 165, 55), (205, 85, 210), (40, 61, 85)], np.uint8)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_prediction(prediction, session, video, protocol_hash):
    if (prediction.get('status') != 'complete' or prediction.get('session') != session
            or prediction.get('protocol_sha256') != protocol_hash or prediction.get('video') != video):
        raise ValueError('prediction incomplete or protocol/session/video mismatch')
    for flag in ('physical_new_layer_confirmed', 'new_layer_confirmed', 'material_identity_confirmed',
                 'mechanical_revolution_confirmed', 'hardware_access', 'deployment_allowed'):
        if prediction.get(flag) is not False:
            raise ValueError('prediction illegally claims physical confirmation or deployment')
    if prediction.get('actuator_command') is not None:
        raise ValueError('actuator command is forbidden')
    records = prediction['records']
    if [r['frame_index'] for r in records] != video['indices']:
        raise ValueError('missing or reordered source frames')
    if not records:
        raise ValueError('empty prediction')
    rows = [r['row_y'] for r in records[0]['tracker']['row_states']]
    if not rows or not all(finite(y) for y in rows) or any(b <= a for a, b in zip(rows, rows[1:])):
        raise ValueError('invalid spatial row axis')
    flattened = []
    for record in records:
        now = record['time_s']
        if not finite(now) or abs(now - record['frame_index'] / video['fps']) > 1e-9:
            raise ValueError('source-frame time mismatch')
        tracker = record['tracker']
        if tracker['time_s'] != now:
            raise ValueError('tracker/record time mismatch')
        if any(tracker.get(flag) is not False for flag in ('material_identity_confirmed', 'new_layer_confirmed', 'mechanical_revolution_confirmed')):
            raise ValueError('tracker claims physical confirmation')
        if tracker.get('actuator_command') is not None:
            raise ValueError('tracker actuator command forbidden')
        if any(len(record[name]) != len(rows) for name in ('quality_mask', 'quality_reason', 'profile_residual')):
            raise ValueError('record row dimensions mismatch')
        states = tracker['row_states']
        if len(states) != len(rows):
            raise ValueError('tracker row count mismatch')
        for j, row in enumerate(states):
            if row['index'] != j or row['row_y'] != rows[j] or not isinstance(record['quality_mask'][j], bool):
                raise ValueError('row identity or quality type mismatch')
            if row['quality'] is not record['quality_mask'][j]:
                raise ValueError('tracker/direct quality mismatch')
            if row['displacement_px'] != record['profile_residual'][j]:
                raise ValueError('tracker/direct displacement mismatch')
            if now <= 5.:
                if row['phase'] != 'warmup' or row['quality'] or any(row[k] is not None for k in ('displacement_px', 'high_threshold_px', 'low_threshold_px')):
                    raise ValueError('warmup leaks startup-derived measurements')
            elif row['quality']:
                if not all(finite(row[k]) for k in ('displacement_px', 'high_threshold_px', 'low_threshold_px')):
                    raise ValueError('quality row has missing measurement/threshold')
                if not 0 < row['low_threshold_px'] < row['high_threshold_px']:
                    raise ValueError('invalid fixed thresholds')
        for candidate in tracker['new_candidates']:
            if candidate['first_available_s'] != now or candidate['first_available_frame_index'] != record['frame_index']:
                raise ValueError('candidate availability differs from emission record')
            flattened.append(candidate)
    if flattened != prediction['candidates']:
        raise ValueError('top-level candidates differ from chronological emitted list')
    previous = -math.inf
    seen = set()
    for candidate in flattened:
        onset, report, delay = (candidate[k] for k in ('estimated_onset_s', 'first_available_s', 'confirmation_delay_s'))
        interval = candidate['last_low_first_high_interval_s']
        if (not all(finite(x) for x in (onset, report, delay)) or not 0 <= onset <= report
                or report < previous or abs(report - onset - delay) > 1e-8
                or len(interval) != 2 or not all(finite(x) for x in interval)
                or not 0 <= interval[0] <= interval[1] == onset):
            raise ValueError('invalid causal candidate times or interval')
        if (candidate['estimated_onset'] != onset or candidate['first_available'] != report
                or candidate['delay'] != delay or candidate['lastlow_firsthigh_interval'] != interval):
            raise ValueError('candidate timing aliases disagree')
        if candidate['candidate_id'] in seen:
            raise ValueError('duplicate candidate identity')
        if any(candidate.get(flag) is not False for flag in ('material_identity_confirmed', 'new_layer_confirmed', 'mechanical_revolution_confirmed')) or candidate.get('actuator_command') is not None:
            raise ValueError('candidate claims a physical event or device command')
        seen.add(candidate['candidate_id'])
        previous = report
    if prediction['quality_counts']['frames'] != len(records) or prediction['quality_counts']['rows'] != len(rows):
        raise ValueError('quality totals have incorrect dimensions')
    return rows


def load_complete_cohort(base=BASE):
    """This function never reads annotations; all gate failures stop scoring."""
    from .v19_front_trial import verify
    base = Path(base).resolve()
    protocol = verify(base)
    protocol_hash = sha(base / 'protocol.json')
    upstream = Path(protocol['upstream_base'])
    predictions, bindings = {}, {str(base / 'protocol.json'): protocol_hash}
    for session in SESSIONS:
        path = base / session / 'predictions.json'
        prediction = read(path)
        validate_prediction(prediction, session, protocol['videos'][session], protocol_hash)
        for name, field in (('measurement.npz', 'source_measurement_sha256'), ('index.json', 'source_index_sha256')):
            source = upstream / session / name
            digest = sha(source)
            if prediction.get(field) != digest or protocol['dependencies'].get(str(source)) != digest:
                raise ValueError('prediction/upstream hash binding mismatch')
            bindings[str(source)] = digest
        predictions[session] = prediction
        bindings[str(path)] = sha(path)
    return protocol, predictions, bindings


def direct_state_matrix(prediction):
    """Do not turn historical confirmed states into current direct evidence."""
    records = prediction['records']
    matrix = np.full((len(records), len(records[0]['tracker']['row_states'])), UNKNOWN, np.uint8)
    for i, record in enumerate(records):
        for j, row in enumerate(record['tracker']['row_states']):
            if row['phase'] == 'warmup':
                matrix[i, j] = WARMUP
            elif row['quality'] and all(finite(row[k]) for k in ('displacement_px', 'low_threshold_px', 'high_threshold_px')):
                value = row['displacement_px']
                matrix[i, j] = LOW if value <= row['low_threshold_px'] else HIGH if value >= row['high_threshold_px'] else MID
    return matrix


def longest_true_run(flags):
    maximum = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        maximum = max(maximum, current)
    return maximum


def quality_summary(prediction):
    matrix = direct_state_matrix(prediction)
    after = np.array([r['time_s'] > 5. for r in prediction['records']])
    active = matrix[after]
    names = {'unknown': UNKNOWN, 'low': LOW, 'mid': MID, 'high': HIGH}
    counts = {name: int((active == code).sum()) for name, code in names.items()}
    consistent = (active == LOW) | (active == MID) | (active == HIGH)
    high_runs = [longest_true_run(row == HIGH) for row in active]
    low_runs = [longest_true_run(row == LOW) for row in active]
    quality_runs = [longest_true_run(row) for row in consistent]
    reasons = Counter()
    phase_counts = Counter()
    historical_confirmed_without_quality = 0
    for record in prediction['records']:
        if record['time_s'] <= 5.:
            continue
        reasons.update(record['quality_reason'])
        for row in record['tracker']['row_states']:
            if row['quality']:
                phase_counts[row['phase']] += 1
            elif row['phase'] == 'confirmed':
                historical_confirmed_without_quality += 1
    declared = prediction['quality_counts']
    if (declared['after_startup_cells'] != active.size
            or declared['profile_consistent_cells'] != int(consistent.sum())
            or declared['unknown_after_startup_cells'] != counts['unknown']):
        raise ValueError('declared profile quality totals disagree with direct row evidence')
    return dict(after_startup_frames=int(after.sum()), total_after_startup_cells=int(active.size),
        direct_state_cell_counts=counts, profile_consistent_cells=int(consistent.sum()),
        frames_with_three_adjacent_direct_high=sum(n >= 3 for n in high_runs),
        frames_with_three_adjacent_direct_low=sum(n >= 3 for n in low_runs),
        frames_with_three_adjacent_direct_quality=sum(n >= 3 for n in quality_runs),
        maximum_adjacent_direct_high=max(high_runs, default=0), maximum_adjacent_direct_low=max(low_runs, default=0),
        current_quality_tracker_phase_cells=dict(phase_counts),
        historical_confirmed_without_current_quality_cells=historical_confirmed_without_quality,
        quality_reason_counts_after_startup=dict(reasons),
        interpretation='image-profile support counts, NOT recognition accuracy or physical-event detection',
        unknown_means_no_thickening=False,
        adjacent_high_is_sufficient_to_confirm_front=False)


def score_candidates(candidates, manual):
    t1, t2 = manual.get('new_layer_start_s'), manual.get('new_layer_first_revolution_end_s')
    if not finite(t1) or not finite(t2) or t1 < 0 or t2 <= t1:
        raise ValueError('invalid manual t1/t2 reference schema')
    scored = []
    previous = -math.inf
    for rank, candidate in enumerate(candidates):
        onset, report = candidate['estimated_onset_s'], candidate['first_available_s']
        if not finite(onset) or not finite(report) or onset > report or report < previous:
            raise ValueError('candidate list must use chronological emission order')
        interval = candidate['last_low_first_high_interval_s']
        scored.append(dict(candidate_id=candidate['candidate_id'], emission_rank=rank + 1,
            is_primary_first_emitted=rank == 0, is_extra_candidate=rank > 0,
            estimated_onset_s=onset, first_available_s=report,
            confirmation_delay_s=report - onset, onset_error_s=onset - t1,
            absolute_onset_error_s=abs(onset - t1), first_report_minus_t1_s=report - t1,
            onset_is_earlier_than_manual_t1=onset < t1,
            report_is_earlier_than_manual_t1=report < t1,
            source_time_margin_before_manual_t2_s=t2 - report,
            report_not_later_than_manual_t2_on_source_axis=report <= t2,
            last_low_first_high_interval_s=interval,
            manual_t1_in_lastlow_firsthigh_interval=interval[0] <= t1 <= interval[1],
            physical_new_layer_confirmed=False, safe_actuation_timing_established=False))
        previous = report
    first = scored[0] if scored else None
    return dict(t1=dict(evaluation='image_candidate_onset_compared_to_approximate_manual_t1',
        candidate_count=len(scored), no_candidate=len(scored) == 0,
        extra_candidate_count=max(0, len(scored) - 1), truth_s=t1,
        first_candidate=first, onset_error_s=None if first is None else first['onset_error_s'],
        first_report_minus_t1_s=None if first is None else first['first_report_minus_t1_s'],
        source_time_margin_before_manual_t2_s=None if first is None else first['source_time_margin_before_manual_t2_s']),
        all_candidates=scored, manual_t2_reference_s=t2,
        t0=dict(evaluation_status='not_implemented_not_evaluated'),
        t2=dict(evaluation_status='not_implemented_not_evaluated'),
        matching_policy='first_emitted_never_closest_to_truth_all_extras_retained',
        reference_status='approximate_operator_development_annotation_not_blind_validation',
        margin_warning='source-video capture time only; excludes computation, communication and mechanical response latency',
        population_accuracy_estimated=False, deployment_allowed=False)


def score_from_base(base=BASE, truth_path=None):
    base = Path(base).resolve()
    output = base / 'score.json'
    if output.exists():
        raise FileExistsError('score exists; do not overwrite frozen trial report')
    protocol, predictions, bindings = load_complete_cohort(base)
    summaries = {session: quality_summary(p) for session, p in predictions.items()}
    # First and only annotation I/O. All complete predictions/bindings passed.
    truth_path = Path(truth_path) if truth_path is not None else TRUTH_PATH
    truth_digest = sha(truth_path)
    truth = read(truth_path)['sessions']
    cases = []
    for session, prediction in predictions.items():
        scores = score_candidates(prediction['candidates'], truth[SESSIONS[session]]['manual_review'])
        cases.append(dict(session=session, video_session=SESSIONS[session],
            prediction_sha256=bindings[str(base / session / 'predictions.json')],
            quality_summary=summaries[session], candidates=prediction['candidates'], scores=scores))
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError('input changed during independent scoring')
    if sha(truth_path) != truth_digest:
        raise ValueError('manual reference changed during scoring')
    report = dict(schema='v19_independent_front_review_1', protocol_sha256=bindings[str(base / 'protocol.json')],
        reviewer_script_sha256=sha(__file__), truth_sha256=truth_digest,
        truth_read_after_complete_bound_cohort=True, input_bindings=bindings, cases=cases,
        t0_status='not_implemented_not_evaluated', t2_status='not_implemented_not_evaluated',
        physical_new_layer_confirmed=False, hardware_access=False, deployment_allowed=False,
        stop_rule=protocol['stop_rule'], threshold_tuning_after_score=False)
    write_new(output, report)
    return report


def font(size):
    return ImageFont.truetype(str(FONT_PATH), size) if FONT_PATH.is_file() else ImageFont.load_default()


def _state_raster(matrix, times, width, height):
    raster = np.tile(COLORS[UNKNOWN], (matrix.shape[1], width, 1))
    half = min(.15, float(np.median(np.diff(times))) / 2) if len(times) > 1 else .05
    for i, now in enumerate(times):
        a, b = max(0, int(np.floor((now - half) / 60 * width))), min(width, int(np.ceil((now + half) / 60 * width)))
        if b > a:
            raster[:, a:b] = COLORS[matrix[i]][:, None, :]
    return Image.fromarray(raster).resize((width, height), Image.Resampling.NEAREST)


def timeline_figure(prediction, session, scores=None):
    records = prediction['records']
    times = np.array([r['time_s'] for r in records])
    matrix = direct_state_matrix(prediction)
    summary = quality_summary(prediction)
    rows = records[0]['tracker']['row_states']
    x, width = 150, 1860
    canvas = Image.new('RGB', (2100, 1290), '#111820')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), f'V19 {session} · 完整 0–60s 因果图像前沿候选（不是物理新层检测）', font=font(28), fill='white')
    draw.text((20, 55), '未知不代表没有增厚；低/中/高只表示本帧通过质量检查的法向位移。t0 / t2 未实施、未评估。', font=font(21), fill='#ffd580')
    draw.text((20, 91), '启动 0–5s 单独标记；保留原始缺失，不平滑、不补值。人工参考线仅事后评分，不是推理输入。', font=font(20), fill='#ced8e2')
    def tx(value): return x + width * value / 60
    def axis(y, height):
        for value in range(0, 61, 10):
            draw.line((tx(value), y, tx(value), y + height), fill='#52606c', width=1)
            draw.text((tx(value) - 12, y + height + 5), str(value) + 's', font=font(17), fill='#ced8e2')
    draw.text((20, 136), f"全部候选 {len(prediction['candidates'])} 个：橙点=回溯估计起点，蓝点=首次报告（从不挑最接近真值）", font=font(22), fill='white')
    draw.rectangle((x, 178, x + width, 318), fill='#202b35')
    for i, candidate in enumerate(prediction['candidates']):
        cy = 195 + (i % 4) * 29
        a, b = tx(candidate['estimated_onset_s']), tx(candidate['first_available_s'])
        draw.line((a, cy, b, cy), fill='#c6ccd3', width=2)
        draw.ellipse((a - 5, cy - 5, a + 5, cy + 5), fill='#ffb04d')
        draw.ellipse((b - 5, cy - 5, b + 5, cy + 5), fill='#42b5ff')
        draw.text((b + 8, cy - 12), str(i + 1), font=font(16), fill='white')
    if not prediction['candidates']:
        draw.text((x + 18, 225), '无图像前沿候选；时间误差不可计算，不记为 0', font=font(25), fill='#ffd580')
    axis(178, 140)
    counts = np.sum((matrix == LOW) | (matrix == MID) | (matrix == HIGH), axis=1)
    draw.text((20, 373), '本帧直接剖面质量支持行数（不是识别准确率）', font=font(23), fill='white')
    draw.rectangle((x, 415, x + width, 585), fill='#202b35')
    for a, b in zip(range(len(times) - 1), range(1, len(times))):
        if times[b] - times[a] <= .3 + 1e-9:
            draw.line((tx(times[a]), 585 - 170 * counts[a] / len(rows), tx(times[b]), 585 - 170 * counts[b] / len(rows)), fill='#63bbef', width=2)
    draw.text((20, 415), f'{len(rows)} 行', font=font(18), fill='#ced8e2')
    draw.text((20, 563), '0 行', font=font(18), fill='#ced8e2')
    axis(415, 170)
    draw.text((20, 640), '逐位置直接状态：深蓝 warmup；灰 unknown；绿 low；黄 mid-band；品红 high', font=font(22), fill='white')
    canvas.paste(_state_raster(matrix, times, width, 335), (x, 681))
    draw.text((20, 681), f"y={rows[0]['row_y']:.0f}px", font=font(17), fill='#ced8e2')
    draw.text((20, 990), f"y={rows[-1]['row_y']:.0f}px", font=font(17), fill='#ced8e2')
    axis(681, 335)
    if scores is not None:
        for value, color in ((scores['t1']['truth_s'], '#ffffcc'), (scores['manual_t2_reference_s'], '#9fd5ff')):
            px = tx(value)
            for py in range(178, 1016, 12):
                draw.line((px, py, px, min(py + 6, 1016)), fill=color, width=2)
    c = summary['direct_state_cell_counts']
    draw.text((20, 1080), f"t>5s 格数 {summary['total_after_startup_cells']}：unknown {c['unknown']} / low {c['low']} / mid {c['mid']} / high {c['high']}", font=font(22), fill='white')
    draw.text((20, 1120), f"至少3相邻直接 high 的帧数：{summary['frames_with_three_adjacent_direct_high']}；只满足空间条件，不等于按顺序推进或完成整圈。", font=font(20), fill='#ced8e2')
    draw.text((20, 1160), '质量不通过时，即使历史 phase=confirmed 也仍绘为 unknown，不冒充当前观测。', font=font(20), fill='#ced8e2')
    draw.text((20, 1200), '此图只审查证据和候选时序。不得据此直接启动伺服；未测端到端延迟、机械响应或真实新层身份。', font=font(20), fill='#ffd580')
    if scores is not None:
        # Put reference labels at fixed positions, not almost-overlapping time
        # coordinates; a short manual first-turn interval must stay readable.
        draw.text((20, 1240), f"仅后验对照：人工 t1={scores['t1']['truth_s']:.1f}s（淡黄虚线）；人工 t2={scores['manual_t2_reference_s']:.1f}s（浅蓝虚线）。", font=font(19), fill='#ced8e2')
    return canvas


def plot_from_base(base=BASE, score_path=None, output=None):
    base = Path(base).resolve()
    protocol, predictions, bindings = load_complete_cohort(base)
    scores = None
    if score_path is not None:
        report = read(score_path)
        if report['protocol_sha256'] != bindings[str(base / 'protocol.json')] or report['input_bindings'] != bindings:
            raise ValueError('score is not bound to this complete prediction cohort')
        scores = {case['session']: case['scores'] for case in report['cases']}
        if set(scores) != set(SESSIONS):
            raise ValueError('score cohort mismatch')
    output = Path(output) if output is not None else base / 'diagnostics' / ('review_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output.exists():
        raise FileExistsError('review directory exists; do not overwrite')
    output.mkdir(parents=True)
    figures = {}
    for session, prediction in predictions.items():
        path = output / (session + '_timeline.png')
        timeline_figure(prediction, session, None if scores is None else scores[session]).save(path)
        figures[session] = dict(path=path.name, sha256=sha(path), quality_summary=quality_summary(prediction))
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError('bound input changed while plotting')
    manifest = dict(schema='v19_front_visual_review_1', input_bindings=bindings,
        script_sha256=sha(__file__), figures=figures,
        manual_reference_source=None if score_path is None else dict(path=str(Path(score_path).resolve()), sha256=sha(score_path)),
        manual_reference_lines_posthoc_only=scores is not None, raw_truth_file_read_for_plot=False,
        events_not_implemented=['t0', 't2'], physical_new_layer_confirmed=False, hardware_access=False,
        display_rules_changed_from_data=False, full_source_timeline=True)
    write_new(output / 'manifest.json', manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('score', 'plot', 'all'))
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--truth', type=Path)
    parser.add_argument('--score', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage in ('score', 'all'):
        report = score_from_base(args.base, args.truth)
        print(json.dumps(dict(score=str(args.base / 'score.json'), candidates={case['session']: len(case['candidates']) for case in report['cases']}), ensure_ascii=False))
    if args.stage in ('plot', 'all'):
        score_path = args.base / 'score.json' if args.stage == 'all' else args.score
        print(json.dumps(dict(diagnostics=str(plot_from_base(args.base, score_path, args.output))), ensure_ascii=False))


if __name__ == '__main__':
    main()
