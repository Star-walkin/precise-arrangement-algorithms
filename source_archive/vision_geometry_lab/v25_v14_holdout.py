"""One frozen V14-final-joint development-holdout evaluation, not training.

Historical V14 gates and outputs are untouched. This newly authorized audit
evaluates the final ten-image joint model even though its original train gate
failed. It is explicitly NOT the historical V14 test result, nor a blind test.
Inference reads fixed V24 input-only archives; GT enters the later score step.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

LAB = Path(__file__).resolve().parent
BASE = LAB/'outputs/v25_status_review/v14_holdout'
OLD = LAB/'outputs/v14_joint_supervision'
V11 = LAB/'outputs/v11_path_identity'
INPUT_BASE = LAB/'outputs/v24_anchor_consistency/trial_001'
VARIANTS = ('manual', 'auto')


def safe(path):
    path = Path(path)
    if '20260925' in str(path):
        raise ValueError('Reserved blind source forbidden')
    return path


def read(path):
    return json.loads(safe(path).read_text('utf-8-sig'))


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def write(path, value):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def save(path, **arrays):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)


def select_entries(index, samples):
    chosen = {s['sample_id']: s for s in samples if s['split'] == 'holdout'}
    if len(chosen) != 8:
        raise ValueError('Exactly eight development holdout samples required')
    selected = []
    for sid in sorted(chosen):
        for variant in VARIANTS:
            matches = [index['entries'][key] for key in index['evaluation_keys']
                       if index['entries'][key]['sample_id'] == sid
                       and index['entries'][key]['variant'] == variant]
            if len(matches) != 1:
                raise ValueError('Missing or duplicate evaluation variant')
            e = matches[0]
            if e['split'] != 'holdout' or type(e['available']) is not bool:
                raise ValueError('Invalid holdout identity or availability')
            if '20260925' in sid:
                raise ValueError('Reserved blind identity forbidden')
            selected.append(e)
    return selected


def load_input(entry):
    if not entry['available']:
        if entry.get('input_path') or entry.get('input_sha256'):
            raise ValueError('Unavailable input must not contain hidden data')
        return None
    if sha(entry['input_path']) != entry['input_sha256']:
        raise ValueError('Input hash mismatch')
    with np.load(safe(entry['input_path']), allow_pickle=False) as a:
        if set(a.files) != {'input', 'valid_mask'}:
            raise ValueError('Inference arrays must exclude supervision')
        x, valid = a['input'].copy(), a['valid_mask'].copy()
    if x.shape != (8, 256, 512) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Invalid image input')
    if valid.shape != (256, 512) or not np.isin(valid, (0, 1)).all() or not np.array_equal(x[7], valid):
        raise ValueError('Invalid image validity')
    from .v11_path_contract import _checked_geometry
    _checked_geometry(entry['geometry'])
    if entry['geometry'].get('target_or_quality_used') is not False:
        raise ValueError('Geometry provenance missing')
    return dict(input=x, valid_mask=valid)


def freeze(base):
    if base.exists():
        raise FileExistsError('Never overwrite an audit run')
    from .v24_anchor_evaluate import DEPENDENCY_NAMES
    index_path = INPUT_BASE/'cache/input_index.json'
    index = read(index_path)
    if index['protocol_sha256'] != sha(INPUT_BASE/'protocol.json'):
        raise ValueError('Input protocol mismatch')
    manifest = read(V11/'development_manifest.json')
    entries = select_entries(index, manifest['samples'])
    checkpoint = OLD/'development/joint/checkpoint.pt'
    training = read(checkpoint.with_name('report.json'))
    if training['arm'] != 'joint' or training['phase'] != 'train' or training['steps'] != 2400:
        raise ValueError('Wrong preselected V14 final checkpoint')
    if training['checkpoint_sha256'] != sha(checkpoint) or training['protocol_sha256'] != sha(OLD/'protocol.json'):
        raise ValueError('V14 final model provenance mismatch')
    paths = {Path(__file__), checkpoint, checkpoint.with_name('report.json'),
             OLD/'protocol.json', OLD/'gate_single.json', OLD/'gate_train.json',
             index_path, INPUT_BASE/'protocol.json', V11/'development_manifest.json',
             V11/'development_cache/index.json', V11/'development/evaluation.json',
             Path(manifest['inventory_path'])}
    paths.update(LAB/name for name in (*DEPENDENCY_NAMES, 'v14_joint_boundary_model.py', 'v11_path_model.py'))
    for phase, folder in (('single', 'single'), ('train', 'development')):
        for arm in ('center_only', 'joint'):
            paths.add(OLD/folder/arm/('evaluation_'+phase+'.json'))
    selected_samples = [s for s in manifest['samples'] if s['split'] == 'holdout']
    for s in selected_samples:
        for key in ('image_path', 'annotation_path'):
            paths.add(safe(s[key]))
        q = Path(s['annotation_path']).with_suffix('.quality.json')
        if q.exists():
            paths.add(q)
    original = {e['sample_id']: e for e in read(V11/'development_cache/index.json')['entries']}
    for entry in entries:
        arrays = load_input(entry)
        if arrays is not None:
            paths.add(safe(entry['input_path']))
        if entry['variant'] == 'manual':
            old = original[entry['sample_id']]
            if hashlib.sha256(arrays['input'].tobytes()).hexdigest() != old['inference_input_sha256']:
                raise ValueError('Manual input differs from original V11/V14 input')
    snapshot = {str(p.resolve()): sha(p) for p in sorted(paths, key=str)}
    protocol = dict(schema='v25_v14_final_joint_holdout_1',
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        audit_date_local='2026-10-04', model_path=str(checkpoint), model_sha256=sha(checkpoint),
        selection='Preselected historical V14 development joint final checkpoint; no score-based selection.',
        input_index_path=str(index_path), input_index_sha256=sha(index_path),
        variants=list(VARIANTS), sample_ids=sorted(s['sample_id'] for s in selected_samples),
        training=False, checkpoint_updates=0, cpu_threads=2,
        manual_same_as_original=True, scoring='unchanged V6 full/downstream 5 native-pixel arc-length metrics',
        bend='unchanged V24 posthoc GT departure diagnostic, not physical contact or model input',
        missing='retain all eight references per variant; unavailable auto is empty prediction, not dropped',
        historical_gate_override='Separate newly authorized audit only; V14 gate failure remains unchanged.',
        split_warning='Eight reused development holdout images; not reserved blind validation.',
        historical_v14_holdout_scored=False, event_timing_validated=False, identity_confirmed=False,
        physical_precision_validated=False, hardware_access=False, deployment_authorized=False,
        stop='No training, checkpoint selection, decoder/threshold changes or tuning from this evaluation.',
        source_hashes=snapshot)
    write(base/'protocol.json', protocol)
    write(base/'inputs.json', dict(entries=entries, protocol_sha256=sha(base/'protocol.json')))
    write(base/'holdout_manifest.json', dict(manifest, samples=selected_samples))
    write(base/'freeze_receipt.json', dict(protocol_sha256=sha(base/'protocol.json'),
        inputs_sha256=sha(base/'inputs.json'), manifest_sha256=sha(base/'holdout_manifest.json')))
    return protocol


def verify(base):
    p = read(base/'protocol.json')
    receipt = read(base/'freeze_receipt.json')
    for name, key in (('protocol.json', 'protocol_sha256'), ('inputs.json', 'inputs_sha256'),
                      ('holdout_manifest.json', 'manifest_sha256')):
        if sha(base/name) != receipt[key]:
            raise ValueError('Audit snapshot changed: '+name)
    for path, expected in p['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Frozen source changed: '+path)
    if read(base/'inputs.json')['protocol_sha256'] != sha(base/'protocol.json'):
        raise ValueError('Input inventory unbound')
    return p


def infer(base):
    p = verify(base)
    folder = base/'exports'
    if folder.exists():
        raise FileExistsError('Inference cannot be repeated in the same run')
    import torch
    from .v14_joint_boundary_model import build_model
    from .v11_path_model import mask_normalized_probabilities
    from .v11_path_decode import decode_path
    torch.set_num_threads(p['cpu_threads'])
    checkpoint = torch.load(p['model_path'], map_location='cpu', weights_only=True)
    old_protocol = read(OLD/'protocol.json')
    if checkpoint['arm'] != 'joint' or checkpoint['phase'] != 'train' or checkpoint['steps'] != 2400:
        raise ValueError('Wrong checkpoint metadata')
    if checkpoint['protocol_sha256'] != sha(OLD/'protocol.json'):
        raise ValueError('Wrong checkpoint protocol')
    model = build_model(old_protocol['seed'], old_protocol['base_channels'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    rows = []
    with torch.inference_mode():
        for number, entry in enumerate(read(base/'inputs.json')['entries']):
            arrays = load_input(entry)
            row = {k: entry[k] for k in ('key', 'sample_id', 'split', 'variant', 'available', 'reason', 'input_sha256')}
            row.update(probability_path=None, probability_sha256=None, forward_ms=None, decode_ms=None,
                decoded=dict(segments=[], observed_segments=[], reason=entry['reason'], identity_confirmed=False))
            if arrays is not None:
                start = time.perf_counter()
                outputs = model(torch.from_numpy(arrays['input'][None]))
                probability = mask_normalized_probabilities(outputs, torch.from_numpy(arrays['valid_mask'][None] > .5))[0].numpy()
                slope = outputs['slope'][0].numpy()
                inferred = time.perf_counter()
                decoded = decode_path(probability, slope, entry['geometry'], arrays['valid_mask'])
                ended = time.perf_counter()
                path = folder/'probabilities'/f'{number:02d}.npz'
                save(path, probability=probability, slope=slope)
                row.update(decoded=decoded, probability_path=str(path), probability_sha256=sha(path),
                    forward_ms=(inferred-start)*1000, decode_ms=(ended-inferred)*1000)
            rows.append(row)
    verify(base)
    write(folder/'index.json', dict(rows=rows, model_sha256=p['model_sha256'],
        protocol_sha256=sha(base/'protocol.json'), input_index_sha256=sha(base/'inputs.json'),
        labels_used_for_inference=False, quality_used_for_decode=False, no_training=True,
        architecture='V14 JointBoundaryStripPathModel strict=True; center+slope only; edge logits unused',
        torch_version=str(torch.__version__), hardware_access=False))
    return len(rows)


def compact(value):
    return {k: v for k, v in value.items() if not isinstance(v, (list, dict))}


def historical_audit():
    result = dict(scope='historical training evidence only; this is not a holdout estimate', evaluations={})
    for phase, folder in (('single', 'single'), ('train', 'development')):
        for arm in ('center_only', 'joint'):
            path = OLD/folder/arm/('evaluation_'+phase+'.json')
            e = read(path)
            result['evaluations'][phase+'_'+arm] = dict(path=str(path), sha256=sha(path),
                summaries=[s for s in e['summaries'] if s['cohort'] == 'all' and s['samples']],
                per_sample=[dict(sample_id=r['sample_id'], split=r['split'],
                    downstream=compact(r['metrics']['downstream']), full_path=compact(r['metrics']['full_path'])) for r in e['rows']],
                boundaries={k: compact(v) for k, v in e['boundary_summary']['edges'].items()},
                width=compact(e['boundary_summary']['width']), synthetic_pair_pass_fraction=e['synthetic_pair_pass_fraction'])
    result.update(manual_seed_input=True, single_trained_image='dual_20260924_111500__left__f00246',
        single_real_updates=600, single_total_updates=1200, tolerance_native_px=5,
        synthetic_warning='16 two-query synthetic pairs are not 16 real identity examples.',
        historical_holdout_files_exist=bool(list(OLD.glob('**/evaluation_holdout.json'))),
        training_gate_passed=read(OLD/'gate_train.json')['decision']['passed'],
        physical_identity_or_event_validated=False)
    return result


def render_figures(base, rows, summaries, records, history):
    from PIL import Image, ImageDraw, ImageOps
    from .v18_front_diagnostics import font
    from .centerline_review_v5 import read_clean_image
    figures = []
    selected = [s for s in summaries if s['split'] == 'holdout' and s['scope'] == 'downstream' and s['cohort'] == 'all']
    im = Image.new('RGB', (1720, 1120), '#16212d'); d = ImageDraw.Draw(im)
    def text(y, msg, size=27, color='#eef4f9'):
        d.text((36, y), msg, font=font(size), fill=color)
    text(24, 'V14 的 100%：训练拟合通过，不是新画面自动识别100%', 38)
    text(90, '2026-10-04 独立补测 | 固定当时10图联合模型最终权重 | 无训练、无挑模型、无调阈值', 25, '#c7d9e8')
    text(158, '历史训练结果', 30, '#75d5ff')
    text(214, '单张训练图、人工种子后：100%（1382.324 / 1382.324 px 可信参考弧长在5px内）')
    text(267, '同一图完整可信路径：84.55%；双边：323/323 与 324/324 截面在5px内')
    text(320, '10张训练图联合组：97.81%；最差训练图右 f190：82.67%')
    text(373, '这些百分数不是逐图识别成功率；5px容差内不等于零误差，也不证明是同一根绳。', 25, '#ffd68b')
    text(447, '本日补做：8张开发留出（历史多次使用，非盲测）', 30, '#75d5ff')
    text(505, '输入方式                 可用输入      路径覆盖率        候选路径5px内比例       弯曲邻域覆盖', 26)
    for number, s in enumerate(selected):
        available = sum(r['variant'] == s['variant'] and r['prediction_status'] == 'loaded' for r in rows)
        pct = lambda x: '未知' if x is None else f'{x*100:.2f}%'
        msg = f"{s['variant']:8s}                     {available}/8          {pct(s['covered_fraction']):12s}       {pct(s['prediction_precision_at_5px']):12s}            {pct(s['GT_departure_neighborhood_coverage_5px'])}"
        text(561+number*57, msg, 27)
    text(711, 'manual：人工标出自由绳提示，模型推断后续路径；auto：冻结的自动入口。', 26)
    text(758, '自动入口失败的图保留在8图分母中，不删去；未知标签不作背景或成功。', 26)
    text(805, '弯曲邻域由人工路径事后定义，仅用于诊断；不是物理切点，不进入推理。', 26)
    text(879, '结论边界：未验证活动绳/入卷绳同根身份，未验证最后一圈、换层或施力时间。', 28, '#ffd68b')
    text(935, '历史V14并未做这8张留出评分；本页数值是本日独立补测，历史失败门槛保持不变。', 25)
    text(1000, '证据：evaluation.json / historical_audit.json / protocol.json（绑定源代码、权重、输入、标注哈希）', 23, '#c7d9e8')
    path = base/'summary.png'; im.save(path); figures.append(path)
    lookup = {r['sample']['sample_id']: r for r in records}
    for sid in sorted(lookup):
        source = read_clean_image(lookup[sid]['sample']).convert('RGB')
        canvas = Image.new('RGB', (1660, 900), '#16212d'); paint = ImageDraw.Draw(canvas)
        paint.text((25, 17), 'V14最终联合模型 | 开发留出 | '+sid, font=font(25), fill='white')
        paint.text((25, 62), '绿：可信人工参考（只评分）；品红：冻结模型候选；原图全幅等比，不裁难段。', font=font(24), fill='#cfdeee')
        for col, variant in enumerate(VARIANTS):
            row = next(r for r in rows if r['sample_id'] == sid and r['variant'] == variant)
            preview = ImageOps.contain(source, (800, 610)); scale = preview.width/source.width
            pd = ImageDraw.Draw(preview)
            for key, color in (('full_reference_segments', '#23ff77'), ('full_prediction_segments', '#ff36df')):
                for segment in row['metrics']['score_support'][key]:
                    if len(segment) >= 2:
                        pd.line([(float(x)*scale, float(y)*scale) for x, y in segment], fill=color, width=3)
            x = 20+col*830
            paint.text((x, 112), variant+' 输入', font=font(27), fill='white')
            canvas.paste(preview, (x, 157))
            m = row['metrics']['downstream']
            precision = m['prediction_precision_at_5px']
            label = '未知' if precision is None else f'{precision*100:.2f}%'
            paint.text((x, 768), f"可信下游覆盖 {m['covered_fraction']*100:.2f}% | 候选精度 {label}", font=font(23), fill='#ffd68b')
            msg = row['prediction_unavailable_reason'] or '路径是候选，不证明绳身份'
            paint.text((x, 810), msg, font=font(21), fill='#cfdeee')
        path = base/'figures'/(sid+'.png'); path.parent.mkdir(exist_ok=True); canvas.save(path); figures.append(path)
    return {p.stem: dict(path=str(p), sha256=sha(p)) for p in figures}


def score(base):
    p = verify(base)
    if (base/'evaluation.json').exists() or (base/'historical_audit.json').exists():
        raise FileExistsError('Scoring cannot overwrite an audit')
    from .v24_anchor_evaluate import decode_export, score_case, aggregate_fixed
    from .evaluate_centerline_multiscale_v9 import _dataset
    exports = read(base/'exports/index.json')
    entries = read(base/'inputs.json')['entries']
    if exports['protocol_sha256'] != sha(base/'protocol.json') or exports['input_index_sha256'] != sha(base/'inputs.json') or exports['model_sha256'] != p['model_sha256']:
        raise ValueError('Export bindings mismatch')
    pred = {r['key']: r for r in exports['rows']}
    if len(exports['rows']) != 16 or set(pred) != {e['key'] for e in entries}:
        raise ValueError('All 16 predictions must complete before GT scoring')
    _, records, _ = _dataset(base/'holdout_manifest.json')
    original = {e['sample_id']: e for e in read(V11/'development_cache/index.json')['entries']}
    old_scores = {e['sample_id']: e for e in read(V11/'development/evaluation.json')['rows']}
    lookup = {r['sample']['sample_id']: r for r in records}
    rows = []; hashes = {}
    for entry in entries:
        arrays = load_input(entry)
        decoded = decode_export(entry, pred[entry['key']], arrays, hashes)
        row = score_case(lookup[entry['sample_id']], entry, decoded, original[entry['sample_id']]['width'], arrays)
        for scope in ('downstream', 'full_path'):
            if not np.isclose(row['metrics'][scope]['reference_length_px'], old_scores[entry['sample_id']]['metrics'][scope]['reference_length_px'], rtol=0, atol=1e-6):
                raise ValueError('Fixed reference denominator changed')
        row.update(input_sha256=entry.get('input_sha256'), probability_sha256=pred[entry['key']].get('probability_sha256'))
        rows.append(row)
    summaries = [dict(variant=v, **s) for v in VARIANTS for s in aggregate_fixed([r for r in rows if r['variant'] == v])]
    history = historical_audit()
    figures = render_figures(base, rows, summaries, records, history)
    verify(base)
    write(base/'historical_audit.json', history)
    write(base/'evaluation.json', dict(schema='v25_v14_new_holdout_evaluation_1', rows=rows, summaries=summaries,
        protocol_sha256=sha(base/'protocol.json'), model_sha256=p['model_sha256'],
        exports_sha256=sha(base/'exports/index.json'), prediction_hashes=hashes,
        new_evaluation=True, historical_v14_result=False, blind_test=False,
        missing_references_retained=True, manual_same_as_original=True,
        gt_used_for_inference=False, identity_confirmed=False, physical_event_validated=False,
        timing='not evaluated; static geometry only', hardware_access=False, deployment_authorized=False,
        figures=figures, historical_audit_sha256=sha(base/'historical_audit.json')))
    return [s for s in summaries if s['samples'] and s['scope'] == 'downstream' and s['cohort'] == 'all']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'verify', 'infer', 'score'))
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args()
    result = globals()[args.stage](args.base.resolve())
    if args.stage in ('freeze', 'verify'):
        result = dict(protocol_sha256=sha(args.base/'protocol.json'), verified=True)
    print(json.dumps(dict(stage=args.stage, result=result), ensure_ascii=False))


if __name__ == '__main__':
    main()
