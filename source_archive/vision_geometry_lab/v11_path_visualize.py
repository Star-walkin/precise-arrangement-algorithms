"""Post-hoc V11 strip/candidate display, separate from frozen training/scoring.

No image, checkpoint, probability, score, or threshold is modified. The real
path overlay comes from saved decoder top1 segments, never from a GT-selected
candidate. Optional synthetic probes feed only the generated 8-channel inputs
to the model; analytical GT is drawn/scored strictly after inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LAB = Path(__file__).resolve().parent
DEFAULT_OUTPUT = LAB / 'outputs/v11_path_identity'
SAMPLE = 'dual_20260924_111500__left__f00246'
ARMS = ('single_real', 'single_identity')
GREEN, MAGENTA, AMBER = '#4aff79', '#ff48e2', '#ffd04c'
PANEL_W, PANEL_H, CAPTION_H = 768, 384, 76


def read_json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size):
    path = Path('C:/Windows/Fonts/arial.ttf')
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def require_files(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError('Required completed artifacts are missing; no waiting or partial rendering:\n  '
                                + '\n  '.join(missing))


def check_hash(path, expected, description):
    if not expected or sha(path) != expected:
        raise ValueError(description + ' hash mismatch: ' + str(path))


def load_single_artifacts(output):
    """Read only the single train cache and two corresponding saved results."""
    required = [output / 'protocol.json', output / 'single_cache/index.json']
    for arm in ARMS:
        required.extend(output / arm / name for name in ('report.json', 'exports.json', 'evaluation.json', 'checkpoint.pt'))
    require_files(required)
    protocol = read_json(output / 'protocol.json')
    # Source-code verification only. Deliberately do not open/hash the complete
    # dataset_snapshot: this diagnostic does not read any holdout image/label.
    for category in ('source_snapshot', 'old_source_snapshot'):
        for path, expected in protocol.get(category, {}).items():
            check_hash(path, expected, 'Frozen implementation')
    index_path = output / 'single_cache/index.json'
    index = read_json(index_path)
    check_hash(output / 'protocol.json', index['protocol_sha256'], 'Cache protocol')
    check_hash(index['manifest'], index['manifest_sha256'], 'Single manifest')
    if len(index['entries']) != 1 or index['entries'][0]['sample_id'] != SAMPLE:
        raise ValueError('Expected exactly the one f246 training sample')
    entry = index['entries'][0]
    if entry['split'] != 'train' or not entry['available']:
        raise ValueError('The single training strip is not available')
    check_hash(entry['cache'], entry['cache_sha256'], 'Single training cache')
    with np.load(entry['cache'], allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    if arrays['input'].shape != (8, 256, 512) or arrays['valid_mask'].shape != (256, 512):
        raise ValueError('Unexpected cached strip shape')
    if hashlib.sha256(arrays['input'].tobytes()).hexdigest() != entry['inference_input_sha256']:
        raise ValueError('Input array hash mismatch')
    results = {}
    for arm in ARMS:
        run = output / arm
        report, exports, evaluation = (read_json(run / name) for name in ('report.json', 'exports.json', 'evaluation.json'))
        for artifact in (report, exports, evaluation):
            if artifact['arm'] != arm or artifact['protocol_sha256'] != sha(output / 'protocol.json'):
                raise ValueError('Wrong arm/protocol in completed artifact: ' + arm)
        check_hash(run / 'checkpoint.pt', report['checkpoint_sha256'], 'Checkpoint')
        check_hash(run / 'checkpoint.pt', exports['checkpoint_sha256'], 'Export checkpoint')
        check_hash(run / 'checkpoint.pt', evaluation['checkpoint_sha256'], 'Evaluation checkpoint')
        check_hash(run / 'report.json', exports['training_report_sha256'], 'Export training report')
        check_hash(run / 'exports.json', evaluation['exports_sha256'], 'Saved scoring input')
        check_hash(index_path, report['cache_index_sha256'], 'Training cache index')
        check_hash(index_path, exports['cache_index_sha256'], 'Export cache index')
        predictions = [row for row in exports['records'] if row['sample_id'] == SAMPLE]
        scored = [row for row in evaluation['rows'] if row['sample_id'] == SAMPLE]
        if len(predictions) != 1 or len(scored) != 1 or len(evaluation['rows']) != 1:
            raise ValueError('Single-image exported/evaluated row mismatch')
        prediction = predictions[0]
        if not prediction['available'] or prediction['input_sha256'] != entry['inference_input_sha256']:
            raise ValueError('Prediction must use unchanged cached input')
        check_hash(prediction['path'], prediction['sha256'], 'Probability export')
        with np.load(prediction['path'], allow_pickle=False) as data:
            probability = data['probability'].copy()
        if probability.shape != (256, 512) or not np.isfinite(probability).all():
            raise ValueError('Probability image must be finite 256x512')
        if probability.min() < 0 or probability.max() > 1 + 1e-6:
            raise ValueError('Probability values outside [0,1]')
        results[arm] = dict(probability=probability, score=scored[0], report=report,
                            exports=exports, checkpoint=run / 'checkpoint.pt')
    return protocol, entry, arrays, results


def probability_rgb(probability):
    """One common fixed 0..1 lookup table, never per-panel contrast stretching."""
    positions = np.array([0., .25, .5, .75, 1.])
    palette = np.array([[12, 15, 28], [37, 65, 171], [25, 203, 194], [251, 224, 82], [244, 79, 46]])
    values = np.clip(np.asarray(probability), 0, 1)
    return np.stack([np.interp(values, positions, palette[:, channel])
                     for channel in range(3)], axis=-1).astype(np.uint8)


def contiguous_runs(points, keep):
    points, keep = np.asarray(points, float), np.asarray(keep, bool)
    keep = keep & np.isfinite(points).all(axis=1)
    changes = np.r_[0, np.flatnonzero(np.diff(keep.astype(np.int8))) + 1, len(keep)]
    return [points[start:end] for start, end in zip(changes[:-1], changes[1:])
            if keep[start] and end-start >= 2]


def reference_runs(arrays):
    distribution = np.asarray(arrays['target_distribution'], float)
    valid = (arrays['row_valid'] > .5) & (arrays['column_valid'][None] > .5)
    safe = np.where(valid, distribution, 0.)
    mass = safe.sum(0)
    rows = (safe * np.arange(256)[:, None]).sum(0) / np.maximum(mass, 1e-12)
    return contiguous_runs(np.column_stack([np.arange(512), rows]), mass > 0)


def draw_segments(drawer, offset, segments, color, width=2):
    sx, sy = PANEL_W / 512., PANEL_H / 256.
    for segment in segments:
        if len(segment) >= 2:
            vertices = [(offset[0] + float(x)*sx, offset[1] + float(y)*sy) for x, y in segment]
            drawer.line(vertices, fill=color, width=width)


def canvas_for_rows(count, title):
    canvas = Image.new('RGB', (2*PANEL_W, 58 + count*(PANEL_H+CAPTION_H) + 56), '#111827')
    drawer = ImageDraw.Draw(canvas)
    drawer.text((12, 12), title, fill='white', font=font(22))
    return canvas, drawer


def panel(canvas, drawer, row, column, pixels, title, subtitle):
    x, y = column*PANEL_W, 58 + row*(PANEL_H+CAPTION_H)
    drawer.text((x+10, y+7), title, fill='white', font=font(18))
    subtitle_font = font(14)
    for size in (14, 13, 12, 11, 10):
        subtitle_font = font(size)
        if drawer.textbbox((0, 0), subtitle, font=subtitle_font)[2] <= PANEL_W-20:
            break
    drawer.text((x+10, y+33), subtitle, fill='#cbd5e1', font=subtitle_font)
    image = Image.fromarray(np.uint8(np.clip(pixels, 0, 255))).resize((PANEL_W, PANEL_H), Image.Resampling.BILINEAR)
    canvas.paste(image, (x, y+CAPTION_H))
    return x, y+CAPTION_H


def format_fraction(value):
    return 'N/A' if value is None else f'{100*value:.2f}%'


def real_figure(entry, arrays, results):
    from .v11_path_contract import image_to_strip
    rgb = np.uint8(np.clip(arrays['input'][:3].transpose(1, 2, 0), 0, 1)*255)
    canvas, drawer = canvas_for_rows(3, 'V11 | f246 TRAINING strip | fixed exported top1, not field acceptance')
    panel(canvas, drawer, 0, 0, rgb, 'INPUT RGB ONLY',
          'Same strip for both arms; sampling from the original image was fixed before inference.')
    offset = panel(canvas, drawer, 0, 1, rgb, 'REVIEWED REFERENCE / INPUT PROMPT',
                   'Green = visible GT raster center; gold = input seed. Unknown GT gaps are not joined.')
    references = reference_runs(arrays)
    draw_segments(drawer, offset, references, GREEN, 3)
    seed = arrays['input'][4]
    seed_mass = seed.sum(0)
    seed_rows = (seed*np.arange(256)[:, None]).sum(0) / np.maximum(seed_mass, 1e-12)
    draw_segments(drawer, offset, contiguous_runs(np.column_stack([np.arange(512), seed_rows]),
                                                  seed_mass > .05), AMBER, 2)
    for row, arm in enumerate(ARMS, 1):
        result = results[arm]
        panel(canvas, drawer, row, 0, probability_rgb(result['probability']),
              arm + ' | CENTER ROW DISTRIBUTION',
              'Fixed colors: dark=0, blue=.25, cyan=.5, yellow=.75, red=1. Not a visibility score.')
        metric = result['score']['metrics']['downstream']
        caption = (f'Green=GT; magenta=saved decoder top1. Coverage {format_fraction(metric["covered_fraction"])}; '
                   f'precision {format_fraction(metric["prediction_precision_at_5px"])} at native 5px.')
        offset = panel(canvas, drawer, row, 1, rgb, arm + ' | ALGORITHM CANDIDATE VS GT', caption)
        draw_segments(drawer, offset, references, GREEN, 3)
        segments = [image_to_strip(np.asarray(segment, float), entry['geometry'])
                    for segment in result['score']['decoded']['segments']]
        draw_segments(drawer, offset, segments, MAGENTA, 2)
        if not segments:
            drawer.text((offset[0]+12, offset[1]+12), 'NO DECODER PATH', fill=MAGENTA, font=font(20))
    drawer.text((12, canvas.height-41), 'GT is drawn only after inference. Paths remain identity-unconfirmed; no contact, new-layer or force-time claim.',
                fill='white', font=font(16))
    return canvas


def synthetic_figure(protocol, results):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('--synthetic requires vision_training/.venv/Scripts/python.exe (torch); no cv2 is needed') from error
    from .v11_identity_probe import make_pair, assess_pair
    from .v11_path_model import build_model
    from .v11_path_experiment import probabilities
    torch.set_num_threads(int(protocol['cpu_threads']))
    pair = make_pair(900000, test=True)
    same_channels = [0, 1, 2, 3, 5, 6, 7]
    if not np.array_equal(pair['inputs'][0, same_channels], pair['inputs'][1, same_channels]):
        raise ValueError('Synthetic two queries must differ only in seed channel')
    rgb = np.uint8(np.clip(pair['inputs'][0, :3].transpose(1, 2, 0), 0, 1)*255)
    rows_after_inference = np.asarray(pair['rows'])  # Not passed to any network/decoder.
    canvas, drawer = canvas_for_rows(2, 'V11 | synthetic test seed 900000 | identical RGB, only prompt changes')
    details = {}
    for row_index, arm in enumerate(ARMS):
        checkpoint = torch.load(results[arm]['checkpoint'], map_location='cpu', weights_only=True)
        if checkpoint['arm'] != arm or checkpoint['protocol_sha256'] != results[arm]['report']['protocol_sha256']:
            raise ValueError('Synthetic checkpoint provenance mismatch')
        model = build_model(checkpoint['seed'], checkpoint['base_channels'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        # Only RGB/normalization/prompt/coordinate/valid input channels enter here.
        inputs = torch.from_numpy(pair['inputs']).float()
        with torch.inference_mode():
            probability = probabilities(model(inputs), inputs[:, 7]).numpy()
        predicted = probability.argmax(axis=1)
        details[arm] = assess_pair(probability, pair)
        for query in (0, 1):
            detail = details[arm]
            offset = panel(canvas, drawer, row_index, query, rgb, arm + f' | QUERY {query+1}',
                f'Green=selected synthetic GT; magenta=argmax; gold=prompt. '
                f'After-prompt 2-row coverage {100*detail["query_coverage_at_2_strip_rows"][query]:.1f}%.')
            other = np.column_stack([np.arange(512), rows_after_inference[1-query]])
            selected = np.column_stack([np.arange(512), rows_after_inference[query]])
            draw_segments(drawer, offset, [other], '#64748b', 1)
            draw_segments(drawer, offset, [selected], GREEN, 3)
            draw_segments(drawer, offset, [np.column_stack([np.arange(512), predicted[query]])], MAGENTA, 2)
            draw_segments(drawer, offset, [selected[:43]], AMBER, 3)
            drawer.line([(offset[0]+42*PANEL_W/512, offset[1]),
                         (offset[0]+42*PANEL_W/512, offset[1]+PANEL_H)], fill='#cbd5e1', width=1)
    drawer.text((12, canvas.height-41), 'Analytical synthetic curves only, not real neighbouring-rope labels. This picture is not field/generalization acceptance.',
                fill='white', font=font(16))
    return canvas, details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='Completed V11 experiment root, not an image filename')
    parser.add_argument('--synthetic', action='store_true', help='Also render fixed generated two-prompt test (requires torch)')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    folder = output / 'visuals_posthoc'
    targets = [folder / 'real_strip_comparison.png', folder / 'provenance.json']
    if args.synthetic:
        targets.append(folder / 'synthetic_same_rgb_two_prompts.png')
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError('Will not overwrite existing diagnostic output:\n  ' + '\n  '.join(existing))
    protocol, entry, arrays, results = load_single_artifacts(output)
    images = {folder / 'real_strip_comparison.png': real_figure(entry, arrays, results)}
    synthetic_details = None
    if args.synthetic:
        figure, synthetic_details = synthetic_figure(protocol, results)
        images[folder / 'synthetic_same_rgb_two_prompts.png'] = figure
    # Render in memory first; missing files or failed inference create no partial figures.
    folder.mkdir(parents=True, exist_ok=True)
    for path, figure in images.items():
        with path.open('xb') as stream:
            figure.save(stream, format='PNG')
    provenance = dict(diagnostic_only=True, script=str(Path(__file__).resolve()), script_sha256=sha(__file__),
        sample_id=SAMPLE, source_protocol_sha256=sha(output / 'protocol.json'),
        source_cache_sha256=entry['cache_sha256'], output_images={str(path): sha(path) for path in images},
        artifacts={arm: {name: sha(output / arm / name) for name in ('report.json', 'exports.json', 'evaluation.json', 'checkpoint.pt')}
                   for arm in ARMS},
        saved_real_decoder_top1_not_gt_selected=True, gt_affects_inference=False,
        synthetic_seed=900000 if args.synthetic else None, synthetic_details=synthetic_details,
        field_or_physical_accuracy_established=False, hardware_access=False,
        no_training_or_decoding_parameters_changed=True, no_holdout_opened=True)
    with (folder / 'provenance.json').open('x', encoding='utf-8') as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(images=[str(path) for path in images], provenance=str(folder / 'provenance.json')), ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        raise SystemExit('V11 visualization: ' + str(error))
