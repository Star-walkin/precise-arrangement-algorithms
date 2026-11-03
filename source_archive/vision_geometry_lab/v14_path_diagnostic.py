"""Post-hoc, read-only V14 TRAIN path and quantization diagnostics.

No model training, inference, RGB image access or held-out array access.
Writes only a new exclusive diagnostic JSON, never a frozen experiment file.
GT is used for retrospective scoring/rounding only, not for prediction repair.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from .v11_path_contract import image_to_strip
from .v11_path_decode import _column_nodes


LAB = Path(__file__).resolve().parent
BASE = LAB / 'outputs/v14_joint_supervision'
TARGETS = LAB / 'outputs/v13_boundary_targets_train/summary.json'
INDEX = LAB / 'outputs/v11_path_identity/development_cache/index.json'
ARMS = ('center_only', 'joint')
FOCUS_SOURCE = 'dual_20260924_111143__right__'
TOLERANCE_PX = 5.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spans(values):
    values = sorted(map(int, values))
    if not values:
        return []
    result = []
    first = last = values[0]
    for value in values[1:]:
        if value == last + 1:
            last = value
        else:
            result.append([first, last])
            first = last = value
    return result + [[first, last]]


class Reader:
    def __init__(self):
        self.snapshot = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError('Artifact hash mismatch: ' + str(path))
        previous = self.snapshot.setdefault(str(path), value)
        if previous != value:
            raise ValueError('Artifact changed during diagnostic: ' + str(path))
        return path

    def json(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text(encoding='utf-8-sig'))

    def arrays(self, path, expected, keys):
        with np.load(self.bind(path, expected), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in keys}

    def verify(self):
        for path, expected in self.snapshot.items():
            if digest(path) != expected:
                raise ValueError('Diagnostic input changed: ' + path)


def checked_stage(reader, phase, protocol_sha):
    folder = BASE / ('single' if phase == 'single' else 'development')
    gate = reader.json(BASE / ('gate_' + phase + '.json'))
    if gate['protocol_sha256'] != protocol_sha:
        raise ValueError('Foreign gate')
    evaluations, predictions, reports = {}, {}, {}
    for arm in ARMS:
        root = folder / arm
        report_path = root / 'report.json'
        reports[arm] = reader.json(report_path)
        exports_path = root / ('exports_' + phase + '.json')
        exports = reader.json(exports_path)
        evaluation = reader.json(root / ('evaluation_' + phase + '.json'),
                                 gate['evaluation_sha256'][arm])
        if any(value['protocol_sha256'] != protocol_sha
               for value in (reports[arm], exports, evaluation)):
            raise ValueError('Stage protocol mismatch')
        if any(value['phase'] != phase or value['arm'] != arm
               for value in (reports[arm], exports, evaluation)):
            raise ValueError('Stage/arm mismatch')
        reader.bind(root / 'checkpoint.pt', reports[arm]['checkpoint_sha256'])
        if (exports['checkpoint_sha256'] != reports[arm]['checkpoint_sha256'] or
                evaluation['checkpoint_sha256'] != reports[arm]['checkpoint_sha256'] or
                exports['training_report_sha256'] != digest(report_path) or
                evaluation['exports_sha256'] != digest(exports_path)):
            raise ValueError('Stage result chain mismatch')
        rows = evaluation['rows']
        if any(row['split'] != 'train' for row in rows):
            raise ValueError('Only training evaluations are permitted')
        names = [row['sample_id'] for row in rows]
        exported = [row['sample_id'] for row in exports['records']]
        if len(set(names)) != len(names) or set(names) != set(exported) or len(exported) != len(names):
            raise ValueError('Prediction/evaluation sample mismatch')
        for item in exports['records']:
            reader.bind(item['path'], item['sha256'])
        evaluations[arm] = {row['sample_id']: row for row in rows}
        predictions[arm] = {row['sample_id']: row for row in exports['records']}
    for key in ('initial_state_sha256', 'sequence_sha256', 'input_sha256',
                'cache_sha256', 'steps', 'optimizer', 'learning_rate',
                'torch_version', 'numpy_version'):
        if reports[ARMS[0]][key] != reports[ARMS[1]][key]:
            raise ValueError('Unmatched arms: ' + key)
    return evaluations, predictions, gate


def downstream(geometry):
    return (geometry['u_first_widths'] + np.arange(geometry['shape_hw'][1]) *
            geometry['step_widths']) >= 0


def train_deltas(evaluations):
    results = []
    for name, base in evaluations['center_only'].items():
        joint = evaluations['joint'][name]
        b, j = base['metrics']['downstream'], joint['metrics']['downstream']
        edges = joint['boundary']['edges']
        results.append(dict(sample_id=name, group_id=joint['group_id'],
            baseline_coverage=b['covered_fraction'], joint_coverage=j['covered_fraction'],
            coverage_delta_pp=100*(j['covered_fraction']-b['covered_fraction']),
            covered_reference_delta_px=j['covered_reference_length_px']-b['covered_reference_length_px'],
            baseline_precision=b['prediction_precision_at_5px'],
            joint_precision=j['prediction_precision_at_5px'],
            joint_longest_uncovered_reference_run_px=j['longest_uncovered_reference_run_px'],
            joint_edge_hits={key: value['hit_5px_fraction'] for key, value in edges.items()},
            joint_width_relative_mae=joint['boundary']['width']['mean_absolute_relative_error'],
            joint_human_center_bracket_fraction=joint['boundary']['joint']['strict_center_bracket_fraction']))
    return results


def quantization(reader, single, target):
    a = reader.arrays(target['target_npz'], target['target_npz_sha256'],
                      ('edge_rows', 'edge_valid', 'width_valid', 'local_width_px'))
    g = target['geometry']
    step = g['native_px_per_row']
    allowed = downstream(g)
    # Nearest integer row, with exact half-row ties choosing the smaller row.
    rounded = np.ceil(a['edge_rows'].astype(float)-.5)
    width_scope = a['width_valid'] & allowed
    truth = a['local_width_px'][width_scope]
    predicted_width = (rounded[1]-rounded[0])[width_scope]*step
    relative = np.abs(predicted_width-truth)/truth
    actual = single['joint'][target['sample_id']]['boundary']
    actual_rows = np.asarray(actual['decoded']['edge_rows'], float)
    sides = {}
    for side, name in enumerate(('low_normal', 'high_normal')):
        valid = a['edge_valid'][side] & allowed
        errors = np.abs(rounded[side, valid]-a['edge_rows'][side, valid])*step
        offsets = actual_rows[side, valid]-rounded[side, valid]
        sides[name] = dict(trusted_columns=int(valid.sum()),
            rounding_mean_error_px=float(errors.mean()),
            rounding_p95_error_px=float(np.percentile(errors, 95)),
            actual_exact_nearest_gt_row=int(np.sum(offsets == 0)),
            actual_offset_from_nearest_gt_row_counts={str(float(v)): int(np.sum(offsets == v))
                                                     for v in np.unique(offsets)})
    return dict(sample_id=target['sample_id'], ground_truth_is_input=True,
        model_performance=False, representation_reference_not_theoretical_lower_bound=True,
        actual_model_numbers_are_separate=True, rounding_tie='smaller row',
        native_px_per_strip_row=step, five_px_in_strip_rows=TOLERANCE_PX/step,
        trusted_width_columns=int(width_scope.sum()),
        independently_rounded_edges_width_relative_mae=float(relative.mean()),
        independently_rounded_edges_width_native_mae=float(np.abs(predicted_width-truth).mean()),
        independently_rounded_edges_width_relative_p95=float(np.percentile(relative, 95)),
        width_only_nearest_integer_step_relative_mae=float(np.mean(
            np.abs(np.ceil(truth/step-.5)*step-truth)/truth)),
        true_projected_width_px_p05_median_p95=np.percentile(truth, [5, 50, 95]).tolist(),
        actual_joint_width_relative_mae=actual['width']['mean_absolute_relative_error'],
        actual_joint_width_native_mae=actual['width']['mae_px'], edges=sides,
        physical_diameter_accuracy=False)


def probability_diagnostic(reader, name, entry, target, evaluations, predictions):
    if entry['split'] != 'train' or target['split'] != 'train':
        raise ValueError('Training arrays only')
    valid = reader.arrays(entry['cache'], entry['cache_sha256'], ('valid_mask',))['valid_mask']
    a = reader.arrays(target['target_npz'], target['target_npz_sha256'], ('center_rows', 'center_valid'))
    g = entry['geometry']
    step = g['native_px_per_row']
    scope = a['center_valid'] & downstream(g)
    cols = np.flatnonzero(scope)
    truth = a['center_rows']
    result = dict(sample_id=name, trusted_column_range=[int(cols.min()), int(cols.max())],
        trusted_columns=len(cols), native_px_per_strip_row=step, arms={},
        ground_truth_used_for_diagnostic_comparison_only=True,
        same_column_error_not_arc_length_score=True, no_new_decoding_threshold=True)
    for arm in ARMS:
        item = predictions[arm][name]
        p = reader.arrays(item['path'], item['sha256'], ('probability',))['probability'].astype(float)
        predicted = np.argmax(np.where(valid, p, -np.inf), axis=0)
        peak_error = (predicted-truth)*step
        decoded = np.full(g['shape_hw'][1], np.nan)
        row = evaluations[arm][name]
        for segment in row['decoded']['segments']:
            uv = image_to_strip(np.asarray(segment), g)
            for u, v in uv:
                if abs(u-round(u)) > 1e-5:
                    raise ValueError('Expected original integer-column decoder nodes')
                decoded[int(round(u))] = v
        error = (decoded-truth)*step
        present = scope & np.isfinite(decoded)
        bad = present & (np.abs(error) > TOLERANCE_PX)
        near, strongest = [], []
        radius = row['decoded']['diagnostics']['window_radius_rows']
        for column in cols:
            nodes = _column_nodes(p, valid, int(column), radius)
            if any(abs(node['row']-truth[column])*step <= TOLERANCE_PX for node in nodes):
                near.append(column)
            if nodes and abs(nodes[0]['row']-truth[column])*step <= TOLERANCE_PX:
                strongest.append(column)
        result['arms'][arm] = dict(
            raw_argmax_hit_fraction=float(np.mean(np.abs(peak_error[cols]) <= TOLERANCE_PX)),
            raw_argmax_abs_error_p95_px=float(np.percentile(np.abs(peak_error[cols]), 95)),
            decoded_present_columns=int(present.sum()),
            decoded_same_column_hit_fraction=float(np.sum(present & (np.abs(error) <= TOLERANCE_PX))/len(cols)),
            decoded_abs_error_p95_px=float(np.percentile(np.abs(error[present]), 95)),
            decoded_error_gt5_columns=int(bad.sum()),
            any_fixed_candidate_within5px=len(near), strongest_fixed_candidate_within5px=len(strongest),
            bad_decoded_but_raw_argmax_within5px=int(np.sum(bad & (np.abs(peak_error) <= TOLERANCE_PX))),
            bad_decoded_but_any_fixed_candidate_within5px=len(set(np.flatnonzero(bad)) & set(near)),
            bad_decoded_column_runs=spans(np.flatnonzero(bad)),
            missing_decoded_column_runs=spans(np.flatnonzero(scope & ~present)),
            bad_decoder_signed_mean_px=float(np.mean(error[bad])) if bad.any() else None)
    return result


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output path; never overwrite diagnostics')
    reader = Reader()
    protocol = reader.json(BASE/'protocol.json')
    protocol_sha = digest(BASE/'protocol.json')
    for path, expected in protocol['source_snapshot'].items():
        reader.bind(path, expected)
    reader.bind(Path(__file__))
    target_rows = reader.json(TARGETS, protocol['artifact_snapshot'][str(TARGETS)])['rows']
    targets = {row['sample_id']: row for row in target_rows}
    if len(targets) != 10 or any(row['split'] != 'train' for row in target_rows):
        raise ValueError('Expected exactly ten frozen training targets')
    index = reader.json(INDEX, protocol['artifact_snapshot'][str(INDEX)])
    entries = {row['sample_id']: row for row in index['entries'] if row['sample_id'] in targets}
    if set(entries) != set(targets) or any(row['split'] != 'train' for row in entries.values()):
        raise ValueError('Training cache scope mismatch')
    if '20260925' in json.dumps(target_rows):
        raise ValueError('Reserved blind source prohibited')
    single, _, single_gate = checked_stage(reader, 'single', protocol_sha)
    train, predictions, train_gate = checked_stage(reader, 'train', protocol_sha)
    if any(set(rows) != set(targets) for rows in train.values()):
        raise ValueError('Train evaluation scope mismatch')
    if any(set(rows) != {protocol['single_sample']} for rows in single.values()):
        raise ValueError('Single evaluation scope mismatch')
    result = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='post-hoc TRAIN-only audit of frozen V14 outputs; no new model performance claim',
        protocol_sha256=protocol_sha, training=False, inference=False, rgb_read=False,
        heldout_arrays_or_images_read=False, reserved_blind_read=False,
        thresholds_or_frozen_sources_modified=False, hardware_access=False, force_command=None,
        single_gate_passed=single_gate['decision']['passed'], train_gate_passed=train_gate['decision']['passed'],
        train_per_image=train_deltas(train),
        single_quantization=quantization(reader, single, targets[protocol['single_sample']]),
        focus_train_probabilities=[probability_diagnostic(reader, name, entries[name], targets[name], train, predictions)
                                   for name in targets if FOCUS_SOURCE in name],
        interpretation_limits=[
            'Per-column probability diagnostics are not the frozen arc-length path metric.',
            'Training fit and GT rounding do not establish physical or unseen-scene accuracy.',
            'Absence of a good fixed node is not proof that every possible decoder would fail.',
            'Auxiliary gradient conflict is a hypothesis, not identified by these forward outputs.',
            'Boundary width/bracketing does not prove correct rope identity.'])
    reader.verify()
    result['source_and_input_snapshot'] = reader.snapshot
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), sha256=digest(output), train_images=len(result['train_per_image']),
        focus_images=len(result['focus_train_probabilities']),
        quantization_relative_mae=result['single_quantization']['independently_rounded_edges_width_relative_mae']),
        ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE/'diagnostics/path_diagnostic.json')
    run(parser.parse_args().output)
