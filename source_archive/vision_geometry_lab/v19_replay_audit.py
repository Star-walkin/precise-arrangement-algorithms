"""Post-run determinism/prefix audit; does not read physical event truth."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vision_geometry_lab import v19_front_trial as trial


def prefix_arrays(arrays, end_s):
    count = int(np.count_nonzero(arrays['times_s'] <= end_s))
    temporal = {'times_s', 'frame_indices', *trial.FRAME_FIELDS}
    return {key: value[:count].copy() if key in temporal else value.copy()
            for key, value in arrays.items()}


def mutate_future(arrays, end_s):
    """Destroy future observation values, preserving valid time metadata."""
    changed = {key: value.copy() for key, value in arrays.items()}
    future = changed['times_s'] > end_s
    for key in trial.FRAME_FIELDS:
        if key == 'pose_valid':
            changed[key][future] = False
        else:
            changed[key] = changed[key].astype(float)
            changed[key][future] = np.nan
    return changed


def audit(base=trial.BASE):
    base = Path(base)
    protocol = trial.verify(base)
    output = base/'replay_audit.json'
    if output.exists():
        raise FileExistsError('do not overwrite replay audit')
    summaries = []
    for side in ('left', 'right'):
        path = base/side/'predictions.json'
        saved = trial._read(path)
        if saved['status'] != 'complete' or saved['protocol_sha256'] != trial.sha(base/'protocol.json'):
            raise ValueError('incomplete or mismatched prediction')
        _, arrays = trial._load_source(protocol['upstream_base'], side)
        full = trial.replay_arrays(arrays)
        for key in ('records', 'candidates', 'startup_model', 'quality_counts'):
            if saved[key] != full[key]:
                raise AssertionError('nondeterministic replay '+side+' '+key)
        checks = []
        for end_s in (2., 10., 30.):
            reference = [record for record in full['records'] if record['time_s'] <= end_s]
            prefix = trial.replay_arrays(prefix_arrays(arrays, end_s))
            mutated = trial.replay_arrays(mutate_future(arrays, end_s))
            if prefix['records'] != reference or mutated['records'][:len(reference)] != reference:
                raise AssertionError('future observation leaked into prefix '+side)
            expected = [c for c in full['candidates'] if c['first_available_s'] <= end_s]
            if prefix['candidates'] != expected:
                raise AssertionError('prefix candidate bookkeeping '+side)
            if [c for c in mutated['candidates'] if c['first_available_s'] <= end_s] != expected:
                raise AssertionError('future candidate bookkeeping '+side)
            checks.append(dict(end_s=end_s, frames=len(reference), prefix_identical=True,
                               future_observations_destroyed_prefix_identical=True))
        summaries.append(dict(session=side, prediction_sha256=trial.sha(path),
            complete_replay_identical=True, prefix_checks=checks))
        print('V19 prefix/determinism audit complete', side, flush=True)
    trial.verify(base)
    trial._write(output, dict(status='complete', protocol_sha256=trial.sha(base/'protocol.json'),
        audit_source_sha256=trial.sha(__file__), sessions=summaries,
        ground_truth_read=False, hardware_access=False,
        limitation='Implementation causality check, not evidence of physical event accuracy'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=trial.BASE)
    audit(parser.parse_args().base)
