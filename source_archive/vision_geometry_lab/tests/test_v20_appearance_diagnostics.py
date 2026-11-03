import copy

import numpy as np
import pytest

from vision_geometry_lab import v20_appearance_control as core
from vision_geometry_lab.v20_appearance_diagnostics import (
    bucket_records, summarize_and_select, summary_figure, validate_timeline,
)


def make_records():
    result = []
    for i in range(601):
        result.append(dict(index=i, time_s=i / 10, frame_index=i, frame_sha256='a' * 64,
            histories=[dict(match=dict(history=h, current_index=i, time_s=i / 10,
                accepted=False, best_index=None, runner_index=None, control_index=None),
                comparisons={'untrusted_geometry': 'never_read'}) for h in core.HISTORIES]))
    return result


def test_final_time_60_kept_in_last_bucket():
    buckets = bucket_records(make_records(), 5)
    assert [len(x['records']) for x in buckets] == [50] * 11 + [51]
    assert buckets[-1]['records'][-1]['time_s'] == 60
    assert sum(len(x['records']) for x in bucket_records(make_records(), 10)) == 601


def test_first_accepted_not_closest_or_best_geometry():
    records = make_records()
    for i in (99, 100, 101, 200, 600):
        records[i]['histories'][0]['match']['accepted'] = True
    summaries, choices = summarize_and_select(records, core.HISTORIES[0])
    assert sum(x['accepted'] for x in summaries) == 5
    assert [x['selected_record_index'] for x in choices] == [99, 100, 200, None, None, 600]
    records[101]['histories'][0]['comparisons'] = {'arbitrary': 'prefer_me'}
    assert summarize_and_select(records, core.HISTORIES[0])[1] == choices


def test_zero_accepted_explicit_and_no_rejected_fallback():
    records = make_records()
    records[100]['histories'][0]['match']['best_index'] = 40
    buckets, selections = summarize_and_select(records, core.HISTORIES[0])
    assert all(r['status'] == 'no_accepted_pair' and r['selected_record_index'] is None for r in selections)
    assert all(r['accepted'] == 0 for r in buckets)
    assert summary_figure('right', core.HISTORIES[0], buckets, selections).size == (1480, 1110)


def test_timeline_binding_rejects_missing_frames_and_future_partner():
    records = make_records()
    source = [{k: r[k] for k in ('time_s', 'frame_index', 'frame_sha256')} for r in records]
    report, extract, index = dict(records=records), dict(records=source), dict(records=source)
    times, frames = np.arange(601) / 10, np.arange(601)
    validate_timeline(report, extract, index, times, frames, times, frames)
    bad = copy.deepcopy(report)
    bad['records'][100]['histories'][0]['match']['best_index'] = 100
    with pytest.raises(ValueError, match='strictly past'):
        validate_timeline(bad, extract, index, times, frames, times, frames)
    with pytest.raises(ValueError, match='array/source'):
        validate_timeline(report, extract, index, times[:-1], frames[:-1], times, frames)
    bad = copy.deepcopy(report)
    bad['records'][50]['frame_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='provenance'):
        validate_timeline(bad, extract, index, times, frames, times, frames)
