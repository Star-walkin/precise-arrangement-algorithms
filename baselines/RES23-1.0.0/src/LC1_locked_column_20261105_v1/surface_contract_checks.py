"""Behavior checks: observed intervals, missing rows and crossing provenance."""
import json
from pathlib import Path
import numpy as np
from lc1_surface import SurfaceObserver


def main():
    observer = SurfaceObserver(np.full((400, 200), 255, np.uint8), 20)
    zero = np.zeros((400, 200), np.uint8)
    edge = dict(edge_y=100., edge_current_frame_confirmed=True)
    checks = []

    def count(ys, active=190., edge_value=edge, exclusion=zero):
        return observer._count(37.5, active, [dict(y=y) for y in ys], edge_value, exclusion)

    result, reasons, _ = count([110., 130., 150., 170., 190., 210.])
    assert result['count_valid'] and abs(result['continuous_pitch_distance'] - 4.5) < 1e-8, reasons
    checks.append('complete_observed_intervals_give_fractional_pitch_distance')
    result, reasons, _ = count([110., 130., 170., 190., 210.])
    assert not result['count_valid'] and result['continuous_pitch_distance'] is None
    assert 'unsupported_count_gap_gt_1_6_pitch' in reasons
    checks.append('missing_whole_groove_is_not_filled_from_average_pitch')
    result, reasons, _ = count([110., 130., 150., 170., 190., 250.], active=200.)
    assert not result['count_valid'] and 'active_phase_bracket_has_unobserved_large_gap' in reasons
    checks.append('endpoint_phase_cannot_interpolate_across_large_gap_outside_interval')
    result, reasons, _ = count([250., 270., 290., 310.], active=120.)
    assert not result['count_valid'] and 'edge_endpoint_far_from_observed_grooves' in reasons
    checks.append('short_requested_span_does_not_hide_a_distant_observation_gap')
    result, reasons, _ = count([110., 130., 150., 170., 190., 210.],
                              edge_value=dict(edge_y=100., edge_current_frame_confirmed=False))
    assert not result['count_valid'] and 'edge_is_static_prior_only' in reasons
    checks.append('static_mask_edge_candidate_never_certifies_count')

    def points(y, start_id=0):
        return [dict(x=x, y=y, score=.8, point_id=start_id+i) for i, x in enumerate([30, 35, 40, 45])]
    raw = dict(tracks=[dict(track_id='a', points=points(110)),
                       dict(track_id='duplicate_side', points=points(114, 10))], exclusion_mask=zero)
    _, segments = observer._observed_segments(raw)
    crossing, diag = observer._crossings(segments, 37.5, raw)
    assert len(crossing) == 1 and diag['duplicate'] == 1
    checks.append('nearby_double_sides_are_deduplicated_within_0_45_pitch')
    crossing, _ = observer._crossings(segments, 20., raw)
    assert crossing == []
    checks.append('raw_tracks_are_never_extrapolated_to_the_measurement_column')
    excluded = zero.copy(); excluded[110:115, 36:40] = 255
    crossing, diag = observer._crossings(segments, 37.5, dict(raw, exclusion_mask=excluded))
    assert crossing == [] and diag['excluded_bracket'] == 2
    checks.append('unknown_or_occluded_raw_bracket_is_not_a_measured_crossing')
    fit_only = dict(tracks=[], fit=dict(curves=[dict(sampled_points_xy=[[30, 110], [45, 110]])]))
    observed, segments = observer._observed_segments(fit_only)
    assert observed == [] and segments == []
    checks.append('fitted_geometry_is_not_promoted_to_raw_observation')
    value = dict(status='passed', checks=checks, count=len(checks),
                 scope='Deterministic count/crossing contract checks; no detection accuracy claim')
    root = Path(__file__).resolve().parent / 'surface_smoke_results'
    root.mkdir(exist_ok=True)
    (root / 'contract_checks.json').write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
