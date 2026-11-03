import numpy as np

from vision_geometry_lab.texture_boundary_probe_v5 import (
    FEATURES, _polygon_mask, brightness_bin, paired_metrics, probe_annotation, probe_pair,
)


def _stripes(seam=False, internal_seam=False):
    yy, xx = np.mgrid[:150, :220]
    phase = np.zeros_like(xx, float)
    if seam:
        phase[yy >= 60] += np.pi
    if internal_seam:
        phase[yy >= 72] += np.pi
    return (128 + 35 * np.sin(2 * np.pi * xx / 10 + phase)).astype(np.float32)


def _pair(image, **kwargs):
    return probe_pair(image, [110, 60], [1, 0], [0, 1], 30., **kwargs)


def test_texture_phase_seam_beats_internal_control():
    result = _pair(_stripes(seam=True))
    assert result['valid_pair']
    assert result['boundary_scores']['phase_decorrelation'] > .99
    assert result['interior_scores']['phase_decorrelation'] < .01
    assert paired_metrics([result], 'phase_decorrelation')['paired_win_rate'] == 1.


def test_texture_continuous_stripes_do_not_imply_boundary():
    result = _pair(_stripes())
    for key in ('boundary_scores', 'interior_scores'):
        assert result[key]['phase_decorrelation'] < .001
        assert result[key]['energy_jump'] < .001
    assert paired_metrics([result], 'texture_combined')['paired_win_rate'] == .5


def test_texture_internal_strand_seam_can_reverse_evidence():
    result = _pair(_stripes(internal_seam=True))
    assert result['interior_scores']['phase_decorrelation'] > .99
    assert result['boundary_scores']['phase_decorrelation'] < .01
    assert paired_metrics([result], 'phase_decorrelation')['paired_win_rate'] == 0.


def test_texture_direction_and_energy_change_are_measurable():
    yy, xx = np.mgrid[:150, :220]
    direction = np.where(yy < 60, np.sin(2 * np.pi * xx / 10),
                         np.sin(2 * np.pi * (xx + yy) / 10))
    result = _pair((128 + 35 * direction).astype(np.float32))
    assert result['boundary_scores']['orientation_jump'] > .2
    assert result['interior_scores']['orientation_jump'] < .01
    energy = 128 + np.where(yy < 60, 35., 15.) * np.sin(2 * np.pi * xx / 10)
    result = _pair(energy.astype(np.float32))
    assert result['boundary_scores']['energy_jump'] > .3
    assert result['interior_scores']['energy_jump'] < .01


def test_texture_brightness_step_is_not_fabricated_texture():
    image = np.full((150, 220), 80., np.float32)
    image[60:] = 160.
    result = _pair(image)
    assert result['boundary_scores']['brightness_gradient'] > 10.
    assert result['interior_scores']['brightness_gradient'] == 0.
    assert result['boundary_scores']['texture_combined'] is None


def test_texture_black_and_saturated_are_missing_not_success():
    for value, reason in ((0., 'black_or_too_dark'), (255., 'saturated_or_overexposed')):
        result = _pair(np.full((150, 220), value, np.float32))
        assert not result['valid_pair'] and result['reason'] == reason
        for feature in FEATURES:
            metric = paired_metrics([result], feature)
            assert metric['valid_pairs'] == 0 and metric['coverage'] == 0.
            assert metric['auc'] is None and metric['paired_win_rate'] is None


def test_texture_flat_midgray_retains_brightness_but_abstains_texture():
    result = _pair(np.full((150, 220), 128., np.float32))
    assert result['valid_pair']
    assert result['boundary_scores']['brightness_gradient'] == 0.
    assert result['boundary_scores']['phase_decorrelation'] is None
    assert result['texture_reasons'] == ['low_texture_amplitude'] * 2
    assert paired_metrics([result], 'texture_combined')['coverage'] == 0.


def test_texture_quality_excludes_whole_band_not_just_anchor():
    # Anchor [110,60] is outside this polygon; its along-rope support intersects it.
    mask = _polygon_mask((150, 220), [[[125, 52], [130, 52], [130, 80], [125, 80]]])
    result = _pair(_stripes(True), exclusion_mask=mask)
    assert not result['valid_pair'] and result['reason'] == 'quality_or_manual_exclusion'


def test_texture_negative_must_stay_inside_manual_body():
    mask = _polygon_mask((150, 220), [[[0, 60], [219, 60], [219, 70], [0, 70]]])
    result = _pair(_stripes(), body_mask=mask)
    assert result['reason'] == 'internal_control_not_inside_manual_body'


def test_texture_annotation_samples_both_edges_and_respects_exclusions():
    label = dict(polylines=dict(edge_flange=[[30, 60], [190, 60]], edge_surface=[[30, 90], [190, 90]]))
    result = probe_annotation(_stripes(), label)
    assert {r['edge'] for r in result} == {'edge_flange', 'edge_surface'}
    assert any(r['valid_pair'] for r in result)
    excluded = probe_annotation(_stripes(), label, [[[0, 0], [219, 0], [219, 149], [0, 149]]])
    assert not any(r['valid_pair'] for r in excluded)
    assert paired_metrics(excluded, 'phase_decorrelation')['auc'] is None


def test_texture_fixed_brightness_bins_and_empty_summary():
    assert [brightness_bin(x) for x in (0, 32, 96, 192, 240, None)] == [
        'very_dark', 'dim', 'mid', 'bright', 'near_clipping', 'unavailable']
    metric = paired_metrics([], 'phase_decorrelation')
    assert metric['coverage'] is None and metric['auc'] is None
