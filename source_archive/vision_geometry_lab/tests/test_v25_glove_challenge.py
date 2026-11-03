import numpy as np
import pytest

from vision_geometry_lab.v25_glove_challenge import source_contract, verify_raw_frame, summarize_rollout, write, safe
from vision_geometry_lab.v11_path_contract import array_sha256


def meta():
    return dict(session='right', frame_index=168, time_s=16.8, fps=10.,
        free_seed=[[1, 2], [21, 2]], frame_shape=[8, 9, 3],
        frame_sha256='example', available=True, seed_reason='old',
        geometry=dict(width_px=10., free_seed_xy=[[1, 2], [21, 2]], target_or_quality_used=False))


def test_reuse_seed_exactly():
    m = meta(); s = source_contract(m)
    assert s['past_seed_xy'] == m['free_seed']
    assert s['width_px'] == 10.
    assert s['seed_is_physical_contact'] is False


def test_wrong_frame_or_seed_or_width_rejected():
    for key, value in (('frame_index', 169), ('session', 'left'), ('available', False)):
        m = meta(); m[key] = value
        with pytest.raises(ValueError):
            source_contract(m)
    m = meta(); m['geometry']['width_px'] = float('nan')
    with pytest.raises(ValueError):
        source_contract(m)
    m = meta(); m['free_seed'][1] = [22, 2]
    with pytest.raises(ValueError):
        source_contract(m)


def test_exact_bgr_frame_binding():
    frame = np.arange(8*9*3, dtype=np.uint8).reshape(8, 9, 3)
    s = dict(frame_shape=list(frame.shape), frame_bgr_array_sha256=array_sha256(frame))
    verify_raw_frame(frame, s)
    with pytest.raises(ValueError):
        verify_raw_frame(frame[:, :, ::-1], s)
    changed = frame.copy(); changed[0, 0, 0] += 1
    with pytest.raises(ValueError):
        verify_raw_frame(changed, s)


def test_empty_is_not_correct_rejection():
    r = dict(segments=[], steps=[dict(accepted=False)], reason='no_connection')
    s = summarize_rollout(r, [[0, 0], [20, 0]], 10.)
    assert not s['nonempty'] and s['occlusion_accuracy'] is None
    assert s['stopped_is_correct_occlusion_rejection'] is None


def test_nonempty_is_not_visible_confirmation():
    r = dict(segments=[[[20, 0], [30, 0]], [[33, 0], [40, 0]]],
             steps=[dict(accepted=True), dict(accepted=True)], reason='short')
    s = summarize_rollout(r, [[0, 0], [20, 0]], 10.)
    assert s['predicted_total_arclength_px'] == 17. # Do not bridge the missing three pixels.
    assert s['furthest_seed_forward_projection_widths'] == 2.
    assert s['visibility_confirmed'] is False and s['glove_overlap_measured'] is False


def test_no_overwrite_or_blind_access(tmp_path):
    p = tmp_path/'x.json'; write(p, dict(a=1))
    with pytest.raises(FileExistsError):
        write(p, dict(a=2))
    with pytest.raises(ValueError):
        safe('D:/dual_20260925_00/frame.png')
