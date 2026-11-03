import numpy as np
import pytest
from PIL import Image

from vision_geometry_lab.v20_material_review import schedule_records
from vision_geometry_lab.v21_bulge_review import (
    TIMES_S, USER_QUOTE, REFERENCE_CONTEXT, contact_sheet, export, full_raw_picture,
)


def test_four_exact_bound_frames():
    assert TIMES_S == (26.8, 27.3, 27.8, 28.3)
    index = dict(video=dict(fps=10.), records=[dict(frame_index=i, time_s=i / 10.) for i in range(601)])
    assert [r['frame_index'] for r in schedule_records(index, TIMES_S)] == [268, 273, 278, 283]


def test_native_full_image_preserves_pixels_size_orientation():
    frame = np.arange(12 * 17 * 3, dtype=np.uint8).reshape(12, 17, 3)
    before = frame.copy()
    picture = full_raw_picture(frame)
    assert picture.size == (17, 12)
    np.testing.assert_array_equal(np.asarray(picture), frame[:, :, ::-1])
    np.testing.assert_array_equal(frame, before)
    with pytest.raises(ValueError, match='native uint8'):
        full_raw_picture(frame.astype(float))


def test_quote_and_context_are_separate():
    assert '27.3' not in USER_QUOTE
    assert REFERENCE_CONTEXT['approximate_reference_time_s'] == 27.3
    assert '肉眼判断' in USER_QUOTE and '明显的凸起' in USER_QUOTE
    assert REFERENCE_CONTEXT['source'].endswith('not_part_of_verbatim_quote')


def test_four_panel_contract_and_no_overwrite(tmp_path):
    picture = Image.new('RGB', (400, 300), (20, 30, 40))
    cards = [dict(source_frame=round(t * 10), time_s=t, preview=picture) for t in TIMES_S]
    assert contact_sheet(cards).size == (2000, 1750)
    with pytest.raises(ValueError, match='exact four-frame'):
        contact_sheet(cards[:-1])
    with pytest.raises(FileExistsError):
        export(output=tmp_path)
