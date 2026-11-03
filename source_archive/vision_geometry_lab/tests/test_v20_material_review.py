import copy

import numpy as np
import pytest
from PIL import Image

from vision_geometry_lab.v18_front_diagnostics import pixels_sha
from vision_geometry_lab.v20_material_review import (
    UNIFORM_TIMES_S, base_manifest, contact_sheet, decode_bound, raw_images,
    save_new_image, schedule_records,
)


def test_fixed_schedule_exact_and_missing_not_substituted():
    index = dict(video=dict(fps=10.), records=[dict(frame_index=i * 50, time_s=i * 5.) for i in range(13)])
    assert UNIFORM_TIMES_S == tuple(range(0, 61, 5))
    assert [r['frame_index'] for r in schedule_records(index)] == list(range(0, 601, 50))
    bad = copy.deepcopy(index)
    bad['records'].pop(3)
    with pytest.raises(ValueError, match='missing bound'):
        schedule_records(bad)


def test_no_requested_time_rounding_or_false_record_time():
    index = dict(video=dict(fps=3.), records=[dict(frame_index=0, time_s=0.)])
    with pytest.raises(ValueError, match='exact source frame'):
        schedule_records(index, (.1,))
    index['records'][0]['time_s'] = .1
    with pytest.raises(ValueError, match='record time'):
        schedule_records(index, (0,))


def test_raw_crop_keeps_native_pixels_and_original_orientation():
    frame = np.zeros((6, 8, 3), np.uint8)
    frame[:, :, 0] = np.arange(8)[None, :]
    frame[:, :, 2] = np.arange(6)[:, None]
    full, crop = raw_images(frame, (1, 2, 7, 5))
    np.testing.assert_array_equal(np.asarray(full), frame[:, :, ::-1])
    np.testing.assert_array_equal(np.asarray(crop), frame[2:5, 1:7, ::-1])
    assert crop.size == (6, 3)
    with pytest.raises(ValueError, match='ROI outside'):
        raw_images(frame, (-1, 2, 7, 5))


class FakeCapture:
    def __init__(self, frame, actual):
        self.frame, self.actual = frame, actual
        self.requested = None

    def set(self, _prop, value):
        self.requested = value

    def read(self):
        return True, self.frame.copy()

    def get(self, _prop):
        return self.actual + 1


def test_decoded_index_and_pixel_hash_not_requested_index():
    frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    record = dict(frame_index=50, time_s=5., frame_sha256=pixels_sha(frame))
    video = dict(width=8, height=6)
    cap = FakeCapture(frame, 50)
    np.testing.assert_array_equal(decode_bound(cap, record, video), frame)
    assert cap.requested == 50
    with pytest.raises(ValueError, match='decoded source index'):
        decode_bound(FakeCapture(frame, 49), record, video)
    different = frame.copy()
    different[0, 0] = 255
    with pytest.raises(ValueError, match='frame hash'):
        decode_bound(FakeCapture(different, 50), record, video)


def test_artifact_no_overwrite(tmp_path):
    picture = Image.new('RGB', (5, 7), 'red')
    result = save_new_image(picture, tmp_path / 'raw.png', tmp_path)
    assert result['size'] == [5, 7]
    with pytest.raises(FileExistsError):
        save_new_image(picture, tmp_path / 'raw.png', tmp_path)


def test_review_contract_and_contact_sheet_do_not_mutate_raw_pixels():
    manifest = base_manifest()
    for key in ('ground_truth_files_opened', 'enhancement', 'mirrored', 'algorithm_overlays',
                'model_inference', 'event_output', 'hardware_access', 'main_ui_modified'):
        assert manifest[key] is False
    image = Image.new('RGB', (180, 120), (15, 26, 37))
    original = np.array(image)
    sheet = contact_sheet([dict(source_frame=50, time_s=5., full=image, crop=image)], 'left', 1, 4)
    assert sheet.size == (2000, 1400)
    np.testing.assert_array_equal(np.asarray(image), original)
