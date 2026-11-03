import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import cv2
import numpy as np

from vision_geometry_lab.prepare_centerline_review_batch_v7 import prepare, select_review_pixels


def _jpeg(bgr):
    ok, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return encoded.tobytes()


def _fixture(folder, session='fixture_session'):
    folder = Path(folder)
    image = folder / 'frame.jpg'
    pixels = np.random.default_rng(37).integers(0, 255, (24, 32, 3), dtype=np.uint8)
    image.write_bytes(_jpeg(pixels))
    row = dict(candidate_id='C01', source_session_id=session, image_path=str(image.resolve()),
               image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(), source_frame_index=7,
               source_time_s=.7, source_camera='right', source_orientation='original',
               source_manifest_sha256='a'*64, legacy_visibility='uncertain')
    path = folder / 'inventory.json'
    path.write_text(json.dumps(dict(candidates=[row], excluded_current_session_ids=['forbidden'])), 'utf-8')
    return path, pixels


def test_v7_exact_video_frame_reproduction_and_mirror_are_verified():
    bgr = np.random.default_rng(4).integers(0, 255, (24, 32, 3), dtype=np.uint8)
    for orientation, oriented in [('original', bgr), ('canonical_horizontal_mirror', cv2.flip(bgr, 1))]:
        rgb, mode, evidence = select_review_pixels(_jpeg(oriented), bgr, orientation)
        assert mode == 'verified_selected_avi_frame_to_png' and evidence['video_match_confirmed']
        assert np.array_equal(rgb, cv2.cvtColor(oriented, cv2.COLOR_BGR2RGB))


def test_v7_nonmatching_or_unavailable_video_falls_back_honestly():
    original = np.full((24, 32, 3), 90, np.uint8)
    for wrong in (None, np.full_like(original, 100)):
        rgb, mode, evidence = select_review_pixels(_jpeg(original), wrong, 'original')
        assert mode == 'existing_jpeg_decoded_to_png_not_lossless_recovery'
        assert not evidence['video_match_confirmed'] and np.all(rgb == 90)


def test_v7_repeat_preserves_images_manifest_and_any_human_label_without_decode():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); inventory, _ = _fixture(root); output = root / 'review'
        first = prepare(inventory, output)
        assert first['annotation_files_created'] == 0
        manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
        sample = manifest['samples'][0]
        label = Path(sample['annotation_path']); label.write_bytes(b'user-owned annotation sentinel')
        before = {p: p.read_bytes() for p in (Path(sample['image_path']), output/'manifest.json', label)}
        with patch('vision_geometry_lab.prepare_centerline_review_batch_v7.decode_one_frame', side_effect=AssertionError('must not decode on rerun')):
            repeated = prepare(inventory, output)
        assert repeated['status'] == 'already_prepared_verified_no_writes'
        assert repeated['existing_label_files_preserved'] == 1
        assert all(p.read_bytes() == content for p, content in before.items())
        assert sample['roi_xyxy'] == [0, 0, 32, 24] and sample['split'] == 'train'


def test_v7_reserved_session_is_holdout_and_status_is_not_preselected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); inventory, _ = _fixture(root, 'dual_20260923_113029')
        with patch('vision_geometry_lab.prepare_centerline_review_batch_v7.resolve_source_video', return_value=(None, 'fixture_no_video')):
            prepare(inventory, root/'review')
        m = json.loads((root/'review/manifest.json').read_text('utf-8'))
        assert m['samples'][0]['split'] == 'holdout'
        assert not Path(m['samples'][0]['annotation_path']).exists()
        assert not list((root/'review/labels').iterdir())
        assert 'status' not in m['samples'][0] and not m['training_authorized']


def test_v7_current_or_blind_sources_and_existing_image_changes_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for session in ('forbidden', 'dual_20260925_232941'):
            inventory, _ = _fixture(root, session)
            try: prepare(inventory, root/session)
            except ValueError: pass
            else: raise AssertionError('Forbidden source accepted')
            assert not (root/session).exists()
        inventory, _ = _fixture(root); prepare(inventory, root/'review')
        m = json.loads((root/'review/manifest.json').read_text('utf-8'))
        image = Path(m['samples'][0]['image_path']); image.write_bytes(b'user-edited image sentinel')
        try: prepare(inventory, root/'review')
        except ValueError: pass
        else: raise AssertionError('Changed image silently replaced')
        assert image.read_bytes() == b'user-edited image sentinel'
