"""Prepare a separate manual-review batch without creating labels.

Decode only the listed AVI frames. An AVI frame is accepted only when the
historical JPEG-quality-95 export can be reproduced exactly (bytes or decoded
JPEG pixels). Otherwise preserve the verified candidate JPEG as a PNG copy and
explicitly record that the lost JPEG detail has not been recovered.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json

import cv2
import numpy as np
from PIL import Image

from .centerline_review_v5 import load_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / 'vision_geometry_lab/outputs/centerline_v7/candidate_inventory/manifest.json'
DEFAULT_OUTPUT = ROOT / 'vision_geometry_lab/outputs/centerline_v7/review_batch01'
RESERVED_SESSION = 'dual_20260923_113029'


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def resolve_source_video(candidate):
    """Resolve only documented paths and confirm dual-camera frame/time metadata."""
    session, camera = candidate['source_session_id'], candidate['source_camera']
    index = candidate['source_frame_index']
    if session.startswith('dual_20260923_'):
        folder = ROOT / '现场视频/训练采集/双相机同步' / session
        metadata_path, pairs_path = folder / 'session.json', folder / 'paired_frames.csv'
        if not metadata_path.is_file() or not pairs_path.is_file():
            return None, 'missing_dual_session_metadata'
        metadata = _json(metadata_path)
        serial = metadata.get('serials', {}).get(camera)
        if not serial or metadata.get('status') != 'complete_read_only_capture' or metadata.get('errors'):
            return None, 'unverified_capture_status_or_camera'
        with pairs_path.open(encoding='utf-8-sig', newline='') as stream:
            matching = [row for row in csv.DictReader(stream)
                        if int(row[camera + '_video_frame']) == index
                        and abs(float(row[camera + '_host_time_s']) - candidate['source_time_s']) < 1e-6]
        if len(matching) != 1:
            return None, 'frame_and_time_not_confirmed_in_paired_metadata'
        path = folder / f'{camera}_{serial}.avi'
        evidence = dict(session_metadata_sha256=file_hash(metadata_path),
                        paired_frames_sha256=file_hash(pairs_path), matched_pair_id=int(matching[0]['pair_id']))
    elif session == 'R_run_stop_run_20260922_01' and camera == 'right':
        path = ROOT / '现场视频/训练采集/右侧相机/中层/R_run_stop_run_20260922_01.avi'
        evidence = dict(frame_index_source='candidate inventory and historical extracted JPEG filename')
    else:
        return None, 'no_verified_video_path_rule'
    if not path.is_file():
        return None, 'verified_video_path_missing'
    return dict(video_path=str(path.resolve()), frame_index=index, metadata_evidence=evidence), None


def decode_one_frame(video):
    cap = cv2.VideoCapture(video['video_path'])
    try:
        if not cap.isOpened():
            return None, dict(reason='cannot_open_selected_video')
        index = video['frame_index']
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if index < 0 or index >= total or not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
            return None, dict(reason='selected_frame_seek_failed', frame_count=total)
        ok, frame = cap.read()  # Exactly one read call for this selected frame.
        after = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
        evidence = dict(requested_frame_index=index, reported_next_frame=after,
                        frame_count=total, nominal_fps=float(cap.get(cv2.CAP_PROP_FPS)),
                        requested_read_calls=1)
        if not ok or frame is None or abs(after - (index + 1)) > .01:
            return None, dict(evidence, reason='read_or_reported_frame_position_mismatch')
        return frame, dict(evidence, reason='selected_frame_decoded')
    finally:
        cap.release()


def select_review_pixels(candidate_bytes, video_bgr, orientation):
    """Return RGB pixels and exact historical-export match evidence, no guessing."""
    if orientation not in ('original', 'canonical_horizontal_mirror'):
        raise ValueError('Unknown stored candidate orientation')
    with Image.open(io.BytesIO(candidate_bytes)) as image:
        fallback_rgb = np.asarray(image.convert('RGB'))
    evidence = dict(match_policy='exact quality-95 JPEG bytes or exact re-encoded JPEG decoded pixels',
                    video_match_confirmed=False, original_video_bgr_sha256=None,
                    reencoded_jpeg_sha256=None, fallback_reason='video_unavailable')
    if video_bgr is not None:
        evidence['original_video_bgr_sha256'] = array_hash(video_bgr)
        oriented = cv2.flip(video_bgr, 1) if orientation == 'canonical_horizontal_mirror' else video_bgr
        if oriented.shape == fallback_rgb.shape:
            ok, encoded = cv2.imencode('.jpg', oriented, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                encoded_bytes = encoded.tobytes()
                evidence['reencoded_jpeg_sha256'] = hashlib.sha256(encoded_bytes).hexdigest()
                same_bytes = encoded_bytes == candidate_bytes
                candidate_bgr = cv2.imdecode(np.frombuffer(candidate_bytes, np.uint8), cv2.IMREAD_COLOR)
                reproduced_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                same_pixels = candidate_bgr is not None and np.array_equal(candidate_bgr, reproduced_bgr)
                if same_bytes or same_pixels:
                    evidence.update(video_match_confirmed=True,
                                    match_basis='exact_encoded_bytes' if same_bytes else 'exact_decoded_reencoded_jpeg_pixels',
                                    fallback_reason=None)
                    return cv2.cvtColor(oriented, cv2.COLOR_BGR2RGB), 'verified_selected_avi_frame_to_png', evidence
        evidence['fallback_reason'] = 'historical_candidate_JPEG_not_exactly_reproduced'
    return fallback_rgb, 'existing_jpeg_decoded_to_png_not_lossless_recovery', evidence


def verify_existing(output, inventory_hash):
    output = Path(output)
    path = output / 'manifest.json'
    if not path.is_file():
        raise FileExistsError('Output exists without a complete manifest; refusing to erase or replace anything')
    manifest = load_manifest(path)
    if manifest.get('candidate_inventory_sha256') != inventory_hash:
        raise ValueError('Candidate inventory changed; existing review batch and labels are preserved')
    for sample in manifest['samples']:
        if file_hash(sample['image_path']) != sample['image_sha256']:
            raise ValueError('Existing review image changed; refusing to regenerate or overwrite it')
    labels = sum(Path(sample['annotation_path']).is_file() for sample in manifest['samples'])
    return dict(status='already_prepared_verified_no_writes', samples=len(manifest['samples']),
                existing_label_files_preserved=labels, manifest=str(path.resolve()), video_frames_decoded=0)


def prepare(inventory_path=DEFAULT_INVENTORY, output=DEFAULT_OUTPUT):
    inventory_path, output = Path(inventory_path).resolve(), Path(output).resolve()
    inventory_hash = file_hash(inventory_path)
    if output.exists():
        return verify_existing(output, inventory_hash)
    inventory = _json(inventory_path)
    candidates = inventory['candidates']
    if not 1 <= len(candidates) <= 8:
        raise ValueError('Expected a bounded batch of at most eight selected candidates')
    forbidden = set(inventory['excluded_current_session_ids'])
    for candidate in candidates:
        group = candidate['source_session_id']
        if group in forbidden or '20260925' in group:
            raise ValueError('Current pilot or future blind-test source is forbidden')
        if type(candidate['source_frame_index']) is not int or candidate['source_frame_index'] < 0:
            raise ValueError('A known nonnegative frame index is required')
        if file_hash(candidate['image_path']) != candidate['image_sha256']:
            raise ValueError('Candidate source image changed')
    output.mkdir(parents=True)
    (output / 'images').mkdir()
    (output / 'labels').mkdir()  # Empty directory only: no annotation JSON is created.
    samples, video_hashes, modes = [], {}, {}
    requested_reads = 0
    for candidate in candidates:
        source_bytes = Path(candidate['image_path']).read_bytes()
        video, resolution_reason = resolve_source_video(candidate)
        frame, decode_evidence = (None, dict(reason=resolution_reason))
        if video is not None:
            video_path = video['video_path']
            if video_path not in video_hashes:
                video_hashes[video_path] = file_hash(video_path)
            frame, decode_evidence = decode_one_frame(video)
            requested_reads += int(decode_evidence.get('requested_read_calls', 0))
        rgb, mode, match = select_review_pixels(source_bytes, frame, candidate['source_orientation'])
        modes[mode] = modes.get(mode, 0) + 1
        height, width = rgb.shape[:2]
        sample_id = candidate['candidate_id'] + '__' + Path(candidate['image_path']).stem
        image_path = output / 'images' / (sample_id + '.png')
        with image_path.open('xb') as stream:
            Image.fromarray(rgb, 'RGB').save(stream, format='PNG')
        group = candidate['source_session_id']
        provenance = dict(
            origin_mode=mode, candidate_id=candidate['candidate_id'],
            candidate_image_path=candidate['image_path'], candidate_image_sha256=candidate['image_sha256'],
            candidate_source_manifest_sha256=candidate['source_manifest_sha256'],
            image_orientation=candidate['source_orientation'], orientation_applied_to_avi_only=True,
            original_camera_xy_to_review_xy=('x_review=width-1-x_camera; y_review=y_camera'
                if candidate['source_orientation'] == 'canonical_horizontal_mirror' else 'identity'),
            review_rgb_pixels_sha256=array_hash(rgb), video_match=match,
            resolved_video=video, source_video_sha256=video_hashes.get(video['video_path']) if video else None,
            decode_evidence=decode_evidence,
            source_loss_note=('PNG preserves the decoded AVI pixels; AVI codec itself may be lossy'
                if match['video_match_confirmed'] else 'PNG is a decoded existing JPEG copy; lost JPEG information is not recovered'))
        samples.append(dict(
            sample_id=sample_id, group_id=group,
            split='holdout' if group == RESERVED_SESSION else 'train',
            image_path=str(image_path), image_sha256=file_hash(image_path), image_shape_hw=[height, width],
            roi_xyxy=[0, 0, width, height], roi_source='full image; review scope only, not detector truth',
            annotation_path=str(output / 'labels' / (sample_id + '.json')),
            source=dict(video_path=video['video_path'] if video else '', frame_index=candidate['source_frame_index'],
                        time_s=candidate['source_time_s'], camera=candidate['source_camera'],
                        time_basis='historical selected-frame metadata; not exposure-calibrated',
                        video_frame_identity_verified=bool(match['video_match_confirmed'])),
            provenance=provenance, legacy_visibility_hint=candidate['legacy_visibility'],
            annotation_hint_policy='legacy visibility is not new target truth; user chooses status',
            target_identity_confirmed=False, initial_annotation_created=False))
    manifest = dict(
        schema_version=1, task='seed_conditioned_new_winding_centerline',
        purpose='independent V7 batch01 human review; no training or merge performed',
        candidate_inventory_path=str(inventory_path), candidate_inventory_sha256=inventory_hash,
        samples=samples, initial_human_labels=0, training_authorized=False,
        split_semantics='proposed train and reserved development holdout by whole source session; not blind validation',
        reserved_holdout_session=RESERVED_SESSION,
        reserved_scope='all frames, both cameras and all derived data in the reserved session',
        current_pilot_modified=False, candidate_inventory_modified=False,
        future_blind_test_images_read=False, hardware_access=False, physical_control=False,
        old_annotations_copied=False, preparation_modes=modes,
        requested_video_frame_read_calls=requested_reads, whole_video_decoding=False,
        preparation_source_sha256=file_hash(__file__))
    path = output / 'manifest.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    load_manifest(path)
    if file_hash(inventory_path) != inventory_hash:
        raise ValueError('Candidate inventory changed during preparation')
    return dict(status='prepared_without_labels', samples=len(samples), manifest=str(path),
                preparation_modes=modes, video_frames_decoded=requested_reads, annotation_files_created=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    print(json.dumps(prepare(args.inventory, args.output), ensure_ascii=False))
