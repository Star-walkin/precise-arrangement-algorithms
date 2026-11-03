"""Build a small review-only inventory from existing extracted frames.

No video decode, detector/model, annotation write or training-manifest mutation.
The fixed choices are metadata-based and explicitly are not new target truth.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'vision_geometry_lab/outputs/centerline_v7/candidate_inventory'
CURRENT_MANIFEST = ROOT / 'vision_geometry_lab/outputs/centerline_v5/pilot/manifest.json'
SOURCES = {
    'middle': ROOT / 'vision_training/task_dual_midlayer_v1/train/manifest.json',
    'thick': ROOT / 'vision_training/task_dual_thicklayer_v1/train/manifest.json',
    'old_right': ROOT / 'vision_training/task_v1/manifest.json',
}
QUALITY = {
    'middle': ROOT / 'vision_training/task_dual_midlayer_v1/quality_screen.json',
    'thick': ROOT / 'vision_training/task_dual_thicklayer_v1/quality_screen.json',
}
RESERVED_SESSION = 'dual_20260923_113029'
# Chosen from metadata before viewing the sheet; never from model scores.
CHOICES = (
    ('middle', 'dual_20260923_101517__pair0050__right.jpg', 'middle-layer early visible route'),
    ('middle', 'dual_20260923_101517__pair0550__right.jpg', 'middle-layer later route near the observed edge'),
    ('middle', 'dual_20260923_101517__pair0050__left.jpg', 'legacy out-of-view candidate; verify new target semantics'),
    ('thick', 'dual_20260923_113029__t05__left.jpg', 'reserved-session early thick-layer route'),
    ('thick', 'dual_20260923_113029__t25__left.jpg', 'reserved-session later thick-layer route'),
    ('thick', 'dual_20260923_113029__t55__right.jpg', 'reserved-session uncertainty candidate; not a negative label'),
    ('old_right', 'R_run_stop_run_20260922_01__t012.033__f000368.jpg', 'different-date right-camera start event'),
    ('old_right', 'R_run_stop_run_20260922_01__t047.607__f001456.jpg', 'same-session restart event; not an independent new condition'),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def font(size):
    for path in ('C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/segoeui.ttf'):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build_inventory(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; never overwrite a reviewed candidate inventory')
    current = load_json(CURRENT_MANIFEST)
    forbidden = {s['group_id'] for s in current['samples']}
    source_data = {key: load_json(path) for key, path in SOURCES.items()}
    quality_data = {key: load_json(path) for key, path in QUALITY.items()}
    candidates = []
    for index, (source_key, image_name, reason) in enumerate(CHOICES, 1):
        manifest = source_data[source_key]
        tasks = [row for row in manifest['tasks'] if row['image_name'] == image_name]
        if len(tasks) != 1:
            raise ValueError('Expected exactly one existing task: ' + image_name)
        task = tasks[0]
        session = Path(task['clip']).stem
        if session in forbidden or '20260925' in session:
            raise ValueError('Current split or future blind-test session is prohibited: ' + session)
        image_path = Path(task['image']).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        png_sibling = image_path.with_suffix('.png')
        # A sibling PNG is not silently substituted without independent provenance.
        image_hash = digest(image_path)
        with Image.open(image_path) as image:
            image_shape = [image.height, image.width]
            image_format = image.format
        legacy = None
        if source_key in quality_data:
            records = [row for split in ('train', 'holdout')
                       for row in quality_data[source_key][split]['records']
                       if row['image_name'] == image_name]
            if len(records) != 1:
                raise ValueError('Ambiguous or missing legacy quality entry: ' + image_name)
            legacy = records[0]
        frame = task.get('source_frame')
        if frame is None:
            found = re.search(r'__f(\d+)', image_name)
            frame = int(found.group(1)) if found else None
        candidate = dict(
            candidate_id=f'C{index:02d}', source_session_id=session,
            image_path=str(image_path), image_sha256=image_hash, image_shape_hw=image_shape,
            source_image_format=image_format, same_basename_png_exists=png_sibling.is_file(),
            image_is_existing_extracted_frame=True, source_frame_index=frame,
            source_time_s=task['time_s'], source_camera=task.get('camera', 'right'),
            source_orientation=task.get('annotation_orientation', 'original'),
            source_manifest=str(SOURCES[source_key]), source_manifest_sha256=digest(SOURCES[source_key]),
            legacy_split_metadata=manifest.get('split', 'not_defined'),
            legacy_visibility=legacy['incoming_visibility'] if legacy else 'not_recorded_in_quality_screen',
            legacy_quality_flags=legacy.get('qa_flags', []) if legacy else [],
            legacy_quality_screen=str(QUALITY[source_key]) if legacy else None,
            legacy_quality_screen_sha256=digest(QUALITY[source_key]) if legacy else None,
            selection_reason=reason, selection_basis='fixed metadata choice; no V7 prediction scores',
            proposed_role=('reserved_future_session_evaluation' if session == RESERVED_SESSION
                           else 'candidate_development_pool'),
            reviewed_for_v7=False, target_identity_confirmed=False, confirmed_absent=False,
            training_allowed=False, new_annotation_created=False,
            status='awaiting_scope_and_identity_review_not_ground_truth')
        candidates.append(candidate)
    uncertain = [r for r in candidates if r['legacy_visibility'] in ('out_of_view', 'uncertain')]
    if len(candidates) > 8 or len(uncertain) > 2:
        raise ValueError('Small candidate/uncertainty budget exceeded')
    if len({r['image_sha256'] for r in candidates}) != len(candidates):
        raise ValueError('Duplicate source frames')
    output.mkdir(parents=True)
    width, card_height, header = 1720, 730, 160
    sheet = Image.new('RGB', (width, header + card_height * 4 + 45), '#101722')
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 16), 'V7 | 8 REVIEW CANDIDATES - NOT NEW GROUND TRUTH', font=font(30), fill='white')
    draw.text((20, 60), 'No new labels, model outputs, target curves or masks are overlaid. Full existing frames only.',
              font=font(21), fill='#b8c6d8')
    draw.text((20, 93), 'Purple: whole session reserved for future evaluation (historically viewed, NOT blind).',
              font=font(21), fill='#dab5ff')
    draw.text((20, 124), 'Legacy out_of_view / uncertain are review leads only, NOT confirmed V7 absent labels.',
              font=font(20), fill='#ffce78')
    for index, row in enumerate(candidates):
        left, top = 10 + (index % 2) * 860, header + (index // 2) * card_height
        reserved = row['source_session_id'] == RESERVED_SESSION
        accent = '#bb83ee' if reserved else '#55b9cb'
        draw.rounded_rectangle((left, top, left + 840, top + 715), radius=8, fill='#202b3a', outline=accent, width=3)
        name = Path(row['image_path']).stem
        draw.text((left + 12, top + 10), f"{row['candidate_id']}  {name}", font=font(21), fill='white')
        role = 'RESERVE WHOLE SESSION - NO TRAINING' if reserved else 'REVIEW POOL - NO TRAINING AUTHORIZED'
        draw.text((left + 12, top + 42), role, font=font(18), fill=accent)
        visible = row['legacy_visibility'].replace('not_recorded_in_quality_screen', 'not recorded')
        draw.text((left + 12, top + 70), f"Legacy visibility: {visible} | t={row['source_time_s']:.3f}s",
                  font=font(18), fill='#ffce78' if visible in ('out_of_view', 'uncertain') else '#d2dae5')
        draw.text((left + 12, top + 96), f"Stored orientation: {row['source_orientation']} | Source: {row['source_image_format']}",
                  font=font(18), fill='#d2dae5')
        with Image.open(row['image_path']) as source:
            preview = source.convert('RGB')
            preview.thumbnail((816, 570), Image.Resampling.LANCZOS)
            sheet.paste(preview, (left + 12 + (816-preview.width)//2, top + 130 + (570-preview.height)//2))
        if digest(row['image_path']) != row['image_sha256']:
            raise ValueError('Source changed while making preview')
    draw.text((20, sheet.height - 32), 'Preview resized only; source files remain untouched. 2026-09-25 blind-test images were not read.',
              font=font(19), fill='#b8c6d8')
    contact = output / 'contact_sheet.png'
    sheet.save(contact)
    metadata = dict(
        schema_version=1, artifact_type='review_candidate_inventory_not_training_manifest',
        created_local_date='2026-09-27', selection_policy='metadata-first fixed 8; no model-score selection',
        current_pilot_manifest=str(CURRENT_MANIFEST), current_pilot_manifest_sha256=digest(CURRENT_MANIFEST),
        excluded_current_session_ids=sorted(forbidden), prohibited_blind_test_date='2026-09-25',
        future_blind_test_images_read=False, video_decoded=False, hardware_access=False,
        source_images_modified=False, source_labels_modified=False, training_started=False,
        session_reservations=[dict(session_id=RESERVED_SESSION, role='reserved_future_evaluation',
            covers='all frames, both cameras, all derived masks/pseudo-labels from this session',
            not_used_in_current_v5_v6_training=True, legacy_role_overridden='old train is NOT the new split',
            pristine_blind_test=False, reason='historically reviewed; this preview also reveals appearance')],
        training_injection_permitted=False, candidates=candidates,
        contact_sheet=str(contact), contact_sheet_sha256=digest(contact),
        generator=str(Path(__file__).resolve()), generator_sha256=digest(__file__),
        limitations=[
            'Existing JPEGs are reused as-is; this is not lossless video-frame extraction.',
            'Canonical horizontal mirrors are explicitly identified; coordinates are not native camera coordinates.',
            'Legacy mask/visibility semantics do not confirm the new incoming-winding identity or absence.',
            'One reserved historical session is not a new blind-field acceptance test.',
            'No candidate is approved for training or has new target annotations.'
        ])
    with (output / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(manifest=str(output/'manifest.json'), contact_sheet=str(contact), candidates=len(candidates)), ensure_ascii=False))
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    build_inventory(parser.parse_args().output)
