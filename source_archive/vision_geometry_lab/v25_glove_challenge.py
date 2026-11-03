"""Fixed real-glove challenge for V25 local continuation; no accuracy claim.

The existing V16 automatic seed/width and byte-bound video frame168 are reused.
Both V25 arms must finish before inference. There is no occlusion GT mask, and
neither empty output nor stopping before the glove is scored as correct rejection.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
BASE = LAB/'outputs/v25_local_continuation/trial_001'
V16 = LAB/'outputs/v16_event_feasibility'
META = V16/'extract/right/frames/f000168.json'
ARMS = ('high', 'low')


def safe(path):
    path = Path(path)
    if '20260925' in str(path):
        raise ValueError('Reserved blind sources are forbidden')
    return path


def read(path):
    return json.loads(safe(path).read_text('utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with safe(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def source_contract(meta):
    if meta['session'] != 'right' or meta['frame_index'] != 168 or meta['time_s'] != 16.8 or meta['fps'] != 10.:
        raise ValueError('Only preselected right frame168/16.8s is permitted')
    seed = np.asarray(meta['free_seed'], dtype=float)
    width = float(meta['geometry']['width_px'])
    if seed.ndim != 2 or seed.shape[1] != 2 or len(seed) < 2 or not np.isfinite(seed).all():
        raise ValueError('Invalid archived automatic seed')
    if not np.isfinite(width) or width <= 0 or not meta['available']:
        raise ValueError('Valid archived automatic width required')
    if not np.array_equal(seed, np.asarray(meta['geometry']['free_seed_xy'], float)):
        raise ValueError('Archived seed differs across geometry and source')
    if meta['geometry']['target_or_quality_used'] is not False:
        raise ValueError('Automatic geometry cannot contain target conditioning')
    return dict(frame_index=168, time_s=16.8, fps=10., frame_shape=meta['frame_shape'],
        frame_bgr_array_sha256=meta['frame_sha256'], past_seed_xy=seed.tolist(), width_px=width,
        seed_reason=meta['seed_reason'], seed_is_physical_contact=False,
        source='unchanged V16 current-frame automatic free span and optical width; no remeasurement')


def verify_raw_frame(frame_bgr, source):
    from .v11_path_contract import array_sha256
    if list(frame_bgr.shape) != source['frame_shape'] or frame_bgr.dtype != np.uint8:
        raise ValueError('Decoded source image shape/dtype changed')
    if array_sha256(frame_bgr) != source['frame_bgr_array_sha256']:
        raise ValueError('Decoded frame does not match archived V16 bytes')


def summarize_rollout(result, seed, width):
    """Only geometry/output availability. No mask or occlusion-success score."""
    origin = np.asarray(seed[-1], float)
    direction = origin-np.asarray(seed[0], float)
    direction /= np.linalg.norm(direction)
    segments = [np.asarray(s, float) for s in result['segments'] if len(s) >= 2]
    points = np.concatenate(segments) if segments else np.empty((0, 2))
    return dict(nonempty=bool(segments), accepted_segment_count=len(segments),
        attempted_steps=len(result['steps']), accepted_steps=sum(bool(s['accepted']) for s in result['steps']),
        predicted_total_arclength_px=float(sum(np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in segments)),
        furthest_seed_forward_projection_widths=float(np.max((points-origin)@direction)/width) if len(points) else None,
        final_xy=points[-1].tolist() if len(points) else None,
        image_bbox_xyxy=[*points.min(0).tolist(), *points.max(0).tolist()] if len(points) else None,
        stop_reason=result['reason'], stopped_is_correct_occlusion_rejection=None,
        glove_overlap_measured=False, occlusion_accuracy=None,
        visibility_confirmed=False, identity_confirmed=False,
        warning='Nonempty and extent only. Empty/short output is not evidence of recognizing a glove or retaining visible rope.')


def freeze(base):
    from .v25_local_experiment import verify as verify_main
    from .v11_path_contract import array_sha256
    import cv2
    from PIL import Image
    main = verify_main(base)
    out = base/'qualitative_challenge'
    if out.exists():
        raise FileExistsError('Never overwrite a challenge protocol')
    source = source_contract(read(META))
    old_protocol = read(V16/'protocol.json')
    old_index = read(V16/'extract/right/index.json')
    old_row = next(r for r in old_index['records'] if r['frame_index'] == 168)
    if old_row['metadata_sha256'] != sha(META) or old_index['protocol_sha256'] != sha(V16/'protocol.json'):
        raise ValueError('Historical V16 frame binding changed')
    video = safe(old_protocol['videos']['right']['path'])
    expected_video = old_protocol['videos']['right']['sha256']
    if sha(video) != expected_video or old_index['video_sha256'] != expected_video:
        raise ValueError('Video hash differs from original V16 source')
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-10.) > 1e-9:
            raise ValueError('Cannot open expected original video')
        cap.set(cv2.CAP_PROP_POS_FRAMES, 168)
        ok, frame = cap.read()
        if not ok:
            raise ValueError('Could not decode original frame168')
    finally:
        cap.release()
    verify_raw_frame(frame, source)
    out.mkdir(parents=True)
    image_path = out/'source.png'
    Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1])).save(image_path)
    sources = {Path(__file__), META, V16/'protocol.json', V16/'extract/right/index.json',
               base/'protocol.json', LAB/'v25_local_experiment.py', LAB/'v25_local_contract.py',
               LAB/'v25_local_data.py', LAB/'v11_path_model.py', LAB/'v11_path_contract.py'}
    write(out/'protocol.json', dict(schema='v25_fixed_real_glove_challenge_1',
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        main_protocol_sha256=sha(base/'protocol.json'), source=source,
        source_png_sha256=sha(image_path), source_video_path=str(video), source_video_sha256=expected_video,
        source_video_hash_verified_at_freeze=True, source_frame_hash_verified=True,
        source_rgb_array_sha256=array_sha256(np.ascontiguousarray(frame[:, :, ::-1])),
        arms=list(ARMS), input_mode='each model evaluated using its matching high/low mode',
        rollout_maximum_steps=main['rollout_steps'],
        inference='exact frozen V25 model_predictor + rollout, own predicted prefix only; no target or new prompting',
        model_selection='both main final600step checkpoints; never choose best arm from challenge',
        missing_model='refuse inference until both final checkpoints/reports exist',
        qualitative_only=True, trusted_occlusion_mask_available=False,
        no_occlusion_accuracy=True, no_early_stop_success=True,
        no_identity_or_physical_event_confirmation=True, hardware_access=False,
        stop='one source/frame/seed/width, no threshold tuning, no rerun or deployment',
        source_hashes={str(p.resolve()): sha(p) for p in sorted(sources, key=str)}))
    return out


def verify(base):
    from .v25_local_experiment import verify as verify_main
    verify_main(base)
    out = base/'qualitative_challenge'
    p = read(out/'protocol.json')
    if p['main_protocol_sha256'] != sha(base/'protocol.json') or p['source_png_sha256'] != sha(out/'source.png'):
        raise ValueError('Challenge binding changed')
    for path, expected in p['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Frozen challenge source changed: '+path)
    return p


def render_comparison(source, p, results):
    from PIL import Image, ImageDraw, ImageOps
    # Avoid importing CV-only display helpers in the Torch runtime.
    from PIL import ImageFont
    def font(size):
        return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size)
    im = Image.new('RGB', (2460, 1000), '#15212d'); d = ImageDraw.Draw(im)
    d.text((24, 16), 'V25 真实手套遮挡挑战 | 右相机原始第168帧 / 16.8s | 仅定性', font=font(34), fill='white')
    d.text((24, 73), '青：冻结自动自由绳种子；品红：接受的延续候选。没有手套真值掩膜；提前停止不等于识别了遮挡。', font=font(26), fill='#ffd68a')
    for col, arm in enumerate(('source', *ARMS)):
        preview = ImageOps.contain(source, (800, 610)); scale = preview.width/source.width
        paint = ImageDraw.Draw(preview)
        paint.line([(x*scale, y*scale) for x, y in p['source']['past_seed_xy']], fill='#28edff', width=3)
        if arm != 'source':
            for segment in results[arm]['rollout']['segments']:
                if len(segment) >= 2:
                    paint.line([(float(x)*scale, float(y)*scale) for x, y in segment], fill='#ff3cde', width=4)
        x = 20+col*820
        title = '原始画面 + 同一个自动提示' if arm == 'source' else arm+' 模型 / '+arm+' 输入'
        d.text((x, 128), title, font=font(27), fill='white'); im.paste(preview, (x, 174))
        if arm != 'source':
            s = results[arm]['summary']
            d.text((x, 800), f"接受 {s['accepted_steps']} / 尝试 {s['attempted_steps']} 步 | 总路径 {s['predicted_total_arclength_px']:.1f}px", font=font(23), fill='#cfdfef')
            d.text((x, 842), '停止原因：'+s['stop_reason'], font=font(17), fill='#cfdfef')
        else:
            d.text((x, 800), '全幅等比显示；原图另存 source.png', font=font(24), fill='#cfdfef')
            d.text((x, 842), '未提供人工路径、接触点或手套区域作为输入', font=font(23), fill='#cfdfef')
    d.text((24, 931), '是否走入手套需原图人工复核；几何连续、非空输出、早停都不能单独证明同根身份或可见性。', font=font(27), fill='#ffd68a')
    return im


def infer(base):
    from PIL import Image
    import torch
    from .v11_path_contract import array_sha256
    from .v25_local_experiment import model_predictor, rollout
    p = verify(base); out = base/'qualitative_challenge'
    if any((out/name).exists() for name in ('models.json', 'results.json', 'comparison.png')):
        raise FileExistsError('This fixed challenge may only be inferred once')
    models = {}
    for arm in ARMS:
        checkpoint_path = base/'models'/arm/'checkpoint.pt'
        report_path = checkpoint_path.with_name('report.json')
        report = read(report_path)
        if report['arm'] != arm or report['steps'] != 600 or report['protocol_sha256'] != sha(base/'protocol.json') or report['checkpoint_sha256'] != sha(checkpoint_path):
            raise ValueError('Both exact final model reports required before inference')
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if ck['arm'] != arm or ck['steps'] != 600 or ck['protocol_sha256'] != sha(base/'protocol.json'):
            raise ValueError('Final checkpoint metadata mismatch')
        models[arm] = dict(path=str(checkpoint_path), sha256=sha(checkpoint_path), report_sha256=sha(report_path))
    write(out/'models.json', dict(models=models, protocol_sha256=sha(out/'protocol.json'), selection='both final checkpoints; bound before challenge inference'))
    with Image.open(out/'source.png') as image:
        source = image.convert('RGB'); rgb = np.asarray(source)
    if array_sha256(rgb) != p['source_rgb_array_sha256']:
        raise ValueError('Decoded PNG no longer matches frozen raw RGB')
    results = {}
    for arm in ARMS:
        predict, path = model_predictor(base, arm)
        r = rollout(rgb, p['source']['past_seed_xy'], p['source']['width_px'], predict, arm,
                    maximum_steps=p['rollout_maximum_steps'])
        results[arm] = dict(rollout=r, summary=summarize_rollout(r, p['source']['past_seed_xy'], p['source']['width_px']),
                           model_sha256=models[arm]['sha256'])
    render_comparison(source, p, results).save(out/'comparison.png')
    verify(base)
    for arm in ARMS:
        if sha(models[arm]['path']) != models[arm]['sha256']:
            raise ValueError('Model changed during challenge')
    write(out/'results.json', dict(schema='v25_real_glove_qualitative_results_1', results=results,
        protocol_sha256=sha(out/'protocol.json'), models_sha256=sha(out/'models.json'),
        source_png_sha256=sha(out/'source.png'), comparison_sha256=sha(out/'comparison.png'),
        trusted_occlusion_mask_available=False, occlusion_accuracy=None,
        no_early_stop_success=True, manual_review_required=True, frame_count=1,
        identity_confirmed=False, physical_event_validated=False, hardware_access=False))
    return {arm: result['summary'] for arm, result in results.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'verify', 'infer'))
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args()
    result = globals()[args.stage](args.base.resolve())
    if args.stage == 'freeze':
        result = dict(output=str(result), protocol_sha256=sha(result/'protocol.json'))
    elif args.stage == 'verify':
        result = dict(verified=True, protocol_sha256=sha(args.base/'qualitative_challenge/protocol.json'))
    print(json.dumps(dict(stage=args.stage, result=result), ensure_ascii=False))


if __name__ == '__main__':
    main()
