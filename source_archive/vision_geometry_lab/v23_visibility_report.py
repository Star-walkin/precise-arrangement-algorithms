"""Posthoc V23 artifact inspection only; no training, tuning, or hardware."""
from pathlib import Path
import argparse
import hashlib
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw

from .v23_visibility_trial import BASE, INVENTORY, CHALLENGE, read, sha, write, verify


def panel(rgb, distribution, observable, caption, threshold=None):
    image=Image.fromarray(np.clip(rgb.transpose(1,2,0)*255,0,255).astype(np.uint8)).resize((1024,512))
    canvas=Image.new('RGB',(1024,568),'#eeeeee');canvas.paste(image,(0,32))
    draw=ImageDraw.Draw(canvas);draw.text((12,8),caption,fill='black')
    row=distribution.argmax(axis=0)
    for col in range(1,512):
        if threshold is None or (observable[col-1]>=threshold and observable[col]>=threshold):
            draw.line((2*(col-1),32+2*row[col-1],2*col,32+2*row[col]),fill='#ff39df',width=2)
        draw.line((2*col,553,2*col,565),fill=('#289d41' if observable[col]>=.5 else '#cc3333'),width=2)
    return canvas


def run(base):
    p=verify(base);out=base/'inspection'
    if out.exists():raise FileExistsError('No overwrite')
    report=read(base/'evaluation/report.json')
    if report['status']!='completed':raise ValueError('No completed evaluation')
    out.mkdir(parents=True)
    artifacts=[]
    challenge=report['challenges'][0]
    with np.load(challenge['prediction_path'],allow_pickle=False) as a:
        before=panel(a['input_rgb'],a['probability'],a['observable'],
            'Original V11 conditional argmax (not confirmed rope identity)')
        after=panel(a['input_rgb'],a['probability'],a['observable'],
            'V23 observable >= 0.5; green = retained. This glove challenge is NOT rejected.',.5)
    joined=Image.new('RGB',(1024,1136));joined.paste(before,(0,0));joined.paste(after,(0,568))
    image_path=out/'right_16_8s_strip_before_after.png';joined.save(image_path)
    artifacts.append(dict(path=str(image_path),sha256=sha(image_path)))
    # Original-frame decoder paths, byte-verified against previously archived raw frame.
    meta=read(CHALLENGE)
    old_protocol=read(CHALLENGE.parents[3]/'protocol.json')
    video=old_protocol['videos']['right']['path']
    cap=cv2.VideoCapture(video);cap.set(cv2.CAP_PROP_POS_FRAMES,168);ok,frame=cap.read();cap.release()
    actual=hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() if ok else None
    expected=meta['frame_sha256']
    # V11 array SHA includes shape/dtype metadata; use the same archived contract.
    from .v11_path_contract import array_sha256
    matched=ok and array_sha256(frame)==expected
    if not matched:raise ValueError('Frame binding failed; no unbound image is rendered')
    for name,key in (('before','before_segments'),('after','after_segments')):
        drawn=frame.copy()
        for segment in challenge[key]:
            points=np.round(segment).astype(np.int32)
            if len(points)>=2:cv2.polylines(drawn,[points],False,(255,40,255),5)
        rendered=Image.fromarray(cv2.cvtColor(drawn,cv2.COLOR_BGR2RGB)).resize((1024,750))
        d=ImageDraw.Draw(rendered);d.rectangle((0,0,1024,30),fill='black')
        d.text((10,8),f'{name.upper()} - right source frame168 / 16.8s; candidate only, identity unconfirmed',fill='white')
        path=out/f'right_16_8s_{name}_original.png';rendered.save(path)
        artifacts.append(dict(path=str(path),sha256=sha(path),frame_sha256=expected))
    # Record the rejected C06 input explicitly; never substitute a guessed width.
    from .evaluate_centerline_multiscale_v9 import independent_width
    from .centerline_review_v5 import read_clean_image
    item=next(s for s in read(INVENTORY)['samples'] if s['candidate_id']=='C06')
    original=read(item['source_manifest_path'])
    sample=next(s for s in original['samples'] if s['sample_id']==item['sample_id'])
    label=read(sample['annotation_path']);rgb=np.asarray(read_clean_image(sample))
    width=independent_width(np.ascontiguousarray(rgb[:,:,::-1]),label['polylines']['free_seed'])
    result=dict(schema='v23_posthoc_inspection_1',report_sha256=sha(base/'evaluation/report.json'),
        protocol_sha256=sha(base/'protocol.json'),script_sha256=sha(__file__),artifacts=artifacts,
        C06=dict(sample_id=item['sample_id'],width_result=width,
            model_input_available=width['width_px'] is not None,model_evaluation_performed=False,
            reason='No independently measured width, not counted as correct rejection by the new head'),
        conclusion='Mechanism pass does not transfer to real glove rejection. No tuning/deployment.',
        geometry_warning='Column5px metric is seed-normal localization, not rope identity or old arc-length accuracy',
        source_video_path=video,frame_index=168,frame_hash_verified=bool(matched),
        hardware_access=False,identity_confirmed=False)
    write(out/'audit.json',result);verify(base)
    print(json.dumps(dict(inspection=str(out),C06=width,artifacts=len(artifacts)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,default=BASE)
    args=parser.parse_args();run(args.base.resolve())
