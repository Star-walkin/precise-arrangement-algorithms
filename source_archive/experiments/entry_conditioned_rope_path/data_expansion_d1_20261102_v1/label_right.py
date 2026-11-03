"""Native-pixel, raw-first weak-label review and bounded original-gray refinement."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from annotation_refine import coarse_axis,axis_xy,refine_segment,runs
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DEST=HERE/'labels_right';DEST.mkdir(exist_ok=True)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=load(HERE/'protocol.json')
JOBS=[j for j in P['candidate_jobs'] if not j['id'].startswith(('L_run_','L_run_stop_'))]
def prepare():
    rows=[]
    for n,j in enumerate(JOBS,1):
        assert sha(j['image_path'])==j['image_sha256']
        src=Image.open(j['image_path']).convert('RGB');im=src.copy();im.thumbnail((1280,938))
        canvas=Image.new('RGB',(1280,1000),'white');canvas.paste(im,(0,60));d=ImageDraw.Draw(canvas)
        f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);d.text((10,10),str(n)+' '+j['id'],font=f,fill='black')
        canvas.save(DEST/f'{n:02d}_raw_context.png')
        box=[1000,1750,2800,2800] if j['camera_role']=='right' else [750,2050,2450,2750]
        src.crop(box).save(DEST/f'{n:02d}_raw_native.png')
        rows.append(dict(n=n,id=j['id'],original_path=j['image_path'],context_preview_scale=1280/4096,native_crop_xyxy=box,
            raw_context_path=str(DEST/f'{n:02d}_raw_context.png'),native_crop_path=str(DEST/f'{n:02d}_raw_native.png'),seed_xy=j['seeds'][0]['xy']))
    dump(DEST/'review_inputs.json',rows)
RANGES={
 1:[(3090,2700,1),(2700,2350,2),(2350,1780,3)],
 2:[(3060,2650,1),(2650,2300,2),(2300,1770,3)],
 3:[(3050,2700,1),(2700,2350,2),(2350,1880,3)],
 4:[(3040,2600,1),(2600,2300,2),(2300,2000,3)],
 5:[(3040,2650,1),(2650,2300,2),(2300,2080,3)],
 6:[(3050,2650,1),(2650,2380,2),(2380,2150,3)],
 7:[(3940,3650,1),(3050,2700,1),(2700,2350,2),(2350,1650,3)],
 8:[(3050,2600,1),(2600,2300,2),(2300,1900,3)],
 9:[(3900,2900,1),(2900,2450,2),(2450,1950,3)],
 10:[(3900,2900,1),(2900,2500,2),(2500,1750,3)],
 11:[(180,950,1),(950,1350,2),(1350,2050,3)],
 12:[(180,950,1),(950,1350,2),(1350,2100,3)]}
def make_labels():
    review=load(DEST/'review_inputs.json');summary=[]
    for n,j in enumerate(JOBS,1):
        bgr=cv2.imdecode(np.fromfile(j['image_path'],np.uint8),cv2.IMREAD_COLOR)
        ax=coarse_axis(j);width=ax['width'];segments=[];point_details=[]
        im=Image.fromarray(bgr[:,:,::-1]);overlay=im.copy();draw=ImageDraw.Draw(overlay)
        for si,(a,b,stage) in enumerate(RANGES[n]):
            anchor=axis_xy(ax,[a,b]);p,valid,diag,pd=refine_segment(bgr,anchor,width,max_shift=4.,minimum_strength=.8)
            good=runs(valid,min_points=8);accepted=np.zeros(len(p),bool)
            for aa,bb in good:
                accepted[aa:bb]=True;points=p[aa:bb]
                segments.append(dict(status='accepted_weak',points_xy=points.tolist(),stage=stage,valid_radius_px=6.,
                    reason='Raw-first full context and native crops show entry-linked steel body; bounded opposing-edge center fit within visually selected finite corridor. Automatic weak, not human identity truth.',
                    initial_anchors_xy=anchor.tolist(),refinement=diag,
                    phase_meaning={1:'free visible outside spool overlap',2:'approach/overlap transition visual zone',3:'visible target projected over spool background; physical contact unconfirmed'}[stage],
                    local_valid_tube_is_not_body_mask=True,physical_contact_confirmed=False))
                draw.line([tuple(q) for q in points],fill=(255,30,30),width=2)
            bad=runs(~accepted,min_points=2)
            for aa,bb in bad:
                segments.append(dict(status='uncertain',points_xy=p[aa:bb].tolist(),stage=stage,valid_radius_px=0.,
                    reason='Opposing edges weak or narrow accurate center cannot be supported; excluded from narrow sigma3 supervision.',initial_anchors_xy=anchor.tolist()))
            draw.ellipse((anchor[0,0]-6,anchor[0,1]-6,anchor[0,0]+6,anchor[0,1]+6),outline='yellow',width=2)
            point_details.append(dict(segment_index=si,initial_anchors_xy=anchor.tolist(),all_refined_points_xy=p.tolist(),valid=accepted.tolist(),refinement=diag,details=pd))
        end=RANGES[n][-1][1];far=end+(-250 if j['camera_role']=='right' else 250)
        unknown=axis_xy(ax,[end,far]);segments.append(dict(status='uncertain',points_xy=unknown.tolist(),stage=3,valid_radius_px=0.,
            reason='Target merges with neighboring historical coil texture; continuation identity or bend cannot be resolved reliably from this single view. No extrapolated training target.'))
        if n==7:
            segments.append(dict(status='uncertain',points_xy=axis_xy(ax,[3650,3050]).tolist(),stage=1,valid_radius_px=0.,
                reason='Tool/guide interrupts original view; matching straight direction does not constitute observed center under occlusion.'))
        seed=np.asarray(j['seeds'][0]['xy']);draw.line([tuple(q) for q in seed],fill=(40,200,255),width=3)
        op=DEST/f'{n:02d}_overlay.png';overlay.save(op)
        cp=DEST/f'{n:02d}_overlay_native.png';box=review[n-1]['native_crop_xyxy'];overlay.crop(box).save(cp)
        preview=overlay.copy();preview.thumbnail((1280,938));preview.save(DEST/f'{n:02d}_overlay_context.png')
        result=dict(id=j['id'],image_path=j['image_path'],image_sha256=j['image_sha256'],source_group=j['source_group'],
            seed_xy=j['seeds'][0]['xy'],seed_independent_of_new_labels=True,seed_source=j['seeds'][0]['source'],camera_role=j['camera_role'],
            segments=segments,neighbor_segments=[],neighbor_status='Not separately accepted: exact competitive neighboring center identity remains unverified; unlabelled surface is ignore.',
            review=dict(original_viewed=True,native_original_viewed=True,native_overlay_viewed=False,
                raw_context_path=review[n-1]['raw_context_path'],native_original_path=review[n-1]['native_crop_path'],overlay_path=str(op),native_overlay_path=str(cp),
                crop_xyxy=box,local_display_scale=1.,source='Codex multimodal raw-image consistency review; no independent human validation',
                initial_identity_basis='Frozen short entrance plus continuously visible steel-body trajectory in raw context; old network heat not consulted for label admission',
                status='pending_final_overlay_view'),
            selection_reason='Pre-frozen temporal quantile; diverse activity position and overlap with bright/diagonal coil texture',
            coarse_proposal={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in ax.items()},
            automatic_weak_labels=True,independent_human_truth=False,old_network_heat_used=False,raw_point_diagnostics_path=str(DEST/f'{n:02d}_diagnostics.json'))
        dump(DEST/f'{n:02d}_diagnostics.json',point_details);dump(DEST/(j['id']+'.json'),result)
        summary.append(dict(id=j['id'],n=n,accepted_segments=sum(s['status']=='accepted_weak' for s in segments),
            accepted_length_px=sum(float(np.linalg.norm(np.diff(s['points_xy'],axis=0),axis=1).sum()) for s in segments if s['status']=='accepted_weak'),
            uncertain_segments=sum(s['status']=='uncertain' for s in segments),width_px=width))
    dump(DEST/'summary_draft.json',summary);print(json.dumps(summary,indent=2))
def record_completed_review():
    """Record the actual 2026-11-02 raw and overlay inspections, not an automatic visual approval."""
    receipt=[];stage_length={str(i):0. for i in range(1,5)}
    for n,j in enumerate(JOBS,1):
        path=DEST/(j['id']+'.json');item=load(path)
        assert sha(j['image_path'])==item['image_sha256']
        item['review'].update(native_overlay_viewed=True,full_context_overlay_viewed=True,
            status='accepted_with_unknown_tail_after_raw_and_overlay_review',review_date='2026-11-02',
            review_agent='/root/d1_labels_right',
            boundary_gate='Accepted points exclude abs(median_shift)>=3.75 px from the +/-4 px refinement corridor.',
            conclusion='Retained finite centers track visible entry-linked steel bands in native crops. Weak automatic labels only; no confirmed bent attached tail, physical contact, true edge endpoint, or independent human truth.',
            full_context_overlay_path=str(DEST/f'{n:02d}_overlay_context.png'))
        dump(path,item)
        acc=[s for s in item['segments'] if s['status']=='accepted_weak']
        lengths={str(k):sum(float(np.linalg.norm(np.diff(s['points_xy'],axis=0),axis=1).sum()) for s in acc if s['stage']==k) for k in range(1,5)}
        for k,v in lengths.items():stage_length[k]+=v
        receipt.append(dict(id=j['id'],label_path=str(path),label_sha256=sha(path),accepted_segments=len(acc),
            length_by_visual_stage_px=lengths,review_assets={f'{kind}_sha256':sha(DEST/f'{n:02d}_{kind}.png') for kind in ['raw_context','raw_native','overlay_context','overlay_native']},
            original_viewed=True,native_original_viewed=True,native_overlay_viewed=True,full_context_overlay_viewed=True))
    report=dict(review_date='2026-11-02',review_agent='/root/d1_labels_right',images=len(receipt),accepted_images=len(receipt),
        accepted_segments=sum(r['accepted_segments'] for r in receipt),length_by_visual_stage_px=stage_length,
        limitations=['Automatic weak labels, not independent human truth.', 'Stage 3 means visible projection over the coil, not confirmed physical contact.',
            'No stage 4 true-edge or bent-tail target accepted.', 'Unclear gaps, guide/tool occlusion, identity-ambiguous tails and unverified neighboring centers remain ignore.'],
        code_sha256=sha(__file__),helper_sha256=sha(HERE/'annotation_refine.py'),items=receipt)
    dump(DEST/'review_receipt.json',report);print(json.dumps({k:v for k,v in report.items() if k!='items'},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',nargs='?',default='prepare',choices=['prepare','labels','record_completed_review']);a=ap.parse_args()
    {'prepare':prepare,'labels':make_labels,'record_completed_review':record_completed_review}[a.mode]()
