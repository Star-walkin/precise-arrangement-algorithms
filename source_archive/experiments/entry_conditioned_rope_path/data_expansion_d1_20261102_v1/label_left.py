"""Codex-reviewed raw-image left-view finite weak-center labels.

Finite x-limits selected after full/native raw viewing; no network response is
read. A7 axis initializes optical center only; low raw support remains unknown.
"""
from pathlib import Path
import json,hashlib,sys
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from annotation_refine import coarse_axis,axis_xy,refine_segment,runs,sha,load
HERE=Path(__file__).resolve().parent;OUT=HERE/'labels_left'
END={15:2030,205:2050,427:2000,585:1980,743:2010,902:2030,1091:1840,1250:1960,
     46:2350,1626:2270,3114:2320,4601:2290}
def num(j):return int(j['id'].split('__f')[1].split('__')[0])
def main():
    OUT.mkdir(exist_ok=True);protocol=load(HERE/'protocol.json');rows=[]
    for job in protocol['candidate_jobs']:
        if not job['source_group'].startswith('L_'):continue
        ident=job['id'];n=num(job);isrun=job['source_group']=='L_run_20260923_01'
        raw=cv2.imdecode(np.fromfile(job['image_path'],np.uint8),cv2.IMREAD_COLOR)
        assert raw.shape==(3000,4096,3) and sha(job['image_path'])==job['image_sha256']
        axis=coarse_axis(job);seed=np.asarray(job['seeds'][0]['xy']);start=max(100.,float(seed[-1,0]+20))
        end=END[n];breakx=1110 if isrun else 950
        anchors=axis_xy(axis,[start,end]);width=float(axis['width'])
        # The narrow first four raw appearances are not the broad cast shadow.
        if isrun and n in [15,205,427,585]:width=min(width,12.)
        pts,valid,diag,detail=refine_segment(raw,anchors,width,max_shift=4.)
        # Finite visual target is clear through the overlap; response-independent
        # gating explicitly keeps ambiguous gaps out of supervision.
        maskpath=HERE.parents[2]/'数据资产/05_派生与实验结果/camera_condition_sam2_20261009_v1/all_001/body_branches_002/masks/left_camera'/(ident+'.png')
        mask=cv2.imdecode(np.fromfile(maskpath,np.uint8),cv2.IMREAD_GRAYSCALE)>0
        qi=np.rint(pts).astype(int);inside=mask[qi[:,1].clip(0,2999),qi[:,0].clip(0,4095)]
        seg=[]
        for stage,where in [(1,~inside),(2,inside)]:
            for a,b in runs(valid&where,8):
                seg.append(dict(status='accepted_weak',points_xy=pts[a:b].tolist(),stage=stage,valid_radius_px=6.,
                    reason='Raw full context establishes entry-connected steel body; native opposing edges refine center inside bounded corridor. Projected-on-coil visual segment, not physical contact truth.' if stage==2 else 'Entry-connected free rope visibly supported before spool overlap.',
                    initial_anchors_xy=anchors.tolist(),refinement=diag,point_indices=[int(a),int(b)],
                    continuity_basis='same finite raw optical trajectory from unchanged short entrance; no U-Net heat or selected H6 path used',
                    optical_width_px=width))
        # Unknown last segment is deliberately not a true endpoint or negative.
        tail=axis_xy(axis,[end,end+200])
        seg.append(dict(status='uncertain',points_xy=tail.tolist(),stage=3,valid_radius_px=0.,
            reason='Beyond finite reviewed visible body, bend/neighbor identity and precise center cannot be separated reliably; ignore rather than extrapolate.'))
        invalids=~valid
        for a,b in runs(invalids,3):
            seg.append(dict(status='uncertain',points_xy=pts[a:b].tolist(),stage=2 if pts[a:b,0].mean()>=breakx else 1,valid_radius_px=0.,reason='Native opposing-edge support insufficient; not gap-filled.'))
        overlay=Image.fromarray(raw[:,:,::-1]);draw=ImageDraw.Draw(overlay)
        for s in seg:
            if len(s['points_xy'])>1:draw.line([tuple(p) for p in s['points_xy']],fill=(255,30,40) if s['status']=='accepted_weak' else (255,190,0),width=1 if s['status']=='accepted_weak' else 2)
        bbox=[max(0,int(start)-80),max(1050,int(min(anchors[:,1]))-120),min(4096,end+180),min(3000,int(max(tail[:,1]))+120)]
        Image.fromarray(raw[:,:,::-1]).crop(bbox).save(OUT/(ident+'__context_raw.png'))
        overlay.crop(bbox).save(OUT/(ident+'__context_overlay.png'));overlay.save(OUT/(ident+'__overlay_full.png'))
        # Two native windows overlap around difficult projected-on-coil body.
        native=[]
        for x0 in [max(500,breakx-180),max(900,end-760)]:
            x1=min(4096,x0+1000);q=axis_xy(axis,[x0,x1]);y0=max(1050,int(q[:,1].min())-110);y1=min(3000,int(q[:,1].max())+130)
            bb=[int(x0),y0,int(x1),y1];rp=OUT/(ident+f'__native_{len(native)}_raw.png');op=OUT/(ident+f'__native_{len(native)}_overlay.png')
            Image.fromarray(raw[:,:,::-1]).crop(bb).save(rp);overlay.crop(bb).save(op)
            native.append(dict(bbox_xyxy=bb,raw_path=str(rp),overlay_path=str(op),scale=1.,mapping='original_xy=crop_xy+[x0,y0]'))
        response=dict(id=ident,image_path=job['image_path'],image_sha256=job['image_sha256'],source_group=job['source_group'],camera_role=job['camera_role'],
            seed_xy=job['seeds'][0]['xy'],seed_source=job['seeds'][0]['source'],center_sigma_original_px=3.,segments=seg,neighbor_segments=[],
            selection_reason='Temporal-quantile real image; leftward entrance to rightward target body with brightness/width change and difficult coil-background overlap',
            raw_candidate_axis={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in axis.items()},
            crop_context_bbox=bbox,native_crops=native,raw_profile_diagnostics=detail,visual_phase_mask=dict(path=str(maskpath),sha256=sha(maskpath),meaning='independent saved spool ROI for projected-position stage only, never target identity'),
            review=dict(original_viewed=True,native_raw_viewed=True,native_overlay_viewed=False,reviewer='Codex multimodal same-agent consistency review; not independent human truth',
                raw_view_before_network_heat=True,network_heat_criterion=False,full_target_identity_claimed=False,
                stage_semantics='1 clearly free visual segment; 2 visible body projected over spool; uncertain3 is unresolved bend, not negative; no confirmed physical contact or edge endpoint'),
            new_annotation_kind='automatic_weak',not_independent_human_truth=True)
        p=OUT/(ident+'.json');p.write_text(json.dumps(response,ensure_ascii=False,indent=2),encoding='utf-8')
        rows.append(dict(id=ident,accepted_segments=sum(s['status']=='accepted_weak' for s in seg),accepted_length_px=sum(float(np.linalg.norm(np.diff(s['points_xy'],axis=0),axis=1).sum()) for s in seg if s['status']=='accepted_weak'),supported_fraction=diag['supported_fraction'],json_path=str(p)))
        print(rows[-1],flush=True)
    (OUT/'pre_review_summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
