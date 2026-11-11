"""Validate selected video artifacts and preserve additional-rejection evidence."""
from html.parser import HTMLParser
from urllib.parse import unquote
import cv2
from common import *
from mixed_comparison import BASELINE,choose_run

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ('href','src','poster') and v)

def main():
    plan=read(ROOT/'plan.json');manifest=read(ROOT/'delivery_manifest.json')
    assert len(manifest['videos'])==11 and manifest['frames']==4819
    assert manifest['algorithm_error_frames']==0 and not manifest['F3_run'] and not manifest['count_run']
    outputs=[];rejections=[]
    for item in plan['videos']:
        id=item['id'];run=choose_run(id);receipt=read(run/'receipt.json')
        v=Path(receipt['video']);assert sha(v)==receipt['video_sha256']
        assert receipt['status']=='complete' and receipt['decoded_output_frames']==item['planned_frames']==receipt['frames']
        assert receipt['source_unchanged'] and receipt['core_unchanged'] and receipt['algorithm_error_frames']==0
        assert receipt['pixel_hash_verified_frames']==receipt['frames'] and receipt['skipped_frames']==0
        cap=cv2.VideoCapture(str(v));assert cap.isOpened()
        w,h,fps,n=[cap.get(k) for k in (3,4,5,7)];cap.release()
        assert (int(w),int(h))==(1920,1080) and abs(fps-item['fps'])<.001 and int(n)==item['planned_frames']
        old=read(BASELINE/'runs'/id/'compact.json');new=read(run/'compact.json')
        lost=[]
        for a,b in zip(old,new):
            if a['tracking_valid'] and not b['tracking_valid']:
                r=record(run,b['frame']);diag=r['track']['diagnostics'].get('SG5',{})
                lost.append(dict(frame=b['frame'],time_s=b['time_s'],reason=b['tracking_reason'],
                    strict_check=diag.get('strict_check'),fallback_check=diag.get('fallback_check')))
        if lost:rejections.append(dict(id=id,additional_rejections=lost,old=sum(x['tracking_valid'] for x in old),new=sum(x['tracking_valid'] for x in new)))
        outputs.append(dict(id=id,frames=int(n),video_sha256=receipt['video_sha256'],width=int(w),height=int(h),fps=fps,
            prior_full_decode_verified=True,receipt_sha256=sha(run/'receipt.json')))
    dump(ROOT/'additional_rejections.json',dict(semantics='These are frames accepted by the baseline but rejected by the new gates. Rejection is not proof of a baseline error; no independent ground truth exists.',videos=rejections))
    parser=Links();parser.feed((ROOT/'结果索引.html').read_text(encoding='utf8'))
    links=sorted(set(parser.links));missing=[x for x in links if x!='audit_receipt.json' and not (ROOT/unquote(x)).resolve().is_file()]
    assert not missing,missing
    snapshot=read(ROOT/'frozen/hashes.json')
    for source,digest in snapshot['files'].items():
        assert sha(source)==digest,source
        copied=ROOT/'frozen'/Path(source).name
        if Path(source).parent==ROOT:assert copied.exists() and sha(copied)==digest
    control=read(ROOT/'control_audit.json');assert control['frames']==sum(s['tested_frames'] for s in control['sources'])==550 and control['detections']==0
    review=read(ROOT/'visual_review.json');assert review['selected_final_frames']==53
    dump(ROOT/'audit_receipt.json',dict(status='passed',algorithm=manifest['algorithm'],videos=outputs,frames=sum(x['frames'] for x in outputs),
        html_local_links_checked=len(links),missing_links=missing,frozen_hashes_unchanged=True,
        full_video_source_count=8,partial_source_count=3,negative_control_sample_frames=control['frames'],
        negative_controls_fully_retested=False,visual_review_frames=53,
        newly_computed_final_supplement_sources=1,reused_after_recorded_transition_invariance_audit=10,
        limitations=['Availability counts are not labelled accuracy.','F3, remaining-root counts and physical-contact decisions were not rerun or certified.']))
    assert all((ROOT/unquote(x)).resolve().is_file() for x in links)
    print('Delivery audit passed:',len(outputs),'videos,',sum(x['frames'] for x in outputs),'frames,',len(links),'local links')

if __name__=='__main__':main()
