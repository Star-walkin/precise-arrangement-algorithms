"""Populate explicit no-count observations where current SG5 has no rope.

No decoded frame is omitted. Frames with an existing F3 result are unchanged.
The running extractor accepts these provenance-checked cache records; they
deliberately suppress unnecessary F3 work when a count has no active endpoint.
"""
import cv2,time
from common20 import *
from roi20 import masks

def main():
    id='dual_20260923_113029__right';item=next(x for x in plan() if x['id']==id)
    cap=cv2.VideoCapture(item['source']);out=ROOT/'f3_v1'/id;rows=[]
    producer_hash=sha(__file__)
    for i in range(item['planned_frames']):
        ok,raw=cap.read();assert ok
        # Stay well ahead of the ongoing extractor; existing records always win.
        if i<230:continue
        r=record(id,i);assert pixelsha(raw)==r['source_pixel_sha256']
        dest=out/'records'/f'{i:06d}.json'
        if r['track'].get('valid') or dest.exists():continue
        tick=time.time();im=cv2.flip(raw,1);_,mask,meta=masks(im,r)
        row=dict(frame=i,time_s=i/item['fps'],analysis_pixel_sha256=pixelsha(im),source_pixel_sha256=r['source_pixel_sha256'],
            input_stage_record_sha256=sha(selected_run(id)/'frames'/f'{i:06d}.json.gz'),roi=meta,mask_sha256=pixelsha(mask),
            called=False,points=0,fields_sha256=None,elapsed_s=time.time()-tick,
            skip_reason='no_current_SG5_active_endpoint_so_no_count_possible',cache_producer=str(Path(__file__)),cache_producer_sha256=producer_hash)
        # This range is known to be ahead; also decline if the producer arrived.
        if dest.exists():continue
        dump(dest,row);rows.append(dict(frame=i,record_sha256=sha(dest)))
    cap.release();assert sha(item['source'])==item['source_sha256']
    dump(ROOT/'inactive_F3_gate_receipt.json',dict(id=id,records=rows,source_verified=True,skipped_decoded_frames=0,
        algorithm='current_identity_prerequisite_for_count',producer_sha256=producer_hash))
    print('explicit inactive-count records',len(rows),flush=True)
if __name__=='__main__':main()
