import argparse,time
import cv2,numpy as np
from PIL import Image
from common import *
from engine import Engine
from render import panel,writer
CORE_FILES=[ROOT/'engine.py',ROOT/'tracker_sg4.py',ROOT/'boundary_bg9.py',ROOT/'camera_cg2.py',
    ROOT/'common.py',ROOT/'render.py',R12/'tracker_sg2.py',R11/'region_bg5_refined.py',
    PARENT/'RopeEdgeSuite_R15_20261109/tracker_sg3.py']
def transform_image(im,camera):return cv2.flip(im,1) if camera=='right' else im
def original_abc(abc,width,camera):
    if abc is None:return None
    a,b,c=abc
    return [-a,b,a*(width-1)+c] if camera=='right' else [a,b,c]
def process(item,limit=None,probe=False):
    cap=cv2.VideoCapture(item.get('analysis_source',item['source']));fps=float(item['fps']);n=item['planned_frames'];end=n if limit is None else min(n,limit)
    out=ROOT/('probes_fresh' if probe else 'runs_fresh')/item['id'];out.mkdir(parents=True,exist_ok=False)
    codes={str(p):sha(p) for p in CORE_FILES};dump(out/'binding.json',dict(item,planned_end=end,cold_start=True,
        reused_reference_d=D,recalibrated_d=False,analysis_orientation='left_entering',original_camera=item['camera'],
        horizontal_mirror=item['camera']=='right',core_hashes=codes,runner_sha256=sha(__file__),
        input_scope='full decoded video, every frame, no per-video threshold tuning',algorithm='RES-1.6/SG4-BG9-CG2'))
    engine=Engine();rows=[];start=time.perf_counter();video=out/(item['id']+'__RES16_fresh.mp4');enc=writer(video,fps)
    snapshot=set([0,min(10,end-1),end//4,end//2,3*end//4,end-1]);assert end>0
    try:
        for i in range(end):
            ok,original=cap.read()
            if not ok:raise RuntimeError(f'Unexpected decode failure at frame {i}, planned {end}')
            image=transform_image(original,item['camera']);r=engine.step(image,i/fps,i)
            r.update(frame_index=i,time_s=i/fps,source_pixel_sha256=pixelsha(original),analysis_pixel_sha256=pixelsha(image),
                original_line_abc=original_abc(r['track']['current_line_abc'],image.shape[1],item['camera']))
            save_record(out,i,r);rgb=panel(original,r,item);enc.stdin.write(rgb.tobytes())
            if i in snapshot:Image.fromarray(rgb).save(out/f'preview_{i:06d}.jpg',quality=92)
            b=r['boundary'];row=dict(frame=i,time_s=i/fps,tracking_valid=r['track']['valid'],proposal_valid=r['proposal']['valid'],
                boundary_fit=b['fit_valid'],count_boundary_eligible=b['count_boundary_valid'],
                d_locally_observed=b.get('d_locally_observed'),foreground_at_d=b.get('d_foreground_occluded'),
                boundary_reasons=b['reasons'],tracking_reason=r['track']['diagnostics']['accept_reason'],
                algorithm_errors=[q['stage'] for q in r['algorithm_errors']],model_forwards=r['track']['model_forward_count'])
            rows.append(row)
            if i%20==0 or i==end-1:
                v=dict(id=item['id'],processed=i+1,planned=end,tracking=sum(q['tracking_valid'] for q in rows),
                    boundary=sum(q['boundary_fit'] for q in rows),errors=sum(bool(q['algorithm_errors']) for q in rows),elapsed_s=round(time.perf_counter()-start,1))
                dump(out/'progress.json',v);print(v,flush=True)
        cap.release();enc.stdin.close();err=enc.stderr.read();assert enc.wait()==0,err
        dump(out/'compact.json',rows)
        # Decode every output frame before declaring a delivered video complete.
        check=cv2.VideoCapture(str(video));count=0
        while True:
            ok,im=check.read()
            if not ok:break
            assert im.shape==(1080,1920,3);count+=1
        outputfps=check.get(5);check.release();assert count==end and abs(outputfps-fps)<.001
        assert all(sha(p)==v for p,v in codes.items())
        assert sha(item['source'])==item['source_sha256']
        receipt=dict(status='complete',frames=end,decoded_output_frames=count,fps=outputfps,video=str(video),video_sha256=sha(video),
            source_unchanged=True,core_unchanged=True,elapsed_s=time.perf_counter()-start,partial_source=item.get('partial_source',False),
            skipped_frames=0,F3_run=False,count_run=False,algorithm_error_frames=sum(bool(q['algorithm_errors']) for q in rows))
        dump(out/'receipt.json',receipt);return receipt
    except BaseException:
        import traceback
        cap.release()
        if enc.stdin and not enc.stdin.closed:enc.stdin.close()
        enc.wait(timeout=60)
        dump(out/'failure.json',dict(error=traceback.format_exc(),completed_frames=len(rows)))
        raise
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--limit',type=int);ap.add_argument('--probe',action='store_true');a=ap.parse_args()
    plan=read(ROOT/'plan.json');item=next(r for r in plan['videos'] if r['id']==a.id);process(item,a.limit,a.probe)
if __name__=='__main__':main()
