"""Update only the event text after an all-frame event-rule regression.

All eleven videos were rendered from original frames and fully decoded first.
Unchanged event outputs retain byte-identical videos. Changed event text is
re-encoded from that verified render; F3, d, distances and source frame mapping
are unchanged. The earlier full renders and metadata are retained.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from common21 import *
import cv2,subprocess,imageio_ffmpeg,collections,shutil,concurrent.futures as cf
from PIL import Image,ImageDraw
from render_final21 import font

def process(item):
    cv2.setNumThreads(1);id=item['id'];out=ROOT/'delivery'/id;rows=read(out/'frames.json');summary=read(out/'summary.json');events=read(ROOT/'events_b'/f'{id}.json')
    changes=events['changed_display_frames'];target=Path(summary['video']);before=sha(target)
    assert before==summary['video_sha256']
    for r,e in zip(rows,events['frames']):r['near_event']=e
    if changes:
        archive=ROOT/'experiments'/'events_A_delivery'/id;archive.mkdir(parents=True,exist_ok=True)
        assert archive.resolve().is_relative_to(ROOT.resolve()) and target.resolve().is_relative_to(ROOT.resolve())
        assert not (archive/target.name).exists()
        for p in [out/'frames.json',out/'summary.json',ROOT/'events'/f'{id}.json']:shutil.copy2(p,archive/p.name)
        tmp=out/(target.stem+'.eventsB.part.mp4');cap=cv2.VideoCapture(str(target))
        proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(item['fps']),'-i','-',
            '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],
            stdin=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        names={'unknown':'近边事件：当前证据不足','near':'近边停留（视觉）','departing':'正在离开边缘（持续回升）','separated':'远离边缘','approaching':'接近边缘'}
        for i,row in enumerate(rows):
            ok,bgr=cap.read();assert ok;rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            if i in changes:
                im=Image.fromarray(rgb);dr=ImageDraw.Draw(im);dr.rectangle((840,874,1904,915),fill='#111b25');e=row['near_event'];label=names[e['state']]
                if e.get('active_event_start_s') is not None:label+=f"，起点 {e['active_event_start_s']:.1f}s"
                dr.text((852,882),label,font=font(24),fill='#edbd6c');rgb=np.asarray(im)
            proc.stdin.write(rgb.tobytes())
            if i in summary['review_frames']:cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(out/'review'/f'{i:06d}.jpg')
        cap.release();proc.stdin.close();err=proc.stderr.read().decode(errors='replace');assert proc.wait()==0,err
        target.replace(archive/target.name);tmp.replace(target)
        summary['event_label_refinement']=dict(previous_video=str(archive/target.name),previous_video_sha256=before,changed_display_frames=changes,
            geometry_count_d_and_source_mapping_unchanged=True,method='event_text_only_over_verified_full_render',encoding_crf=16)
    summary.update(video_sha256=sha(target),near_events=events['events'],event_states=dict(collections.Counter(r['near_event']['state'] for r in rows)))
    dump(out/'frames.json',rows);dump(out/'summary.json',summary);dump(ROOT/'events'/f'{id}.json',events)
    result=dict(id=id,changed_display_frames=changes,previous_video_sha256=before,final_video_sha256=summary['video_sha256'],unchanged_video_bytes=not changes)
    print(id,'event text changes',len(changes),flush=True);return summary,result

if __name__=='__main__':
    old=read(ROOT/'delivery_receipt.json');dump(ROOT/'delivery_receipt_events_A.json',old)
    with cf.ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(process,plan()))
    inherited=dict(old['code_hashes']);inherited['finish_final21_A_frozen.py']=inherited.pop('finish_final21.py')
    dump(ROOT/'event_refinement_receipt.json',dict(videos=[r[1] for r in results],event_code_sha256=sha(ROOT/'events21b.py'),label_update_code_sha256=sha(ROOT/'update_event_labels21.py')))
    dump(ROOT/'delivery_receipt.json',dict(videos=[r[0] for r in results],code_hashes=inherited,
        event_refinement_code_hashes={f:sha(ROOT/f) for f in ['events21b.py','update_event_labels21.py']},future_full_render_entry='finish_final21.py'))
