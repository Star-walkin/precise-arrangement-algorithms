"""Prepare deterministic review sheets from existing completed-run previews only.

No source-video decoding, model invocation, or detection changes. Human/agent
visual observations are separately recorded in visual_review_agent.json.
"""
from pathlib import Path
import json, hashlib
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit'/'visual_review_agent'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    recorded=[]
    batches=sorted(OUT.glob('batch_[0-9][0-9][0-9].json'))
    for p in batches:recorded+=json.loads(p.read_text(encoding='utf8'))['videos']
    done={v['video_id'] for v in recorded}
    source=json.loads((ROOT/'source_trace'/'source_trace.json').read_text(encoding='utf8'))['videos']
    pending=[]
    for item in source:
        ident=item['video_id'];folder=ROOT/'all_001'/ident;sfile=folder/'summary.json'
        if ident in done or not sfile.is_file():continue
        summary=json.loads(sfile.read_text(encoding='utf8'))
        if summary.get('status')!='complete':continue
        choices=sorted(folder.joinpath('previews').glob('*.jpg'),key=lambda p:int(p.stem))
        if not choices:continue
        fps=summary['source_nominal_fps'];last=summary['source_frames_traversed']-1
        targets=[('anchor_plus_3s',min(last,summary['anchor_frame']+round(3*fps))),('middle',last//2),('end',last)]
        samples=[]
        for label,target in targets:
            pick=min(choices,key=lambda p:(abs(int(p.stem)-target),int(p.stem)))
            index=int(pick.stem)
            kind='saved_run_preview';output_frame_index=None
            # The 5Hz writer can miss the source-last +/-fps/10 preview trigger.
            # Never use a midpoint as an endpoint review. Decode only the
            # already-produced four-panel MP4, never a source video.
            if abs(index-target)/fps>.3:
                records=[json.loads(line) for line in (folder/'render_frames.jsonl').read_text(encoding='utf8').splitlines() if line]
                nearest=min(records,key=lambda r:(abs(r['source_frame_index']-target),r['source_frame_index']))
                index=nearest['source_frame_index'];output_frame_index=nearest['output_frame_index']
                supplemental=OUT/'supplemental';supplemental.mkdir(exist_ok=True)
                pick=supplemental/f'{ident}__f{index:06d}__output{output_frame_index:06d}.jpg'
                if not pick.is_file():
                    cap=cv2.VideoCapture(summary['output_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,output_frame_index)
                    ok,pixels=cap.read();cap.release()
                    if not ok:raise RuntimeError('Cannot decode existing output preview')
                    cv2.imencode('.jpg',pixels,[cv2.IMWRITE_JPEG_QUALITY,96])[1].tofile(str(pick))
                kind='decoded_existing_four_panel_output'
                if abs(index-target)/fps>.3:raise ValueError('No output frame close to requested fixed review position')
            samples.append(dict(position=label,target_source_frame=target,selected_source_frame=index,
                media_time_s=index/fps,path=str(pick),sha256=sha(pick),preview_kind=kind,output_frame_index=output_frame_index))
        pending.append(dict(video_id=ident,summary_path=str(sfile),summary_sha256=sha(sfile),samples=samples,
            total_status_counts=summary['counts'],source_duration_s=summary['source_duration_s']))
    number=max([int(p.stem.split('_')[1]) for p in batches],default=0)
    generated=[]
    for start in range(0,len(pending),2):
        videos=pending[start:start+2];number+=1
        canvas=np.full((3*674,len(videos)*800,3),245,np.uint8)
        for col,video in enumerate(videos):
            for row,sample in enumerate(video['samples']):
                pixels=cv2.imdecode(np.fromfile(sample['path'],np.uint8),cv2.IMREAD_COLOR)
                if pixels.shape!=(1280,1600,3):raise ValueError('Unexpected preview dimensions')
                tile=cv2.resize(pixels,(800,640),interpolation=cv2.INTER_AREA)
                caption=Image.new('RGB',(800,34),(255,255,240));draw=ImageDraw.Draw(caption)
                draw.text((5,6),f"{video['video_id']} | {sample['position']} t={sample['media_time_s']:.2f}s f={sample['selected_source_frame']}",font=FONT,fill=(20,20,20))
                cell=np.concatenate([cv2.cvtColor(np.asarray(caption),cv2.COLOR_RGB2BGR),tile],axis=0)
                canvas[row*674:(row+1)*674,col*800:(col+1)*800]=cell
        stem=f'batch_{number:03d}';image_path=OUT/(stem+'.jpg')
        cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,96])[1].tofile(str(image_path))
        manifest=dict(selection_rule='Completed source; nearest saved preview to anchor+3s, source midpoint and final source frame. Ties choose earlier frame. No effect-based selection.',
            videos=videos,contact_sheet=str(image_path),contact_sheet_sha256=sha(image_path),read_source_video=False,model_forward=False)
        (OUT/(stem+'.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        generated.append(dict(batch=stem,videos=[v['video_id'] for v in videos],path=str(image_path)))
    print(json.dumps(dict(previously_prepared_videos=len(done),new_videos=len(pending),generated=generated),ensure_ascii=False))

if __name__=='__main__':main()
