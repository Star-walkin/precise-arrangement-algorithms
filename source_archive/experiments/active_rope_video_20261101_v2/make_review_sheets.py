"""Fixed3positions per video; existing output previews only, no new inference."""
import sys,json
from pathlib import Path
sys.path.append('D:/precise_arrangement/vision_prototype/.venv/Lib/site-packages')
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent;run=HERE/'all_001';out=HERE/'visual_review';out.mkdir(exist_ok=True)
s=json.loads((run/'summary.json').read_text(encoding='utf8')) if (run/'summary.json').exists() else dict(results=[json.loads(p.read_text(encoding='utf8')) for p in run.glob('*/summary.json')])
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);rows=[]
for v in sorted(s['results'],key=lambda x:x['video_id']):
    n=v['source_frames'];fps=v['output_fps'];indices=[min(n-1,round(3*fps)),n//2,n-1];strips=[]
    for index in indices:
        p=run/v['video_id']/'previews'/f'{index:06d}.jpg'
        im=cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_COLOR);assert im is not None
        # Corresponding original y~1700..3000 in ALL four panels. Labels kept
        # separately, and full previews remain available for any wider context.
        crop=np.concatenate([im[386:640],im[1026:1280]],axis=0)
        title=Image.new('RGB',(1600,34),(248,248,248));draw=ImageDraw.Draw(title)
        draw.text((8,2),f"{v['video_id']}  源帧 {index}  媒体{index/fps:.2f}s  上:原图/B热图 下:当帧灰线候选/最终位置",font=font,fill=(10,10,10))
        strips.append(np.concatenate([cv2.cvtColor(np.asarray(title),cv2.COLOR_RGB2BGR),crop]))
    sheet=np.concatenate(strips);target=out/(v['video_id']+'.jpg');cv2.imencode('.jpg',sheet,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(str(target))
    rows.append(dict(video_id=v['video_id'],indices=indices,sheet=str(target),full_previews=[str(run/v['video_id']/'previews'/f'{i:06d}.jpg') for i in indices]))
(out/'selection.json').write_text(json.dumps(dict(rule='fixed3s/midpoint/end, all16videos, no outcome exclusion',videos=rows),ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(videos=len(rows),positions=sum(len(r['indices']) for r in rows))))
