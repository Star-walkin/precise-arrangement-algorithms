"""Replace diagnostic reason-code labels only; detector records stay unchanged."""
import cv2,json,numpy as np,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg
from render_results import ROOT,REASONS,fonts
from run_full import sha,dump

def patch(frame,row):
    reasons=row['reasons'][:3]
    if not reasons:return frame
    y=524 if row['F3_called'] else 452
    box=Image.new('RGB',(615,750-(y-4)),(15,18,22));d=ImageDraw.Draw(box);yy=4
    for reason in reasons:
        assert reason in REASONS,reason
        label=REASONS[reason]
        for k in range(0,len(label),27):d.text((5,yy),label[k:k+27],font=fonts[18],fill=(255,190,140));yy+=26
    frame[y-4:750,1305:1920]=cv2.cvtColor(np.array(box),cv2.COLOR_RGB2BGR)
    return frame

if __name__=='__main__':
    cv2.setNumThreads(1);out=ROOT/'delivery';src=out/'BG1_boundary_gated_full.mp4'
    temp=out/'BG1_boundary_gated_full_localized.tmp.mp4';backup=out/'BG1_boundary_gated_full_diagnostic_labels.mp4'
    if backup.exists() or temp.exists():raise FileExistsError('Preserve prior localization')
    rows=json.loads((ROOT/'full_002/receipt.json').read_text(encoding='utf8'))['frame_rows']
    old_sha=sha(src);cap=cv2.VideoCapture(str(src))
    f=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24',
        '-s','1920x1080','-r','10','-i','-','-an','-c:v','libx264','-threads','2','-crf','18','-preset','fast','-pix_fmt','yuv420p',str(temp)],
        stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=(out/'localize_ffmpeg.log').open('wb'))
    for i,row in enumerate(rows):
        ok,a=cap.read();assert ok and row['frame']==i
        f.stdin.write(patch(a,row).tobytes())
        if i%200==0:print('Localized',i,flush=True)
    assert not cap.read()[0];cap.release();f.stdin.close();assert f.wait()==0
    src.rename(backup);temp.rename(src)
    for p in out.glob('preview_*.png'):
        i=int(p.stem.split('_')[-1]);a=cv2.imdecode(np.fromfile(p,dtype=np.uint8),1)
        cv2.imencode('.png',patch(a,rows[i]))[1].tofile(str(p))
    dump(out/'label_update_receipt.json',dict(frames=601,input_sha256=old_sha,output_sha256=sha(src),
        change='Chinese diagnostic labels in sidebar only; video re-encoded; detector data and numeric values unchanged',
        script_sha256=sha(__file__)))
    print('Localized video complete',flush=True)
