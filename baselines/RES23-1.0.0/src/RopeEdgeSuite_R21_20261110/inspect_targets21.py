import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import cv2
from PIL import Image,ImageDraw,ImageFont
from common21 import *

def main():
    cv2.setNumThreads(1);out=ROOT/'inspection';out.mkdir(exist_ok=True)
    targets={'dual_20260924_111143__right':[0,30,60,90,120,140,150,160],
             'dual_20260924_111500__left':[200,230,240,245,247,249,255,260,270,280,290,300,310,320]}
    for id,frames in targets.items():
        item=next(x for x in plan() if x['id']==id);base=read(R20/'delivery'/id/'frames.json');rows=[];tiles=[]
        cap=cv2.VideoCapture(item['source'])
        for i in frames:
            cap.set(1,i);ok,im=cap.read();r=frontend(id,i);assert ok and pixelsha(im)==r['source_pixel_sha256']
            g=geometry(id,i);c=count20(id,i);a=g.get('axis');m=c['measurement'];t=g.get('tangent') or {}
            rows.append(dict(frame=i,N=m.get('N'),N_interval=m.get('N_interval'),contact=base[i]['contact'],filtered=base[i]['filtered'],
                tangent={k:t.get(k) for k in ['valid','d','interval','neighbor_count','curve_d_mad_px']},
                axis={k:a.get(k) for k in ['line_abc','uncertainty_px','width_px','current_coil_side_measured']} if a else None))
            if item['camera']=='right':im=cv2.flip(im,1)
            # Normalized left-entering crop includes current rope and coil rim.
            crop=im[1800:2850,700:2850];cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,96])[1].tofile(out/f'{id}_{i:06d}_raw.jpg')
            tile=Image.fromarray(cv2.cvtColor(cv2.resize(crop,(860,420)),cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(tile)
            d.rectangle((0,0,860,26),fill='black');d.text((5,3),f'{i/10:.1f}s   N={m.get("N")}   {base[i]["contact"]["state"]}',fill='white')
            tiles.append(tile)
        cap.release()
        for k in range(0,len(tiles),4):
            canvas=Image.new('RGB',(1720,840),'#111111')
            for j,tile in enumerate(tiles[k:k+4]):canvas.paste(tile,((j%2)*860,(j//2)*420))
            canvas.save(out/f'{id}_sheet{k//4}.jpg',quality=94)
        dump(out/f'{id}_samples.json',rows)
        print(id,[dict(frame=r['frame'],N=r['N'],local_N=r['contact'].get('current_local_N'),d=r['tangent'].get('d'),axis=r['axis']) for r in rows],flush=True)
    dump(ROOT/'human_reference.json',dict(source='user supplied approximate visual timestamps; evaluation only, not detector inputs',
        id='dual_20260924_111500__left',near_s=24.9,layer_entry_s=24.7,leave_s=29.0,approximate=True,
        independent_generalization_truth=False))

if __name__=='__main__':main()
