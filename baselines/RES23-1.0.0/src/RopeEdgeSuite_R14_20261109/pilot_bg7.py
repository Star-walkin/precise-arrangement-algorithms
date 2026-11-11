import cv2,numpy as np
from common import *
from boundary_bg7 import Boundary
from PIL import Image,ImageDraw,ImageFont

def main():
    cv2.setNumThreads(1);b=Boundary();cap=cv2.VideoCapture(read(RUN12/'binding.json')['source']);rows=[]
    frames=list(range(0,81,5))+[100,150,200,250,280,300,310,320,336,350,360,370,380,390,400,410,420,450,500,599]
    tiles=[]
    for i in frames:
        cap.set(1,i);ok,im=cap.read();assert ok;r=record(i);assert pixelsha(im)==r['source_pixel_sha256']
        n=b.step(im,r['region_BG5'],r['S1_tracker_full'],i)
        dump(ROOT/'pilot'/f'{i:06d}.json',n)
        rows.append({k:n.get(k) for k in ['frame','fit_valid','count_boundary_valid','edge_y','edge_uncertainty_px','d_locally_observed','shape_age_frames','current_clean_points','reasons']})
        print(rows[-1],flush=True)
        if i in [20,100,250,336,370,390,450,599]:
            old=np.asarray(r['region_BG5']['top_curve_xy'],np.int32)
            cv2.polylines(im,[old],False,(255,120,0),4)
            if n.get('top_curve_xy'):cv2.polylines(im,[np.asarray(n['top_curve_xy'],np.int32)],False,(0,190,255),5)
            for pt in n.get('top_observed_xy',[]):cv2.circle(im,tuple(np.rint(pt).astype(int)),3,(0,255,0),-1)
            for pt in n.get('excluded_candidates_xy',[]):cv2.circle(im,tuple(np.rint(pt).astype(int)),4,(0,0,255),-1)
            crop=Image.fromarray(cv2.cvtColor(im[1900:2450,500:2900],cv2.COLOR_BGR2RGB)).resize((1200,275))
            tile=Image.new('RGB',(1200,310),'#17222c');tile.paste(crop,(0,35));dr=ImageDraw.Draw(tile)
            dr.text((8,5),f"{i/10:.1f}s   fit={n['fit_valid']}   d={n.get('edge_y')}   uncertainty={n.get('edge_uncertainty_px')}",fill='white')
            tiles.append(tile)
    cap.release();dump(ROOT/'pilot_summary.json',rows)
    sheet=Image.new('RGB',(2400,310*((len(tiles)+1)//2)))
    for j,t in enumerate(tiles):sheet.paste(t,((j%2)*1200,(j//2)*310))
    sheet.save(ROOT/'pilot_contact.jpg',quality=95)
if __name__=='__main__':main()
