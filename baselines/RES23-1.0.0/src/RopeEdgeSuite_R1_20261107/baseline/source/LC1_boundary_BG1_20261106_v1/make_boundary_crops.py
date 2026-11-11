"""Fixed review frames: raw pixels beside sparse current-interface evidence."""
from pathlib import Path
import cv2,numpy as np,json
from PIL import Image,ImageDraw,ImageFont
from inspect_regions import SOURCE
from run_full import ROOT
def make(indices,outname):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    sheet=Image.new('RGB',(1240,300*len(indices)),(20,27,35));d=ImageDraw.Draw(sheet)
    cap=cv2.VideoCapture(str(SOURCE))
    for row,i in enumerate(indices):
        r=json.loads((ROOT/'full_002/frames'/f'{i:06d}.json').read_text(encoding='utf8'))
        cap.set(1,i);ok,a=cap.read();assert ok
        edge=r['region']['edge_y'];x0=2119;y0=round(edge or 2340)-80
        crop=a[y0:y0+140,x0:x0+240].copy();ann=crop.copy()
        for x,y in r['region'].get('top_observed_xy',[]):
            if x0<=x<x0+240 and y0<=y<y0+140:cv2.circle(ann,(round(x-x0),round(y-y0)),1,(255,0,255),-1)
        cv2.line(ann,(120,0),(120,139),(0,255,255),1)
        if edge is not None:cv2.drawMarker(ann,(120,round(edge-y0)),(255,255,255),cv2.MARKER_CROSS,9,1)
        sheet.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((480,280)),(0,row*300+20))
        sheet.paste(Image.fromarray(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB)).resize((480,280)),(490,row*300+20))
        d.text((985,row*300+35),f'帧 {i}',font=font,fill='white')
        d.text((985,row*300+70),'边界通过' if r['region']['valid'] else '边界拒识',font=font,fill='#65dfae' if r['region']['valid'] else '#ffa566')
        d.text((985,row*300+105),f'y={edge:.2f}' if edge is not None else 'y未知',font=font,fill='white')
        d.text((985,row*300+145),'左：原图',font=font,fill='white');d.text((985,row*300+180),'中：界面证据',font=font,fill='white')
    cap.release();sheet.save(ROOT/'delivery'/outname)
if __name__=='__main__':make([0,60,121,127,150,200],'BG1_boundary_closeups_early.png')
