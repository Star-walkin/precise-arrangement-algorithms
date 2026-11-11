from common21 import *
import cv2
from PIL import Image,ImageDraw
src=r'E:\study\program\济宁\双相机同步\dual_20260924_111500\right_DA9626975.avi'
cap=cv2.VideoCapture(src);panels=[]
for i in [240,247,255,290]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();assert ok
    panel=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));panel.thumbnail((819,600));ImageDraw.Draw(panel).text((10,10),f'right original {i/10:.1f}s',fill='red');panels.append(panel)
canvas=Image.new('RGB',(1638,1200))
for i,p in enumerate(panels):canvas.paste(p,((i%2)*819,(i//2)*600))
canvas.save(ROOT/'inspection'/'111500_other_camera.jpg')
