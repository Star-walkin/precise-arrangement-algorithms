"""由已保存拼图生成查看用裁图；不改变原图、预测或参数。"""
import sys,json
from pathlib import Path
import cv2,numpy as np
import a7_io as io

def main():
 run=io.HERE/sys.argv[1];target=run/'qa_crops';target.mkdir(exist_ok=True)
 for path in (run/'图像结果').glob('*/*/*.png'):
  pair=io.read_image(path);h,ww=pair.shape[:2];w=ww//2
  record=io.read_json(run/'记录'/path.parent.name/(path.stem+'.json'))
  result=record.get('result',record);points=[]
  for c in result.get('candidates',[]):
   if c['accepted'] or result['status']=='AMBIGUOUS':
    q=np.asarray(c['center_xy']);v=np.asarray(c['visible']);points+=q[v].tolist()
  if points:
   q=np.asarray(points);x0,y0=np.maximum(0,np.floor(q.min(axis=0)-110)).astype(int);x1,y1=np.minimum([w,h],np.ceil(q.max(axis=0)+111)).astype(int)
  else:x0,y0,x1,y1=0,0,w,h
  left=pair[y0:y1,x0:x1];right=pair[y0:y1,w+x0:w+x1]
  out=np.vstack([left,right]);scale=min(1.,2200/out.shape[1]);out=cv2.resize(out,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
  io.write_image(target/path.parent.name/(path.stem+'.jpg'),out)
 print('review crops complete')
if __name__=='__main__':main()
