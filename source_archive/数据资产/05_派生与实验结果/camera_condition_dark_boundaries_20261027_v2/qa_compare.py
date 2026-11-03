"""固定6张暗面分块对照JPG，供闭环目视复核；不生成HTML。"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from data_io import HERE,D1,read_json,read_image,write_image

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run');a=ap.parse_args();run=HERE/a.run
    for case in read_json(HERE/'baseline_diagnosis.json')['fixed_visual_six']:
        rel=Path(case['output_file']);role=rel.parent.name;name=rel.name
        paths=[D1/'all_001/图像结果/02_对比度增强ROI'/role/name,
               D1/'all_001/图像结果/03_C3边界结果'/role/name,
               run/'图像结果'/role/name]
        images=[read_image(p) for p in paths]
        for k,component in enumerate(case['major_dark_components']):
            x0,y0,x1,y1=component['bbox_xyxy'];tiles=[]
            for image,label in zip(images,['fixed enhanced ROI','D1 C3','D2 '+a.run]):
                patch=image[y0:y1,x0:x1];scale=min(1.,850/max(patch.shape[:2]));patch=cv2.resize(patch,(round(patch.shape[1]*scale),round(patch.shape[0]*scale)))
                tile=np.zeros((patch.shape[0]+32,patch.shape[1],3),np.uint8);tile[32:]=patch
                cv2.putText(tile,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.46,(240,240,240),1,cv2.LINE_AA);tiles.append(tile)
            write_image(run/'记录/qa_comparisons'/(rel.stem+'__part'+str(k)+'.jpg'),np.concatenate(tiles,axis=1))

if __name__=='__main__':main()
