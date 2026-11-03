"""两张已知结构失败的固定位置对照：增强图、旧D3、新D3.0.1。"""
from pathlib import Path
import cv2
import numpy as np
from data_io import HERE,D1,read_json,read_image,write_image,dump,sha
parent=HERE.parent/'camera_condition_dark_groove_20261027_v3'
out=HERE/'all_001/记录/qa_repair'
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True)
entries=[]
for item in read_json(HERE/'repair_check.json')['checks']:
    rel=Path(item['output_file']);role=rel.parent.name
    case=read_json(parent/'all_001/记录/cases'/role/(rel.stem+'.json'))
    a,b=item['failure_x_range_inclusive'];pitch=item['pitch_px']
    ys=[y for c in case['fit']['curves'] if c['path_id'] in item['original_problem_pair']
        for x,y in c['sampled_points_xy'] if a<=x<=b]
    assert ys
    box=[max(0,int(a-4*pitch)),max(0,int(min(ys)-4*pitch)),
         min(4096,int(b+4*pitch)+1),min(3000,int(max(ys)+4*pitch)+1)]
    x0,y0,x1,y1=box
    images=[read_image(D1/'all_001/图像结果/02_对比度增强ROI'/role/rel.name),
            read_image(parent/'all_001/图像结果'/role/rel.name),
            read_image(HERE/'all_001/图像结果'/role/rel.name)]
    panels=[]
    for im,title in zip(images,['D1 fixed enhanced','D3.0.0 failed spacing','D3.0.1 repair']):
        tile=np.zeros((y1-y0+30,x1-x0,3),np.uint8);tile[30:]=im[y0:y1,x0:x1]
        cv2.putText(tile,title,(3,19),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
        panels.append(tile)
    target=out/(rel.stem+'.jpg');write_image(target,np.concatenate(panels,axis=1))
    entries.append({'output_file':item['output_file'],'bbox_xyxy':box,'comparison':str(target),'sha256':sha(target)})
dump(out/'index.json',{'scope':'Two pre-existing structural failures only; no tuning','entries':entries})
print({'panels':2,'directory':str(out)})
