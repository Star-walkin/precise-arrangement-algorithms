"""固定6图同坐标对照和形态学阶段检查；不生成HTML。"""
from pathlib import Path
import sys,cv2
import numpy as np
from data_io import HERE,D1,D2,D3,read_json,read_image,write_image,dump,sha
run=HERE/sys.argv[1];out=run/'记录'/(sys.argv[2] if len(sys.argv)>2 else 'qa_comparisons')
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True);entries=[]
for item in read_json(D2/'baseline_diagnosis.json')['fixed_visual_six']:
    p=Path(item['output_file']);role=p.parent.name
    if not (run/'图像结果/04_边界结果'/role/p.name).exists():continue
    ims=[read_image(D1/'all_001/图像结果/02_对比度增强ROI'/role/p.name),
      read_image(D3/'all_001/图像结果'/role/p.name),
      read_image(run/'图像结果/03_新黄点'/role/p.name),
      read_image(run/'图像结果/04_边界结果'/role/p.name)]
    stages=[read_image(run/'图像结果/01_阴影二值化'/role/p.name),
            read_image(run/'图像结果/02_去细股阴影'/role/p.name),
            cv2.applyColorMap(read_image(run/'记录/fields/body_barrier'/role/p.name,0),cv2.COLORMAP_TURBO)]
    for k,c in enumerate(item['major_dark_components']):
        x0,y0,x1,y1=c['bbox_xyxy'];scale=min(1.,750/max(x1-x0,y1-y0));w,h=round((x1-x0)*scale),round((y1-y0)*scale)
        paths=[]
        for group,titles,suffix in [(ims,['Fixed enhancement','D3.0.1','D4 new yellow','D4 accepted curves'],'compare'),(stages,['Raw dynamic shadow','Opened width filter','Body barrier'],'stages')]:
            tiles=[]
            for im,title in zip(group,titles):
                tile=np.zeros((h+30,w,3),np.uint8);tile[30:]=cv2.resize(im[y0:y1,x0:x1],(w,h),interpolation=cv2.INTER_AREA)
                cv2.putText(tile,title,(3,20),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);tiles.append(tile)
            target=out/(p.stem+f'__part{k}_{suffix}.jpg');write_image(target,np.concatenate(tiles,axis=1));paths.append(str(target))
        entries.append({'output_file':item['output_file'],'bbox_xyxy':[x0,y0,x1,y1],'paths':paths,'sha256':{x:sha(x) for x in paths}})
dump(out/'index.json',{'fixed_images':6,'components':entries,'meaning':'Same original-coordinate component boxes; qualitative stage comparison only'})
