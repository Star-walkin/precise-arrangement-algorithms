"""冻结后额外12开发图的检查拼图；固定QA38索引，不挑结果。"""
from pathlib import Path
import cv2
import numpy as np
from data_io import HERE,D1,old,read_image,write_image,dump,sha,read_json

def main():
    rows,qa=old.old.select('all')
    chosen=[]
    for camera in ['left','right']:
        names=sorted(r['output_file'] for r in rows if r['camera_role']==camera and r['output_file'] in qa)
        chosen.extend(names[i] for i in [1,5,8,11,15,17])
    run=HERE/read_json(HERE/'protocol.json').get('final_run','all_001')
    out=run/'记录/qa_additional'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    tiles=[];index=[]
    for rel in chosen:
        p=Path(rel);role=p.parent.name
        mask=read_image(D1/'all_001/记录/masks/dark_roi'/role/p.name,0)
        enhanced=read_image(D1/'all_001/图像结果/02_对比度增强ROI'/role/p.name)
        result=read_image(run/'图像结果'/role/p.name)
        n,_,stats,_=cv2.connectedComponentsWithStats(mask)
        for component in sorted(range(1,n),key=lambda i:-stats[i,4]):
            x,y,w,h,area=map(int,stats[component])
            if area<10000:continue
            tile=np.zeros((510,600,3),np.uint8)
            cv2.putText(tile,p.stem[:55],(6,17),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1)
            for j,im in enumerate([enhanced,result]):
                crop=im[y:y+h,x:x+w];scale=min(588/w,235/h)
                sized=cv2.resize(crop,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
                yy=25+j*240;tile[yy:yy+sized.shape[0],6:6+sized.shape[1]]=sized
            index.append({'tile_index':len(tiles),'output_file':rel,'bbox_xyxy':[x,y,x+w,y+h],
                          'result_sha256':sha(run/'图像结果'/role/p.name)})
            tiles.append(tile)
    sheets=[]
    for start in range(0,len(tiles),6):
        sheet=np.zeros((1530,1200,3),np.uint8)
        for j,tile in enumerate(tiles[start:start+6]):
            yy=(j//2)*510;xx=(j%2)*600;sheet[yy:yy+510,xx:xx+600]=tile
        p=out/f'page_{start//6+1:02}.jpg';write_image(p,sheet);sheets.append(str(p))
    dump(out/'index.json',{'selection':chosen,'rule':'each camera sorted QA19 indices1,5,8,11,15,17; fixed before full run; no further tuning',
        'panels':'top=unchanged D1 enhancement, bottom=final overlay','tiles':index,'sheets':sheets})
    print(sheets)

if __name__=='__main__':main()
