"""固定pilot每图左右两处接缝，原始灰度归一化/F1/F2等比例对照。"""
import sys
from pathlib import Path
import cv2,numpy as np
from seam_io import HERE,F1,read_json,read_image,write_image,dump

def main():
    run=HERE/sys.argv[1]; entries=[]
    for i,rel in enumerate(read_json(run/'run_context.json')['selection'],1):
        p=Path(rel);role=p.parent.name;name=p.name
        case=read_json(run/'记录/cases'/role/(p.stem+'.json'))
        old=read_image(F1/'all_001/图像结果'/role/name,1)
        new=read_image(run/'图像结果'/role/name,1)
        normpath=run/'重叠区增强'/role/name
        if normpath.exists(): norm=read_image(normpath,1)
        else:
            from seam_io import selection,inputs
            from seam_observation import detect_overlap
            data,_=inputs(next(r for r in selection() if r['output_file']==rel))
            norm=detect_overlap(data['original'],data['surface'],data['bright'],data['dark'],data['pitch'],data['c3_record'],data['c3_exclusion'])['normalized_bgr']
        strips=[];rects=[]
        for side in (-1,1):
            groups=[c for c in case['new_groups'] if c['side']==side and c.get('sampled_points_xy')]
            if not groups: continue
            centers=np.array([np.mean(c['anchor_points_xy'],axis=0) for c in groups])
            cx,cy=np.median(centers,axis=0);x=max(0,min(4096-420,int(cx)-210));y=max(0,min(3000-510,int(cy)-255))
            tiles=[]
            for title,im in [('Normalized (same image)',norm),('F1 before',old),('F2 after',new)]:
                tile=im[y:y+510,x:x+420].copy()
                tile=cv2.copyMakeBorder(tile,30,0,0,0,cv2.BORDER_CONSTANT,value=(24,24,24))
                cv2.putText(tile,title,(8,21),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
                tiles.append(tile)
            strips.append(np.hstack(tiles));rects.append({'side':side,'xywh':[x,y,420,510]})
        if strips:
            dest=run/'记录/qa_visual'/f'{i:02d}_{p.stem}_seams.jpg';write_image(dest,np.vstack(strips))
            entries.append({'image':rel,'preview':str(dest),'rectangles':rects})
    dump(run/'记录/qa_visual/index.json',entries)
if __name__=='__main__':main()
