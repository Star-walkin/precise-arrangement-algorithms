"""冻结400图后额外12图同坐标对照：D1增强、D3.0.1、D4。

只读既有开放QA19中预定序号，不按结果好坏挑样；不读取保留集。
必须在all运行完成且源码与freeze_all.json一致后才解码图像。
用法：python qa_additional.py all_001
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
from data_io import HERE, D1, D3, old, read_json, read_image, write_image, dump, sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    args=ap.parse_args()
    run=args.run.absolute() if args.run.is_absolute() else (HERE/args.run).absolute()
    if run.parent!=HERE or not run.is_dir():
        raise ValueError('existing direct child run required')
    context=read_json(run/'run_context.json')
    summary=read_json(run/'summary.json')
    frozen=read_json(HERE/'freeze_all.json')
    # 以下守门发生在任何read_image之前；脚本不能提前打开额外开发图调参。
    if context.get('mode')!='all' or len(context.get('selection',[]))!=400:
        raise ValueError('only frozen400 all run is eligible')
    if context.get('source_sha256')!=frozen.get('source_sha256'):
        raise ValueError('run sources differ from frozen sources')
    if any(summary.get(k)!=v for k,v in {'planned':400,'completed':400,'failed':0,'exit_code':0}.items()):
        raise ValueError('complete400 predictions required before additional QA')
    rows,qa=old.old.old.select('all')
    chosen=[]
    for camera in ('left','right'):
        names=sorted(r['output_file'] for r in rows if r['camera_role']==camera and r['output_file'] in qa)
        if len(names)!=19: raise ValueError('fixed QA19 selection changed')
        chosen.extend(names[i] for i in (1,5,8,11,15,17))
    if not set(chosen).issubset(context['selection']):
        raise ValueError('additional QA images not part of frozen400')
    out=run/'记录/qa_additional'
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    cv2.setNumThreads(1)
    entries=[]
    for rel in chosen:
        p=Path(rel); role=p.parent.name
        mask_path=D1/'all_001/记录/masks/dark_roi'/role/p.name
        paths=[D1/'all_001/图像结果/02_对比度增强ROI'/role/p.name,
               D3/'all_001/图像结果'/role/p.name,
               run/'图像结果/04_边界结果'/role/p.name]
        mask=read_image(mask_path,0)
        images=[read_image(path) for path in paths]
        if any(image.shape[:2]!=mask.shape for image in images):
            raise ValueError('comparison shapes differ: '+rel)
        n,_,stats,_=cv2.connectedComponentsWithStats(mask)
        for k,component in enumerate(sorted(range(1,n),key=lambda i:-stats[i,4])):
            x,y,w,h,area=map(int,stats[component])
            if area<10000: continue
            factor=min(1.,800/max(w,h)); ww,hh=max(1,round(w*factor)),max(1,round(h*factor))
            panels=[]
            for image,title in zip(images,('D1 fixed enhancement','D3.0.1','D4 frozen')):
                tile=np.zeros((hh+30,ww,3),np.uint8)
                tile[30:]=cv2.resize(image[y:y+h,x:x+w],(ww,hh),interpolation=cv2.INTER_AREA)
                cv2.putText(tile,title,(4,20),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                panels.append(tile)
            target=out/(p.stem+f'__part{k}.jpg')
            write_image(target,np.concatenate(panels,axis=1))
            entries.append({'output_file':rel,'bbox_xyxy':[x,y,x+w,y+h],
                'component_area_px':area,'original_wh':[int(mask.shape[1]),int(mask.shape[0])],
                'display_scale':factor,'panel_wh':[ww,hh+30],
                'comparison':str(target),'sha256':sha(target),
                'input_sha256':{str(path):sha(path) for path in [mask_path,*paths]}})
    dump(out/'index.json',{'version':context['version'],'selection':chosen,'selected_count':len(chosen),
        'rule':'each camera sorted existing QA19 indices1,5,8,11,15,17; fixed before final run; no further tuning',
        'freeze_sha256':sha(HERE/'freeze_all.json'),'run_context_sha256':sha(run/'run_context.json'),
        'meaning':'Qualitative same-coordinate review on open development images, not a new independent holdout or accuracy score.',
        'equipment_access':False,'reserved_access':False,'entries':entries})
    print({'selected':len(chosen),'panels':len(entries),'directory':str(out)})


if __name__=='__main__': main()
