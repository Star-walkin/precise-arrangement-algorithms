"""F3只读四列展示：原图/F2/全卷面D4候选/F3，两行旧接缝框。

不导入检测器，不重跑归一化/拟合，不按F3结果选择裁框。
默认只接受run_context内固定12图；全400运行必须显式--selection fixed6。
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
ASSETS=HERE.parent
F2=ASSETS/'camera_condition_seam_joint_20261028_v2'
DATA=ASSETS/'camera_condition_images_20261009_v2'
OLD_FIXED=F2/'pilot_002/记录/qa_original_fixed6/index.json'
OLD_PILOT=F2/'pilot_001/记录/qa_visual/index.json'


def read_json(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_image(path,gray=False):
    im=cv2.imdecode(np.fromfile(str(path),np.uint8),0 if gray else 1)
    if im is None:raise ValueError('decode failed: '+str(path))
    return im


def write_new_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def old_boxes():
    mapping={}
    for path in (OLD_PILOT,OLD_FIXED):
        if not path.exists():continue
        data=read_json(path);entries=data['cases'] if isinstance(data,dict) else data
        for entry in entries:
            key=entry.get('output_file',entry.get('image'))
            mapping[key]={'rectangles':entry['rectangles'],'box_source':str(path),
                          'box_source_sha256':sha(path),'box_source_type':'reused_old_F2_fixed_frame'}
    return mapping


def roi_boxes(rel):
    """仅旧ROI几何回退：中间50%亮ROI行的亮边/暗内缘中点；预先固定规则。"""
    p=Path(rel);role=p.parent.name
    bp=ASSETS/'camera_condition_grouped_conic_20261026_v6/all_001/roi/reference_masks'/role/p.name
    dp=ASSETS/'camera_condition_dark_roi_20261026_v1/all_001/记录/masks/dark_roi'/role/p.name
    b=read_image(bp,True)>0;d=read_image(dp,True)>0;yy,xx=np.nonzero(b)
    if not len(xx):return {'rectangles':[],'box_source_type':'no_old_bright_roi_no_guessed_box'}
    middle=float(np.median(xx));cy=float(np.median(yy));rows=np.unique(yy)
    rows=rows[(rows>=np.percentile(rows,25))&(rows<=np.percentile(rows,75))]
    rects=[]
    for side in (-1,1):
        centers=[]
        for row in rows:
            bb=np.flatnonzero(b[row]);dd=np.flatnonzero(d[row])
            bb=bb[bb<middle] if side<0 else bb[bb>=middle]
            dd=dd[dd<middle] if side<0 else dd[dd>=middle]
            if not len(bb):continue
            bright=float(bb.min() if side<0 else bb.max())
            dark=float(dd.max() if side<0 else dd.min()) if len(dd) else bright
            centers.append(.5*(bright+dark))
        if not centers:continue
        cx=float(np.median(centers))
        rects.append({'side':side,'xywh':[max(0,min(3676,int(round(cx))-210)),
                       max(0,min(2490,int(round(cy))-255)),420,510]})
    return {'rectangles':rects,'box_source_type':'frozen_old_ROI_geometry_rule',
            'box_source_masks':{str(bp):sha(bp),str(dp):sha(dp)}}


def complete_old_boxes(rel,mapping):
    if rel not in mapping:return roi_boxes(rel)
    entry=dict(mapping[rel])
    rects=[{**r,'box_source_type':'reused_old_F2_fixed_frame'} for r in entry['rectangles']]
    missing={-1,1}-{r['side'] for r in rects}
    if missing:
        fallback=roi_boxes(rel)
        rects.extend({**r,'box_source_type':'frozen_old_ROI_geometry_missing_side'}
                     for r in fallback['rectangles'] if r['side'] in missing)
        entry['supplemental_box_source_masks']=fallback.get('box_source_masks',{})
        entry['missing_old_sides_filled_from_old_ROI']=sorted(missing)
    entry['rectangles']=sorted(rects,key=lambda r:r['side'])
    return entry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',help='existing direct-child run, e.g. pilot_001')
    parser.add_argument('--selection',choices=('pilot12','fixed6'),default='pilot12')
    parser.add_argument('--prepare-only',action='store_true',help='save box receipt before reading F3 images')
    parser.add_argument('--output-name',default='qa_surface_visual')
    args=parser.parse_args();run=(HERE/args.run).resolve()
    if run.parent!=HERE or not (run/'run_context.json').exists():raise ValueError('existing direct-child run_context required')
    if Path(args.output_name).name!=args.output_name:raise ValueError('output-name must be one folder name')
    context=read_json(run/'run_context.json');selection=context['selection']
    fixed={e['output_file'] for e in read_json(OLD_FIXED)['cases']}
    if args.selection=='fixed6':selection=[rel for rel in selection if rel in fixed]
    elif len(selection)>12:raise ValueError('refuse default400 loop; use --selection fixed6 explicitly')
    if args.selection=='fixed6' and len(selection)!=6:raise ValueError('fixed6 not complete in run selection')
    target=run/'记录'/args.output_name;target.mkdir(parents=True,exist_ok=True)
    boxes_path=target/'boxes.json'
    if boxes_path.exists():
        box_receipt=read_json(boxes_path)
        if box_receipt['selection']!=selection:raise ValueError('existing box selection differs')
    else:
        mapping=old_boxes();cases=[{'output_file':rel,**complete_old_boxes(rel,mapping)} for rel in selection]
        box_receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'selection':selection,
            'run_context_sha256':sha(run/'run_context.json'),'source_rule':'old F2 frames first; frozen ROI geometry fallback; never F3 output based',
            'cases':cases,'f3_images_read_for_box_selection':False}
        write_new_json(boxes_path,box_receipt)
    if args.prepare_only:
        print(str(boxes_path));return
    index_path=target/'index.json'
    if index_path.exists():raise FileExistsError('never overwrite completed preview index')
    frozen_paths={p:sha(p) for p in context['source_sha256'] if Path(p).exists()}
    entries=[]
    for i,entry in enumerate(box_receipt['cases'],1):
        rel=Path(entry['output_file']);role=rel.parent.name;name=rel.name
        files=[DATA/rel,F2/'all_001/图像结果'/role/name,
            run/'图像结果/01_D4全卷面候选'/role/name,run/'图像结果/02_F3融合边界'/role/name]
        missing=[str(p) for p in files if not p.exists()]
        if missing:raise FileNotFoundError('prediction files not ready: '+str(missing))
        images=[read_image(path) for path in files]
        if any(im.shape!=(3000,4096,3) for im in images):raise ValueError('expected original4096x3000 images')
        strips=[]
        for rect in entry['rectangles']:
            x,y,w,h=map(int,rect['xywh']);tiles=[]
            if not(0<=x<x+w<=4096 and 0<=y<y+h<=3000):raise ValueError('old crop out of bounds')
            for title,im in zip(('Original RGB','F2 saved','D4 full-surface proposals','F3 fused'),images):
                tile=cv2.copyMakeBorder(im[y:y+h,x:x+w],42,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
                side='left seam' if rect['side']<0 else 'right seam'
                cv2.putText(tile,title+' | '+side,(7,18),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(tile,f'x={x}, y={y}; original pixels',(7,35),cv2.FONT_HERSHEY_SIMPLEX,.36,(180,180,180),1,cv2.LINE_AA)
                tiles.append(tile)
            strips.append(np.hstack(tiles))
        if not strips:
            entries.append({**entry,'status':'no_old_box','actually_viewed':False});continue
        preview=target/f'{i:02d}_{rel.stem}_surface_compare.jpg'
        if preview.exists():raise FileExistsError(preview)
        ok,buffer=cv2.imencode('.jpg',np.vstack(strips),[cv2.IMWRITE_JPEG_QUALITY,96])
        if not ok:raise RuntimeError('preview encoding failed')
        buffer.tofile(str(preview))
        entries.append({**entry,'preview':str(preview),'preview_sha256':sha(preview),
            'input_sha256':{str(path):sha(path) for path in files},'resize':False,
            'columns':['original','F2_saved','D4_full_surface_proposals','F3_fused'],'actually_viewed':False})
    unchanged=all(sha(path)==value for path,value in frozen_paths.items())
    if not unchanged:raise AssertionError('algorithm sources changed during preview generation')
    write_new_json(index_path,{'created_utc':datetime.now(timezone.utc).isoformat(),'run':args.run,
        'selection_mode':args.selection,'box_receipt':str(boxes_path),'box_receipt_sha256':sha(boxes_path),
        'cases':entries,'algorithm_sources_unchanged':True,'detector_runs':0,'images_resized':False,
        'accuracy':None,'visual_review_pending':True})
    print(str(index_path))


if __name__=='__main__':main()
