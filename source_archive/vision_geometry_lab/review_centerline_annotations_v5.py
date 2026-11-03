"""Render human-label review sheets only; never alter labels or train a model."""
from pathlib import Path
import argparse
import json

import cv2
import numpy as np

from .geometry import read_image, write_image, dump_json
from .prepare_centerline_v5 import OUTPUT, file_hash
from .centerline_review_v5 import validate_annotation
from .centerline_quality_v5 import load_label_quality


COLORS={'centerline':(60,255,90),'edge_flange':(220,100,255),
        'edge_surface':(255,220,50),'free_seed':(40,180,255)}


def crop(frame,roi,width=1280):
    x0,y0,x1,y1=roi
    patch=frame[y0:y1,x0:x1]
    return cv2.resize(patch,(width,round(patch.shape[0]*width/patch.shape[1])))


def banner(frame,text):
    result=cv2.copyMakeBorder(frame,40,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
    cv2.putText(result,text,(12,27),cv2.FONT_HERSHEY_SIMPLEX,.64,(255,255,255),1,cv2.LINE_AA)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUTPUT.parent/'review_round1')
    args=parser.parse_args()
    manifest=json.loads((OUTPUT/'manifest.json').read_text('utf-8'))
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sample in manifest['samples']:
        annotation=Path(sample['annotation_path'])
        if not annotation.exists(): continue
        label=validate_annotation(json.loads(annotation.read_text('utf-8')),sample)
        quality=load_label_quality(sample)
        if file_hash(sample['image_path'])!=sample['image_sha256']:
            raise ValueError('Clean image mismatch')
        frame=read_image(sample['image_path']);overlay=frame.copy()
        for key,points in label['polylines'].items():
            p=np.rint(points).astype(np.int32)
            if len(p)<2: continue
            cv2.polylines(overlay,[p],False,COLORS[key],2,cv2.LINE_AA)
            for i,point in enumerate(p):
                cv2.circle(overlay,tuple(point),4,COLORS[key],-1)
                if key=='centerline':
                    cv2.putText(overlay,str(i+1),(int(point[0])+6,int(point[1])-8),
                                cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
        for polygon in quality['precision_exclude_polygons']:
            p=np.rint(polygon).astype(np.int32)
            tint=overlay.copy()
            cv2.fillPoly(tint,[p],(180,60,180))
            overlay=cv2.addWeighted(overlay,.85,tint,.15,0)
            cv2.polylines(overlay,[p],True,(230,80,230),3,cv2.LINE_AA)
        panels=[banner(crop(frame,sample['roi_xyxy']),'CLEAN CONTEXT | no reference labels'),
                banner(crop(overlay,sample['roi_xyxy']),'HUMAN LABELS | green center / pink-cyan edges / purple BOX: precision EXCLUDED')]
        write_image(out/(sample['sample_id']+'_context.jpg'),np.vstack(panels))
        center=np.asarray(label['polylines']['centerline'],float)
        # Last three manual center points for magnification; these are diagnostic
        # crop selections, not algorithm inputs or a blind performance experiment.
        if len(center)>=3:
            p=center[-3:]
            lo=np.maximum([0,0],np.floor(p.min(axis=0)-[100,80])).astype(int)
            hi=np.minimum([frame.shape[1],frame.shape[0]],np.ceil(p.max(axis=0)+[100,80])).astype(int)
            roi=[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]
            detail=np.vstack([banner(crop(frame,roi,1000),'CLEAN TERMINAL DETAIL | enlarged, no inference'),
                              banner(crop(overlay,roi,1000),'HUMAN LABELS | purple BOX: context/coarse only, NOT precision truth')])
            write_image(out/(sample['sample_id']+'_terminal.jpg'),detail)
        rows.append(dict(sample_id=sample['sample_id'],status=label['status'],
                         point_counts={k:len(v) for k,v in label['polylines'].items()},
                         label_sha256=file_hash(annotation),source_image_sha256=sample['image_sha256'],
                         quality_sha256=quality['quality_sha256'],
                         quality_regions=quality['regions'],
                         geometry_review_is_not_identity_confirmation=True))
    dump_json(out/'render_review.json',dict(reviewed_files=len(rows),rows=rows,
                   labels_modified=False,model_training=False,hardware_access=False))
    print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__': main()
