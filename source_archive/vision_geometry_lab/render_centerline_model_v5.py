"""Render held-out model probabilities and separate human references, offline."""
from pathlib import Path
import argparse
import json

import cv2
import numpy as np

from .geometry import read_image, write_image, dump_json
from .prepare_centerline_v5 import OUTPUT, file_hash
from .centerline_benchmark_v5 import load_prediction
from .centerline_quality_v5 import load_label_quality
from .centerline_review_v5 import validate_annotation


def panel(pixels, title, width=720, height=400):
    scale=min(width/pixels.shape[1],height/pixels.shape[0])
    resized=cv2.resize(pixels,(round(pixels.shape[1]*scale),round(pixels.shape[0]*scale)))
    canvas=np.full((height+36,width,3),20,np.uint8)
    y=36+(height-resized.shape[0])//2;x=(width-resized.shape[1])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    cv2.putText(canvas,title,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),1,cv2.LINE_AA)
    return canvas


def heat_overlay(crop, probability):
    probability=np.asarray(probability,np.float32)
    if probability.ndim!=2 or not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):
        raise ValueError('Expected finite HxW probabilities in [0,1]')
    color=cv2.applyColorMap(np.rint(probability*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    color=cv2.resize(color,(crop.shape[1],crop.shape[0]))
    # Fixed 0..1 color scale, no per-image min/max stretching or threshold tuning.
    return cv2.addWeighted(crop,.35,color,.65,0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=OUTPUT/'manifest.json')
    parser.add_argument('--predictions-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rows=[]
    manifest=json.loads(args.manifest.read_text('utf-8'))
    for sample in manifest['samples']:
        if sample['split']!='holdout':continue
        label=validate_annotation(json.loads(Path(sample['annotation_path']).read_text('utf-8')),sample)
        loaded=load_prediction(args.predictions_dir,sample,label)
        if loaded is None:continue
        meta,body,center=loaded
        if file_hash(sample['image_path'])!=sample['image_sha256']:
            raise ValueError('Changed source image')
        frame=read_image(sample['image_path']);x0,y0,x1,y1=sample['roi_xyxy']
        crop=frame[y0:y1,x0:x1].copy();reference=crop.copy()
        colors={'centerline':(80,255,80),'edge_flange':(200,100,240),
                'edge_surface':(255,220,50),'free_seed':(0,170,255)}
        for key,points in label['polylines'].items():
            if len(points)>=2:
                p=np.rint(np.asarray(points)-[x0,y0]).astype(np.int32)
                cv2.polylines(reference,[p],False,colors[key],3,cv2.LINE_AA)
        quality=load_label_quality(sample)
        for poly in [*label['ignore_polygons'],*quality['precision_exclude_polygons']]:
            p=np.rint(np.asarray(poly)-[x0,y0]).astype(np.int32)
            cv2.polylines(reference,[p],True,(255,0,255),4,cv2.LINE_AA)
        top=np.hstack([panel(crop,'CLEAN ROI | no labels'),
                       panel(reference,'HUMAN partial reference | orange: input seed')])
        bottom=np.hstack([panel(heat_overlay(crop,body),'BODY probability | fixed scale 0..1, NOT accepted mask'),
                          panel(heat_overlay(crop,center),'CENTER probability | fixed scale 0..1, NOT accepted path')])
        figure=np.vstack([top,bottom])
        path=args.output/(sample['sample_id']+'.jpg');write_image(path,figure)
        rows.append(dict(sample_id=sample['sample_id'],figure=str(path.resolve()),
                         inference_ms=meta['inference_ms'],
                         body_probability_quantiles=np.quantile(body,[0,.5,.95,1]).tolist(),
                         center_probability_quantiles=np.quantile(center,[0,.5,.95,1]).tolist()))
    dump_json(args.output/'summary.json',dict(rows=rows,model_input_uses_reference_center_or_edges=False,
                human_reference_used_for_display_only=True,field_accuracy_established=False,
                hardware_access=False,force_command=None))
    print(json.dumps(dict(figures=len(rows),output=str(args.output)),ensure_ascii=False))


if __name__=='__main__':main()
