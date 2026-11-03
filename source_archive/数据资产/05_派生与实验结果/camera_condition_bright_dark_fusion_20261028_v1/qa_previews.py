"""F1固定六图的只读视觉QA展示；不修改融合源文件或生产结果。

左右两幅采用同一原图坐标裁框，不缩放。左幅保留裁框内原图上下文，
只在D4 owner替换已经保存的D1增强，便于检查SAM裁面；右幅为实际保存
的融合结果。裁框只用于QA，生产图仍为4096x3000。
"""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import cv2
import numpy as np
from fusion_io import HERE,ASSETS,read_json,read_image,write_image,dump,sha,sources,source_paths,selection


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',default='pilot_001')
    args=ap.parse_args();run=HERE/args.run
    if run.parent!=HERE or not (run/'summary.json').exists():raise ValueError('completed direct-child run required')
    destination=run/'记录/qa_visual'
    if destination.exists():raise FileExistsError('never overwrite an existing visual QA')
    fixed_path=ASSETS/'camera_condition_dark_boundaries_20261027_v2/baseline_diagnosis.json'
    fixed=read_json(fixed_path)['fixed_visual_six']
    rows={r['output_file']:r for r in selection('pilot')}
    before=sources();entries=[]
    for i,entry in enumerate(fixed,1):
        key=entry['output_file'];row=rows[key];rel=Path(key);role=rel.parent.name;name=rel.name
        paths=source_paths(row)
        original=read_image(paths['original']);enhanced=read_image(paths['enhanced'])
        masks={k:read_image(run/'记录/masks'/k/role/name,cv2.IMREAD_GRAYSCALE)
               for k in ('surface','bright_owner','dark_owner','uncovered','c3_boundaries','d4_boundaries','merged')}
        actual_path=run/'图像结果'/role/name;actual=read_image(actual_path)
        assert original.shape==actual.shape==(3000,4096,3)
        yy,xx=np.nonzero(masks['surface']);pad=20
        x0=max(0,int(xx.min())-pad);x1=min(original.shape[1],int(xx.max())+pad+1)
        y0=max(0,int(yy.min())-pad);y1=min(original.shape[0],int(yy.max())+pad+1)
        base=original.copy();dark=masks['dark_owner']>0;base[dark]=enhanced[dark]
        left=base[y0:y1,x0:x1];right=actual[y0:y1,x0:x1]
        header=62;gap=18;h,w=left.shape[:2]
        canvas=np.zeros((h+header,2*w+gap,3),np.uint8)
        canvas[header:,:w]=left;canvas[header:,w+gap:]=right
        cv2.putText(canvas,'Input context + saved dark enhancement',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(canvas,'Saved F1 overlay (surface exterior black)',(w+gap+8,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
        label=f'{role} / {rel.stem} | same original-coordinate bbox; no resize'
        cv2.putText(canvas,label,(8,48),cv2.FONT_HERSHEY_SIMPLEX,.50,(190,190,190),1,cv2.LINE_AA)
        out=destination/f'{i:02d}_{rel.stem}_full_surface_pair.jpg';write_image(out,canvas)
        s=masks['surface']>0;b=masks['bright_owner']>0;d=masks['dark_owner']>0
        c3=masks['c3_boundaries']>0;d4=masks['d4_boundaries']>0;merged=masks['merged']>0
        checks={'saved_union_exact':bool(np.array_equal(merged,c3|d4)),
                'owners_disjoint':bool(not np.any(b&d)),
                'c3_inside_bright_owner':bool(not np.any(c3&~b)),
                'd4_inside_dark_owner':bool(not np.any(d4&~d)),
                'overlay_exterior_black':bool(not np.any(actual[~s])),
                'source_original_hash_matches_fixed_manifest':sha(paths['original'])==row['output_sha256']}
        assert all(checks.values())
        entries.append({'output_file':key,'preview':str(out.resolve()),'preview_sha256':sha(out),
            'bbox_original_xyxy':[x0,y0,x1,y1],'production_shape_hwc':list(actual.shape),
            'preview_shape_hwc':list(canvas.shape),'resize_applied':False,
            'input_paths':{k:str(paths[k]) for k in ('original','enhanced','surface')},
            'saved_output':str(actual_path.resolve()),'saved_output_sha256':sha(actual_path),
            'pixel_counts':{k:int(np.count_nonzero(v)) for k,v in masks.items()},'checks':checks})
    after=sources();assert before==after
    dump(destination/'index.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'fixed six full-surface display previews only; original-coordinate crops, production images unchanged',
        'fixed_selection_source':str(fixed_path),'fixed_selection_sha256':sha(fixed_path),
        'source_before':before,'source_after':after,'frozen_sources_unchanged':True,
        'preview_script_sha256':sha(Path(__file__)),'cases':entries,
        'visual_review_pending':True,'accuracy':None,'equipment_access':False,'reserved_access':False})
    print(str(destination/'index.json'))


if __name__=='__main__':main()
