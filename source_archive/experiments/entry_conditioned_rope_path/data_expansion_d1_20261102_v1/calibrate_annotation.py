"""Training-only raw refinement diagnostic against historical partial centers."""
import json,hashlib
from pathlib import Path
import cv2,numpy as np
from annotation_refine import refine_segment,resample
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    ix=load(BASE/'runs/run_20261030_002/data/index.json');rows=[]
    for ident in ['dual_20260924_111500__left__f00240','dual_20260924_111143__right__f00130']:
        item=next(i for i in ix['items'] if i['id']==ident);source=load(BASE/'runs/run_20261029_001/data/label_sources'/(ident+'.json'))
        im=cv2.imdecode(np.fromfile(item['image_path'],np.uint8),cv2.IMREAD_COLOR)
        line=np.asarray(max(source['trusted_center_segments_xy'],key=len));coarse=resample(line,4.)
        q,ok,diag,samples=refine_segment(im,line,12.,max_shift=4.,minimum_strength=.8)
        # This is anchored-on-reference localization agreement, not automatic identity accuracy.
        dist=np.linalg.norm(q-coarse[:len(q)],axis=1)
        rows.append(dict(id=ident,reference_points=len(coarse),accepted_fraction=float(ok.mean()),
            accepted_median_displacement_px=float(np.median(dist[ok])) if ok.any() else None,
            accepted_p95_displacement_px=float(np.quantile(dist[ok],.95)) if ok.any() else None,
            diagnostic=diag,identity_error_rate=None,scope='historical partial center used only to initialize calibration corridor; not independent recognition'))
    (HERE/'annotation_calibration.json').write_text(json.dumps(dict(training_only=True,rows=rows,not_used_to_tune_evaluation=True),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
