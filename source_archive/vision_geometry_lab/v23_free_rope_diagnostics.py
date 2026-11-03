"""Post-hoc diagnostics of archived V23 failures. Does not run/modify a detector."""
from pathlib import Path
import json
import cv2
import numpy as np
from vision_geometry_lab.v23_free_rope_audit import read, write, sha, sample_polyline, read_image, BASE, INVENTORY


def run():
    out=BASE/'posthoc_failure_diagnostics'
    if out.exists(): raise FileExistsError(out)
    report=read(BASE/'report.json')
    samples={s['sample_id']:s for s in read(INVENTORY)['samples']}
    rows=[]
    for row in report['rows']:
        if row['metrics'].get('line_fit_pass') is not False: continue
        sid=row['sample_id']; sample=samples[sid]; pred=row['prediction']
        if sha(sample['image_path'])!=sample['image_sha256'] or sha(sample['annotation_path'])!=sample['annotation_sha256']:
            raise ValueError('Changed source')
        seed=np.asarray(read(sample['annotation_path'])['polylines']['free_seed'],float)
        pts,length=sample_polyline(seed)
        image=read_image(sample['image_path'])
        record=dict(sample_id=sid,detector_valid=pred.get('valid'),reason=pred.get('reason'),
                    fractions=[.1,.5,.9],posthoc_only=True,image_sha256=sample['image_sha256'],
                    annotation_sha256=sample['annotation_sha256'])
        if pred.get('valid'):
            n=np.asarray(pred['normal']);rho=float(pred['rho_px']); signed=pts@n-rho
            d=np.array([-n[1],n[0]]);limits=sorted(np.asarray(pred['observed_segment'])@d)
            inside=(pts@d>=limits[0])&(pts@d<=limits[1])
            stats=lambda x:dict(count=len(x),signed_median=float(np.median(x)),abs_p95=float(np.quantile(abs(x),.95))) if len(x) else None
            w=pred.get('width_px')
            record.update(signed_error_all=stats(signed),error_within_observed_projection=stats(signed[inside]),
                error_outside_observed_projection=stats(signed[~inside]),
                reference_arclength_px=length,optical_width_px=w,
                optical_width_interval_px=pred.get('width_interval_px'),
                width_kind=pred['diagnostics'].get('width_kind'),width_status=pred['diagnostics'].get('width_status'),
                line_support_kind=pred['diagnostics'].get('line_support_kind'),
                five_px_fraction_of_reported_optical_width=5/w if w else None,
                human_center_is_not_absolute_pixel_truth=True,
                points=[])
        panels=[[],[]]
        for fraction in [.1,.5,.9]:
            point=pts[round(fraction*(len(pts)-1))]
            x,y=np.rint(point).astype(int);x0=max(0,min(image.shape[1]-320,x-160));y0=max(0,min(image.shape[0]-144,y-72))
            raw=image[y0:y0+144,x0:x0+320].copy();marked=raw.copy()
            poly=np.rint(seed-[x0,y0]).astype(np.int32)
            cv2.polylines(marked,[poly],False,(255,255,0),1,cv2.LINE_AA)
            if pred.get('valid'):
                q=point-(point@n-rho)*n
                ends=np.rint(np.array([q-400*d,q+400*d])-[x0,y0]).astype(int)
                cv2.line(marked,tuple(ends[0]),tuple(ends[1]),(0,140,255),1,cv2.LINE_AA)
                record['points'].append(dict(fraction=fraction,GT_xy=point.tolist(),signed_error_px=float(point@n-rho),
                    inside_observed_projection=bool(limits[0]<=point@d<=limits[1])))
            for j,panel in enumerate([raw,marked]):
                panel=cv2.resize(panel,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
                header=np.zeros((26,panel.shape[1],3),np.uint8)
                cv2.putText(header,f'{fraction:.0%} of human free segment; '+('RAW' if j==0 else 'cyan GT / orange automatic LINE'),(5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1)
                panels[j].append(np.vstack([header,panel]))
        canvas=np.vstack([np.hstack(panels[0]),np.hstack(panels[1])])
        path=out/(sid+'.jpg');path.parent.mkdir(parents=True,exist_ok=True)
        ok,data=cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,96]);assert ok
        with path.open('xb') as f:f.write(data.tobytes())
        record['diagnostic_crop_path']=str(path);rows.append(record)
    write(out/'diagnostics.json',dict(status='complete',report_sha256=sha(BASE/'report.json'),
          diagnostic_source_sha256=sha(__file__),uses_scoring_annotations_after_inference=True,
          detector_rerun=False,threshold_or_score_changed=False,rows=rows))
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__':run()
