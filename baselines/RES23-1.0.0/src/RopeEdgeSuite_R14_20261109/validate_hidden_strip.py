"""Predict withheld current detector anchors from other anchors + past shapes.
This evaluates reconstruction consistency, NOT physical boundary ground truth.
"""
import copy,time
import cv2,numpy as np
from common import *
from boundary_bg7 import Boundary
IDS=[141,181,221,261,461,501,541,581]
def main():
    cv2.setNumThreads(1);run=ROOT/'boundary_final';cap=cv2.VideoCapture(read(RUN12/'binding.json')['source']);rows=[]
    for i in IDS:
        while not (run/'frames'/f'{i:06d}.json.gz').exists():time.sleep(.2)
        cap.set(1,i);ok,im=cap.read();assert ok;r=record(i);prior=record(i-1,run);current=record(i,run)
        b=Boundary();b.shapes=copy.deepcopy(prior.get('shape_state',[]));b.last_shape_frame=max((s['frame'] for s in b.shapes),default=None)
        n=b.step(im,r['region_BG5'],r['S1_tracker_full'],i,forced_exclude=[[D-180,D+180]])
        pts=np.asarray(current.get('top_observed_xy',[]),float).reshape(-1,2)
        held=pts[abs(pts[:,0]-D)<180]
        eligible=bool(len(held)>=20 and current.get('d_locally_observed') and not current.get('d_foreground_occluded'))
        item=dict(frame=i,eligible_reference=eligible,withheld_points=len(held),prediction_valid=n['fit_valid'],
            reasons=n['reasons'],no_current_withheld_points_in_shape_learning=True)
        if n.get('top_curve_xy') and len(held):
            c=np.asarray(n['top_curve_xy']);err=np.abs(held[:,1]-np.interp(held[:,0],c[:,0],c[:,1]))
            item.update(median_abs_error_px=float(np.median(err)),p90_abs_error_px=float(np.quantile(err,.9)),
                maximum_abs_error_px=float(err.max()))
        rows.append(item)
    cap.release();dump(ROOT/'hidden_strip_validation.json',dict(frames=IDS,rows=rows,
        reference='withheld current BG5 candidate points surviving independent current-frame rejection; not hand labels',
        interpretation='tests reconstructing a 360px missing strip using previous shapes and current external anchors; no physical accuracy claim'))
    print(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
