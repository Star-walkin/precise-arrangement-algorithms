"""Offline motion phases for validation, not causal direction ground truth."""
import numpy as np
from scipy.ndimage import median_filter,gaussian_filter1d
from scipy.signal import find_peaks
from common20 import *
from geometry20 import boundary_at
from tangent_column import roots_for_curve

def phases(item):
    n=item['planned_frames'];values=np.full(n,np.nan)
    for i in range(n):
        r=record(item['id'],i);t=r['track'];b=r['boundary']
        if not t.get('valid') or not b.get('fit_valid') or len(b.get('top_curve_xy',[]))<30:continue
        root=roots_for_curve(b['top_curve_xy'],t['line_abc'],PITCH_PRIOR)
        if root:values[i]=root['gap']/PITCH_PRIOR
    ii=np.flatnonzero(np.isfinite(values));turns=[];smooth=None
    if len(ii)>=80:
        # Filled values serve ONLY for broad phase hypotheses, never counts.
        filled=np.interp(np.arange(n),ii,values[ii]);smooth=gaussian_filter1d(median_filter(filled,size=15),7.)
        mins,props=find_peaks(-smooth,prominence=1.5,distance=80,width=10)
        for k,i in enumerate(mins):
            lo=max(0,i-50);hi=min(n,i+51)
            if np.isfinite(values[lo:i]).sum()<20 or np.isfinite(values[i+1:hi]).sum()<20:continue
            if smooth[lo]-smooth[i]<.75 or smooth[hi-1]-smooth[i]<.75:continue
            turns.append(dict(frame=int(i),time_s=i/item['fps'],prominence_pitch=float(props['prominences'][k]),
                type='sustained_projected_approach_to_departure',physical_layer_change_confirmed=False))
    cuts=[0]+[x['frame'] for x in turns]+[n];epochs=[]
    for k,(lo,hi) in enumerate(zip(cuts[:-1],cuts[1:])):
        valid=ii[(ii>=lo)&(ii<hi)];direction='undetermined'
        if len(valid)>=20:
            delta=float(np.median(values[valid[-min(30,len(valid)):]])-np.median(values[valid[:min(30,len(valid))]]))
            direction='approaching' if delta<-.5 else 'departing' if delta>.5 else 'nearly_stationary'
        epochs.append(dict(epoch=k,start=lo,end=hi,direction=direction))
    return dict(id=item['id'],epochs=epochs,turns=turns,raw_projected_gap_pitch=[None if not np.isfinite(v) else float(v) for v in values],
        smooth_phase_signal=None if smooth is None else smooth.tolist(),offline_phase_hypotheses=True,physical_layer_or_reversal_truth=False)

if __name__=='__main__':
    videos=[]
    for item in plan():
        r=phases(item);videos.append(r);print(item['id'],r['epochs'],flush=True)
    dump(ROOT/'motion_phases.json',dict(videos=videos,code_sha256=sha(__file__)))
