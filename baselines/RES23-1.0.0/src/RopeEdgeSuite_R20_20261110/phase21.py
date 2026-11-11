"""Recognize reversals separated by an extended near-edge dwell."""
import numpy as np
from scipy.signal import find_peaks
from common20 import *

def main():
    old=read(ROOT/'motion_phases.json');videos=[]
    for item,base in zip(plan(),old['videos']):
        n=item['planned_frames'];raw=np.array([np.nan if x is None else x for x in base['raw_projected_gap_pitch']]);s=base['smooth_phase_signal'];turns=[]
        if s is not None:
            s=np.array(s);peaks,prop=find_peaks(-s,prominence=1.5,distance=80,width=10)
            for k,i in enumerate(peaks):
                lo=int(prop['left_bases'][k]);hi=int(prop['right_bases'][k]);near=(s<=s[i]+.5)
                l=i;r=i
                while l>lo and near[l-1]:l-=1
                while r<hi and near[r+1]:r+=1
                if np.isfinite(raw[lo:i]).sum()<20 or np.isfinite(raw[i+1:hi+1]).sum()<20:continue
                if s[lo]-s[i]<1.5 or s[hi]-s[i]<1.5:continue
                turns.append(dict(frame=int(i),time_s=i/item['fps'],near_dwell_frame_range=[int(l),int(r)],
                    prominence_pitch=float(prop['prominences'][k]),type='projected_reversal_with_possible_dwell',physical_layer_change_confirmed=False))
        cuts=[0]+[x['frame'] for x in turns]+[n];epochs=[]
        for k,(lo,hi) in enumerate(zip(cuts[:-1],cuts[1:])):
            inds=np.flatnonzero(np.isfinite(raw[lo:hi]))+lo;direction='undetermined'
            if len(inds)>=20:
                delta=float(np.median(raw[inds[-min(30,len(inds)):]])-np.median(raw[inds[:min(30,len(inds))]]))
                direction='approaching' if delta<-.5 else 'departing' if delta>.5 else 'nearly_stationary'
            epochs.append(dict(epoch=k,start=lo,end=hi,direction=direction))
        videos.append(dict(base,epochs=epochs,turns=turns));print(item['id'],epochs,turns,flush=True)
    dump(ROOT/'motion_phases_v2.json',dict(videos=videos,code_sha256=sha(__file__),input_sha256=sha(ROOT/'motion_phases.json')))
if __name__=='__main__':main()
