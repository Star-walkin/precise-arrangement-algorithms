"""Classical multi-region temporal texture recurrence, no encoder or learned model.

A periodic appearance is evidence of a repeating cycle, not by itself proof
that this cycle equals 360 degrees. Motor outputs retain that qualification.
"""
import numpy as np
from scipy.signal import find_peaks

CONFIG=dict(min_period_s=.8,max_period_s=6.,history_s=18.,min_cycles=3.,
    min_peak=.12,min_prominence=.06,min_regions=2,max_region_spread_s=.3)

def normalize(features,mean=None):
    x=np.asarray(features,np.float32)
    if mean is None:mean=np.mean(x,axis=0)
    z=x-mean
    norm=np.linalg.norm(z,axis=-1,keepdims=True)
    return z/np.maximum(norm,1e-6),mean

def estimate_period(features,fps):
    x=np.asarray(features,np.float32)[-round(CONFIG['history_s']*fps):]
    out=dict(available=False,reason='insufficient_periodic_evidence',physical_revolution_confirmed=False)
    if x.ndim!=3 or x.shape[1]<CONFIG['min_regions']:return dict(out,reason='insufficient_visible_yellow_regions')
    if len(x)<max(25,round(CONFIG['min_period_s']*CONFIG['min_cycles']*fps)):return out
    if np.max(np.std(x,axis=0))<1e-4:return dict(out,reason='stationary_or_empty_texture')
    z,mean=normalize(x)
    lo=max(2,round(CONFIG['min_period_s']*fps))
    hi=min(round(CONFIG['max_period_s']*fps),len(x)//3)
    if hi<=lo+2:return out
    lags=np.arange(lo,hi+1);curves=[]
    for region in range(z.shape[1]):
        curves.append([float(np.mean(np.sum(z[k:,region]*z[:-k,region],axis=1))) for k in lags])
    scores=np.array(curves);region_peaks=[]
    for region,y in enumerate(scores):
        ix,props=find_peaks(y,height=CONFIG['min_peak'],prominence=CONFIG['min_prominence'],distance=max(2,round(.4*fps)))
        for j in ix:region_peaks.append(dict(region=region,lag=int(lags[j]),score=float(y[j])))
    clusters=[]
    for lag in sorted(set(q['lag'] for q in region_peaks)):
        hits=[]
        for region in range(z.shape[1]):
            candidates=[q for q in region_peaks if q['region']==region and abs(q['lag']-lag)<=CONFIG['max_region_spread_s']*fps]
            if candidates:hits.append(max(candidates,key=lambda q:q['score']))
        if len(hits)<CONFIG['min_regions']:continue
        center=float(np.median([q['lag'] for q in hits]))
        if np.ptp([q['lag'] for q in hits])>CONFIG['max_region_spread_s']*fps:continue
        clusters.append(dict(lag=center,score=float(np.mean([q['score'] for q in hits])),hits=hits))
    out.update(region_peaks=region_peaks,history_frames=len(x))
    if not clusters:return out
    best=max(q['score'] for q in clusters)
    # Prefer the first well-supported recurrence rather than its stronger multiple.
    candidates=[q for q in clusters if q['score']>=max(CONFIG['min_peak'],.55*best)]
    q=min(candidates,key=lambda q:q['lag']);hits=q['hits'];lag=q['lag']
    half=max(1.,float(np.ptp([v['lag'] for v in hits]))/2)
    out.update(available=True,reason=None,period_s=lag/fps,period_interval_s=[(lag-half)/fps,(lag+half)/fps],
        regions=[v['region'] for v in hits],score=q['score'],region_support=len(hits),
        supported_cycles=len(x)/lag,semantics='periodic_appearance_cycle_not_independent_360_degree_measurement')
    return out
