"""Read-only diagnostics for R22 early first-turn candidate."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parent
R22=ROOT.parent/'RopeEdgeSuite_R22_20261111'
sys.path.append(str(R22))
from common22 import read,dump,plan

def slope(rows):
    if len(rows)<10:return None
    t=np.array([r['time_s'] for r in rows]);n=np.array([r['N'] for r in rows])
    dt=t[:,None]-t[None,:];dn=n[:,None]-n[None,:];ok=dt>=2
    values=dn[ok]/dt[ok]
    return dict(slope=float(np.median(values)),T=float(1/abs(np.median(values))),span=float(t[-1]-t[0]),
        slope_quartiles=np.percentile(values,[25,75]).tolist(),support=len(rows))

def main():
    out=[]
    for item in plan():
        rows=read(R22/'delivery'/item['id']/'frames.json')
        events=read(R22/'events'/f"{item['id']}.json")['events'];diagnostics=[]
        for event in events:
            t0=event['entry_confirmed_s'];fits=[]
            for span in [6,8,10,12,16,20]:
                selected=[r for r in rows if t0-span<=r['time_s']<t0-.5 and r['available'] and r['N']>2
                    and r['current_coil_side_axis'] and not r['boundary_occluded']]
                result=slope(selected)
                fits.append(dict(window_s=span,fit=result,t2=t0+2*result['T'] if result else None))
            diagnostics.append(dict(near=event,fits=fits))
        out.append(dict(id=item['id'],events=diagnostics))
        print(item['id'],[(d['near']['entry_confirmed_s'],[(f['window_s'],round(f['fit']['T'],3) if f['fit'] else None) for f in d['fits']]) for d in diagnostics])
    target=read(R22/'delivery/dual_20260924_111500__left/frames.json')
    selected=[{k:r.get(k) for k in ['frame','time_s','N','N_interval','pitch','axis_slope','axis_uncertainty_px',
        'axis_width_px','current_coil_side_axis','boundary_occluded','motion_distance']} for r in target if 22<=r['time_s']<=32]
    dump(ROOT/'inspection/window22_32.json',selected);dump(ROOT/'inspection/period_diagnostics.json',out)
    print('POSE',[dict(t=r['time_s'],N=round(r['N'],3) if r['N'] is not None else None,slope=r['axis_slope']) for r in selected if r['frame']%2==0])

if __name__=='__main__':main()
