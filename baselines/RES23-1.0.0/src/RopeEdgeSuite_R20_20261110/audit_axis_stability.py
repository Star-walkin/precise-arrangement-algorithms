"""Assess the user's fixed-slope assumption from current image measurements."""
from common20 import *

def main():
    output=[]
    for item in plan():
        id=item['id'];phases=next(x for x in read(ROOT/'motion_phases_v2.json')['videos'] if x['id']==id)['epochs'];res=[]
        for phase in phases:
            observations=[]
            for f in sorted((ROOT/'geometry_v5'/id/'frames').glob('*.json.gz')):
                i=int(f.name.split('.')[0])
                if not phase['start']<=i<phase['end']:continue
                with gzip.open(f,'rt',encoding='utf8') as h:r=json.load(h)
                a=r['axis']
                if not a or not a['current_coil_side_measured']:continue
                aa,bb,cc=a['line_abc'];observations.append([i,float(np.degrees(np.arctan(-aa/bb))),a['residual_q90_px'],a['uncertainty_px']])
            q=np.asarray(observations).reshape(-1,4);entry=dict(epoch=phase['epoch'],start=phase['start'],end=phase['end'],observed_frames=len(q))
            if len(q):entry.update(angle_median_deg=float(np.median(q[:,1])),angle_p05_p95_deg=np.percentile(q[:,1],[5,95]).tolist(),angle_MAD_deg=float(1.4826*np.median(abs(q[:,1]-np.median(q[:,1])))),median_current_fit_residual_px=float(np.median(q[:,2])))
            res.append(entry)
        output.append(dict(id=id,epochs=res))
    dump(ROOT/'axis_stability.json',dict(videos=output,interpretation='variation combines real guide/rope geometry and measurement error; not ground-truth angle',partial=not (ROOT/'geometry_v5_receipt.json').exists()))
    for v in output:print(v,flush=True)

if __name__=='__main__':main()
