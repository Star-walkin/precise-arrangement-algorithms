import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from common21 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    id='dual_20260924_111500__left'; rows=[]
    for i in range(0,601,2):
        r=count20(id,i);m=r['measurement'];c=m.get('central') or {};a=r.get('axis') or {};g=geometry(id,i);b=c.get('boundary') or {}
        abc=a.get('line_abc');y=-(abc[0]*2100+abc[2])/abc[1] if abc else None
        rows.append(dict(frame=i,t=i/10,N=m.get('N'),gap=c.get('contact_projected_gap_N'),boundary=b.get('y'),active=c.get('active_y'),
            boundary_unc=b.get('uncertainty_px'),local=b.get('locally_observed'),occluded=b.get('occluded'),axis_unc=a.get('uncertainty_px'),
            slope=-abc[0]/abc[1] if abc else None,axis_y2100=y,tangent=(g.get('tangent') or {}).get('d'),width=a.get('width_px'),
            interval=m.get('N_interval'),source=a.get('source'),coil_current=a.get('current_coil_side_measured')))
    dump(ROOT/'inspection'/'111500_signals.json',rows)
    fig,axes=plt.subplots(5,1,figsize=(15,14),sharex=True)
    for ax,ks in zip(axes,[['N','gap'],['active','boundary','axis_y2100'],['slope'],['tangent'],['boundary_unc','axis_unc','width']]):
        for k in ks:ax.plot([x['t'] for x in rows],[x[k] for x in rows],label=k)
        ax.legend();ax.grid();ax.set_xlim(15,40)
        for t in [24.7,24.9,29]:ax.axvline(t,color='gray',linestyle=':')
    fig.tight_layout();fig.savefig(ROOT/'inspection'/'111500_signals.png',dpi=120)
    print(json.dumps([r for r in rows if 220<=r['frame']<=330 and r['frame']%10==0],indent=2))

if __name__=='__main__':main()
