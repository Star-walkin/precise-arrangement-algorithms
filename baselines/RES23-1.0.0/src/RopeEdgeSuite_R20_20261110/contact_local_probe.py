"""Diagnostic: current tangency evidence, independent of fixed-d calibration."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from common20 import *
from count20 import measure
from temporal_contact20 import contact_evidence

def main():
    id='dual_20260924_111143__right';out=[]
    for i in [10,50,80,100,120,140,150,160,180,210,250,280,310]:
        with gzip.open(ROOT/'geometry_v5'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:geo=json.load(f)
        tc=geo.get('tangent') or {};row=dict(frame=i,tangent_valid=tc.get('valid'),tangent_reasons=tc.get('reasons'),d=tc.get('d'),N=None,contact=None)
        if tc.get('valid') and geo['axis']['current_coil_side_measured']:
            fr=read(ROOT/'f3_v2'/id/'records'/f'{i:06d}.json')
            with np.load(fr['fields_path']) as z:data={k:z[k] for k in z.files}
            with gzip.open(ROOT/'groups_v2'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:g=json.load(f)
            m=measure(g['groups'],data,record(id,i)['boundary'],geo['axis'],dict(status='current_tangency_diagnostic_no_locked_d',d=tc['d'],interval=tc['interval']))
            row.update(N=m.get('N'),N_interval=m.get('N_interval'),contact=contact_evidence(m,geo['axis']),measurement=m)
        out.append(row);print({k:v for k,v in row.items() if k!='measurement'},flush=True)
    dump(ROOT/'contact_local_probe.json',out)

if __name__=='__main__':main()
