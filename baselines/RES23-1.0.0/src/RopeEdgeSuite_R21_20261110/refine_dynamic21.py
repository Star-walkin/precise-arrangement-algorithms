import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from common21 import *
from measurement21b import evaluate,DynamicGate
import concurrent.futures as cf,time,collections

def process(item):
    id=item['id'];cal=read(R20/'calibration'/f'{id}.json');phases=[p for p in cal['epochs'] if p['d'] is None]
    if not phases:return dict(id=id,frames=0)
    gate=DynamicGate();rows=[]
    for phase in phases:
        for i in range(phase['start'],phase['end']):
            old=count20(id,i);r=frontend(id,i);fr=read(R20/'f3_v2'/id/'records'/f'{i:06d}.json');geo=geometry(id,i)
            groups=gzread(R20/'groups_v2'/id/'frames'/f'{i:06d}.json.gz')['groups'];data=None
            if fr['called']:
                assert sha(fr['fields_path'])==fr['fields_sha256']
                with np.load(fr['fields_path']) as z:data={k:z[k] for k in z.files}
            m,diag=evaluate(groups,data,r['boundary'],geo,phase,gate,i/item['fps'],phase['epoch'])
            row=dict(old,measurement=m,axis=geo.get('axis'),dynamic_diagnostic=diag,R21_variant='uncertainty_aware_dynamic_gate_B',R21_baseline_record_sha256=sha(R20/'counts_v1'/id/'frames'/f'{i:06d}.json.gz'))
            path=ROOT/'dynamic_b'/id/'frames'/f'{i:06d}.json.gz';gzwrite(path,row)
            rows.append(dict(frame=i,available=m['available'],N=m.get('N'),interval=m.get('N_interval'),reasons=m['reasons']))
    result=dict(id=id,frames=len(rows),available=sum(r['available'] for r in rows),reason_counts=dict(collections.Counter(s for r in rows if not r['available'] for s in r['reasons'])),rows=rows)
    dump(ROOT/'dynamic_b'/id/'summary.json',result);print(id,result['frames'],result['available'],result['reason_counts'],flush=True);return {k:v for k,v in result.items() if k!='rows'}

if __name__=='__main__':
    with cf.ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(process,plan()))
    dump(ROOT/'dynamic_b_receipt.json',dict(videos=results,code_hashes={f:sha(ROOT/f) for f in ['measurement21b.py','refine_dynamic21.py']}))
