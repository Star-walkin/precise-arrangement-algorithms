import os,time
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from common21 import *
from count20 import measure
for id,i in [('dual_20260924_111143__right',60),('dual_20260924_111500__left',247),('dual_20260923_113745__left',397)]:
    r=frontend(id,i);old=count20(id,i);g=geometry(id,i);fr=read(R20/'f3_v2'/id/'records'/f'{i:06d}.json')
    with np.load(fr['fields_path']) as z:data={k:z[k] for k in z.files}
    groups=gzread(R20/'groups_v2'/id/'frames'/f'{i:06d}.json.gz')['groups']
    cal=next(c for c in read(R20/'calibration'/f'{id}.json')['epochs'] if c['start']<=i<c['end'])
    if cal['d'] is None:cal=dict(d=g['tangent']['d'],interval=g['tangent']['interval'],status='current_dynamic_geometry')
    t=time.time();m=measure(groups,data,r['boundary'],g['axis'],cal)
    print(id,i,'seconds',time.time()-t,'oldN',old['measurement'].get('N'),'newN',m.get('N'),'keys',list(old), 'central', {k:v for k,v in (m.get('central') or {}).items() if k in ['boundary','local_pitch_px','current_pitch_intervals','phase_complete']},flush=True)
