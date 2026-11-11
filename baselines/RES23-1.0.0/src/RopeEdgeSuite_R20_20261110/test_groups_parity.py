import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import time
from common20 import *
from groups20 import group
id='dual_20260923_113745__left';i=150;out=ROOT/'probes'/id
with np.load(out/f'f3_{i:06d}.npz') as f:data={k:f[k] for k in f.files}
old=read(out/f'groups_{i:06d}.json');tick=time.time();new=plain(group(data,record(id,i)['boundary'],old['d']))
assert len(new['curves'])==len(old['curves'])
for a,b in zip(new['curves'],old['curves']):
    assert np.array_equal(np.asarray(a['xy']),np.asarray(b['xy'])*2)
    assert a['point_ids']==b['point_ids'] and a['point_states']==b['point_states']
assert new['crossings_at_initialization']==[dict(c,y=c['y']*2) for c in old['crossings']]
dump(ROOT/'groups_float_cache_parity.json',dict(passed=True,frame=i,id=id,curves=len(new['curves']),
    exact_point_memberships_curves_states_and_crossings=True,elapsed_s=time.time()-tick,
    adapter_sha256=sha(ROOT/'groups20.py'),reference='unchanged BG3 global_points; per-frame field conversion only'))
print('Exact parity passed',len(new['curves']),round(time.time()-tick,2),flush=True)
