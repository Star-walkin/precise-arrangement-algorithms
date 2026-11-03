import sys
from common import HERE,dump,sha
from topology import check_attachment
target=dict(track_id='target',boundary_ids=['u','l'],points_xy=[[0.,0.],[5.,3.],[10.,0.],[20.,0.]])
r=check_attachment([[0.,0.],[10.,0.]],[[10.,0.],[20.,0.]],[target],'target')
dump(HERE/'diagnostics_source'/sys.argv[1],dict(result=r,source_sha256=sha(HERE/'topology.py'),expected_allowed=False))
assert not r['allowed'],'Bridge vertices hide a departure between target knots'
print('Target-knot suffix regression passed')
