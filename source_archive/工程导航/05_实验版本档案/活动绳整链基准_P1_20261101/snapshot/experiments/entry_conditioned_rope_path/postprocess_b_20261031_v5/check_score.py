"""Numerical score contracts; not rope identity validation."""
import numpy as np
from common import HERE,dump,sha
from connect_paths import straight_credit,utility,resample
p=np.stack([np.arange(0.,501.,3.),np.full(167,200.)],1)
protected=np.ones(len(p),bool);protected[70:85]=False
states=['entry_condition' if i<=20 else 'image_supported' if protected[i] else 'ambiguous_gap' for i in range(len(p))]
free=dict(points_xy=p.tolist(),protect_support=protected.tolist(),point_status=states,width_half_px=10.)
full=straight_credit(p,free)
expected=sum(6. for i in range(1,len(p)) if protected[i] and protected[i-1] and i-1>20)
assert abs(full['supported_straight_original_px']-expected)<1e-6
copy=straight_credit(resample(p,1.1),free)
assert abs(copy['supported_straight_original_px']-full['supported_straight_original_px'])<1e-6
turn=np.vstack([p[:100],np.stack([np.arange(300.,501.,3.),np.linspace(200.,250.,67)],1)])
short=straight_credit(turn,free)
assert short['supported_straight_original_px']<full['supported_straight_original_px']-100
a,_=utility(dict(supported_straight_original_px=480),0,0,0,10)
b,_=utility(dict(supported_straight_original_px=431),1,1,1,0)
assert a>b
c,detail=utility(full,.8,.7,.6,.1)
d,_=utility(full,.3,.7,.6,.1)
assert c>d and detail['loss']==-c
e,_=utility(full,.8,.9,.6,.1)
assert e>c
observed=protected.copy();observed[70:85]=True
af=dict(free,observed_straight_support=observed.tolist())
ac=straight_credit(p,af)
assert ac['supported_straight_original_px']>full['supported_straight_original_px']
assert free['protect_support']==protected.tolist()
result=dict(status='passed',checks=['unknown_and_seed_excluded','sampling_density_invariant','early_geometric_turn_loses_length','higher_length_tier_dominates_any_secondary','smaller_turn_scores_higher','supported_rail_scores_higher','explicit_A7_support_distinct_from_B'],full_credit=full,early_turn_credit=short,source_sha256=sha(HERE/'connect_paths.py'),identity_accuracy_measured=False)
dump(HERE/'score_contract_check.json',result)
print('7 numerical score contracts passed')
