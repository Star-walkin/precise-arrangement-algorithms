"""Freeze a completed fixed-development revision before extra8 inspection."""
from datetime import datetime,timezone
import argparse
from common import HERE,load,dump,sha

ap=argparse.ArgumentParser();ap.add_argument('--pilot',required=True);ap.add_argument('--free-checks',required=True);a=ap.parse_args()
assert not (HERE/'freeze_final.json').exists()
assert not (HERE/'extra_001').exists() and not (HERE/'all_001').exists()
s=load(HERE/a.pilot/'summary.json');start=load(HERE/a.pilot/'run_start.json');cfg=load(HERE/'inputs.json')
assert s['exit_code']==0 and s['completed']==14 and s['source_unchanged']
assert set(start['input_ids'])==set(cfg['pilot_ids'])
assert all(sha(p)==h for p,h in (start['sources']|start['reused_sources']).items())
audit=load(HERE/a.pilot/'independent_geometry_audit.json');assert audit['status']=='passed'
checks=load(HERE/a.free_checks/'check_receipt.json')
assert checks['numeric']['passed'] and checks['source_sha256']==sha(HERE/'heat_free.py')
for name,source in [('score_contract_check.json','connect_paths.py'),('direction_contract_check.json','run_h5.py')]:
    d=load(HERE/name);assert d['status']=='passed' and d['source_sha256']==sha(HERE/source)
review=HERE/a.pilot/'visual_review_receipt.json';assert review.exists()
compare=HERE/f'comparison_{a.pilot}'/'summary.json';assert compare.exists()
f=dict(start,freeze_utc=datetime.now(timezone.utc).isoformat(),selected_pilot=a.pilot,
    free_check_receipt=dict(path=str(HERE/a.free_checks/'check_receipt.json'),sha256=sha(HERE/a.free_checks/'check_receipt.json')),
    fixed14_review=dict(path=str(review),sha256=sha(review)),fixed14_comparison=dict(path=str(compare),sha256=sha(compare)),
    extra8_not_yet_run=True,extra8_for_frozen_open_development_check_not_blind_test=True,
    stop_reason='Four bounded fixed-sample development revisions; keep remaining visible misses, unknown tails and endpoint ambiguity. No near-perfect identity claim.',
    no_postfreeze_tuning=True,labels_read=False,reserved_data_read=False,devices_connected=False)
dump(HERE/'freeze_final.json',f)
print('Final H5 core frozen before extra8 and full292')
