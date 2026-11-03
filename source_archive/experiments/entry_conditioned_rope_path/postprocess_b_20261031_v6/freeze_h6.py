from datetime import datetime,timezone
import argparse
from common import HERE,load,dump,sha
ap=argparse.ArgumentParser();ap.add_argument('--pilot',required=True);a=ap.parse_args()
assert not (HERE/'freeze_final.json').exists() and not (HERE/'extra_001').exists()
start=load(HERE/a.pilot/'run_start.json');s=load(HERE/a.pilot/'summary.json')
assert s['completed']==16 and s['exit_code']==0 and s['source_unchanged']
assert all(sha(p)==v for p,v in (start['sources']|start['reused_sources']).items())
audit=load(HERE/a.pilot/'independent_geometry_audit.json');assert audit['status']=='passed'
review=HERE/a.pilot/'visual_review_receipt.json';assert review.exists()
checks=load(HERE/'topology_contract_check.json');assert checks['status']=='passed' and checks['source_sha256']==sha(HERE/'topology.py')
suffix=load(HERE/'diagnostics_source/suffix_final_check.json');assert not suffix['result']['allowed'] and suffix['source_sha256']==sha(HERE/'topology.py')
surface_check=HERE/'surface_checks_001/receipt.json';assert surface_check.exists()
dump(HERE/'freeze_final.json',dict(start,selected_pilot=a.pilot,freeze_utc=datetime.now(timezone.utc).isoformat(),
    fixed16_review=dict(path=str(review),sha256=sha(review)),source_surface_checks=dict(path=str(surface_check),sha256=sha(surface_check)),
    stop_reason='Fixed16 source/geometry review and numerical no-crossing contracts pass. Remaining missing-source/identity failures retained; no global correctness guarantee.',
    extra8_already_viewed_open_regression=True,no_postfreeze_tuning=True,labels_read=False,devices_connected=False))
print('H6 sources frozen for extra8 and full292')
