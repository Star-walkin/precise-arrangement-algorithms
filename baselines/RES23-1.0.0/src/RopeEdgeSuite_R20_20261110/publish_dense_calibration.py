import shutil
from common20 import *
p=ROOT/'d_calibration.json';old=read(p);archive=ROOT/'dense_revision_previous/d_calibration_old.json'
assert not archive.exists();shutil.copy2(p,archive)
old['videos']=[read(ROOT/'calibration'/f"{x['id']}.json") for x in plan()]
old['refinement_receipt']=dict(previous_aggregate_sha256=sha(archive),changed_video='dual_20260923_113745__left',
    changed_epoch=1,code_sha256=sha(ROOT/'refine_sparse_d.py'),calibrator_sha256=sha(ROOT/'calibration21_dense.py'),
    action='densify existing far-reference window; retain all five-rope and time holdout gates')
old['producer_hashes'].update({str(ROOT/f):sha(ROOT/f) for f in ('refine_sparse_d.py','calibration21_dense.py','publish_dense_calibration.py')})
dump(p,old);print('Aggregate calibration updated',flush=True)
