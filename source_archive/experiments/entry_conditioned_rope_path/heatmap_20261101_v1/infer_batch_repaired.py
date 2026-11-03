"""Repair only a missing logging import; frozen model/input code stays intact.

The first one-response failed attempt is preserved in batch_inference_failed_001.
No prediction, parameter, choice, or frozen inference source is modified.
"""
import argparse,json,sys,time
import numpy as np
import infer_heatmaps as frozen
from common import HERE,load,dump,sha
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--threads',type=int,default=8);a=ap.parse_args()
    sys.stdout.reconfigure(encoding='utf8');frozen.torch.set_num_threads(a.threads);frozen.torch.use_deterministic_algorithms(True)
    assert 'json' not in frozen.__dict__
    frozen.json=json  # the sole repair: progress logging uses this stdlib name.
    tick=time.time();frozen.batch()
    old=HERE/'batch_inference_failed_001/responses/dual_20260923_101517__pair0015__right/entry_00/response.npz'
    new=HERE/'batch_inference/responses/dual_20260923_101517__pair0015__right/entry_00/response.npz'
    with np.load(old,allow_pickle=False) as a,np.load(new,allow_pickle=False) as b:equal=np.array_equal(a['response'],b['response'])
    assert equal
    dump(HERE/'batch_inference/repair_receipt.json',dict(command=sys.argv,exit_code=0,repair_source_sha256=sha(__file__),
        frozen_inference_sha256=sha(HERE/'infer_heatmaps.py'),
        reason='NameError in progress logging: missing stdlib json name. Numerical code and chosen model unchanged.',
        failed_attempt=str(HERE/'batch_inference_failed_001'),failed_forward_calls=1,successful_forward_calls=301,
        saved_first_failed_response_equals_rerun=equal,seconds=time.time()-tick))
if __name__=='__main__':main()
