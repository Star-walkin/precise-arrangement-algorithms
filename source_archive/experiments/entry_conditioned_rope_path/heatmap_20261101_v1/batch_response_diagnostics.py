"""Numerical full-domain diagnostics, explicitly not target-identity scoring."""
import sys,time
from pathlib import Path
import numpy as np
from common import HERE,load,dump,sha

def stats(x):
    return dict(mean=float(x.mean()),std=float(x.std()),minimum=float(x.min()),maximum=float(x.max()),
                fraction_gt_half=float((x>.5).mean()),fraction_gt_08=float((x>.8).mean()))

def main():
    tick=time.time();protocol=load(HERE/'protocol.json');manifest=load(HERE/'batch_inference/manifest.json')
    lookup={r['id']:r for r in protocol['jobs']};rows=[]
    for item in manifest['rows']:
        job=lookup[item['id']]
        for entry in item['entries']:
            new_path=Path(entry['response_file']);old_path=Path(job['response_files'][entry['entry_index']])
            assert sha(new_path)==entry['response_sha256']
            with np.load(new_path,allow_pickle=False) as z:new=z['response']
            with np.load(old_path,allow_pickle=False) as z:old=z['response']
            assert old.shape==new.shape==(975,2048) and np.isfinite(new).all() and 0<=new.min()<=new.max()<=1
            rows.append(dict(id=item['id'],entry_index=entry['entry_index'],old_B=stats(old),new=stats(new),
                mean_abs_change=float(np.mean(np.abs(new-old))),
                old_response_sha256=sha(old_path),new_response_sha256=entry['response_sha256']))
    assert len(rows)==301
    summary={arm:{k:float(np.mean([r[arm][k] for r in rows])) for k in rows[0][arm]} for arm in ['old_B','new']}
    result=dict(command=sys.argv,exit_code=0,seconds=time.time()-tick,script_sha256=sha(__file__),
        manifest_sha256=sha(HERE/'batch_inference/manifest.json'),images=292,conditions=301,
        diagnostic_domain='Entire computed975x2048 search rectangle; includes target, other ropes, objects and unknown background.',
        mean_abs_change=float(np.mean([r['mean_abs_change'] for r in rows])),condition_macro=summary,
        near_constant_std_lt_001={a:sum(r[a]['std']<.01 for r in rows) for a in ['old_B','new']},rows=rows,
        limits='No labels or paths used. Finite/nonconstant output and more high responses do not establish activity-rope clarity or identity. Conditions are not independent samples.',
        used_for_selection=False)
    dump(HERE/'batch_inference/response_diagnostics.json',result)
    print({k:result[k] for k in ['images','conditions','mean_abs_change','condition_macro','near_constant_std_lt_001']})

if __name__=='__main__':main()
