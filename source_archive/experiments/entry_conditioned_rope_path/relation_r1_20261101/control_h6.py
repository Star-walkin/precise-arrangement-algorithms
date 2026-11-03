"""Same26 fixed open images, original H6 rules, A750 versus B750."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from common import *
import numpy as np,cv2,gzip,time,sys,traceback,concurrent.futures
H=None
def init():
    global H
    cv2.setNumThreads(1);H=h6()
def one(job):
    tick=time.time();priors=H.load_surface_priors(dict(job,seeds=job['original_seeds']))
    gray=cv2.cvtColor(priors['original_half'][525:],cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    results={}
    for variant in ['A','B']:
        rows=[]
        for k,seed in enumerate(job['seeds']):
            path=HERE/'cache/gallery'/job['id']/f'entry_{k:02d}.npz' if variant=='A' else Path(job['response_files'][k])
            with np.load(path) as z:heat=z['response']
            tracks=H.build_tracks(priors,heat,gray);full=H.describe_free(priors,seed,heat)
            free,projection=H.prepare_task_free(full,tracks['tracks'])
            linked=H.connect(gray,heat,free,tracks['tracks'],[],priors['surface_half'][525:])
            rows.append(dict(entry_index=k,seed=seed,response_path=str(path),response_sha256=sha(path),full_free_observation=full,free=free,
                free_projection=projection,connection=linked,track_diagnostics=tracks['diagnostics']))
        H.task_direction_feasibility(rows,priors['sam2_surface_half'][525:])
        results[variant]=rows
    record=dict(id=job['id'],variants=results,geometric_rules_changed=False,thresholds_changed=False,labels_read=False,seconds=time.time()-tick)
    dest=HERE/'h6_control/records'/(job['id']+'.json.gz');dest.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(dest,'wt',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,allow_nan=False)
    return dict(id=job['id'],seconds=time.time()-tick,variants={v:[dict(entry=r['entry_index'],candidate_count=len(r['connection']['candidates']),
        main_kind=r['connection']['candidates'][0]['kind'] if r['connection']['candidates'] else 'none',
        observed_straight_length=r['full_free_observation'].get('observed_straight_length_original_px')) for r in rows] for v,rows in results.items()},path=str(dest))
def main():
    sys.stdout.reconfigure(encoding='utf8');p=load(HERE/'protocol.json');jobs=[j for j in p['jobs'] if j['id'] in p['controlled_h6_ids']]
    out=HERE/'h6_control';out.mkdir(exist_ok=False);rows=[];errors=[];tick=time.time()
    with concurrent.futures.ProcessPoolExecutor(4,initializer=init) as ex:
        pending={ex.submit(one,j):j['id'] for j in jobs}
        for f in concurrent.futures.as_completed(pending):
            try:rows.append(f.result())
            except Exception:errors.append(dict(id=pending[f],error=traceback.format_exc()))
            print(dict(done=len(rows),errors=len(errors),total=len(jobs),seconds=round(time.time()-tick,1)),flush=True)
    dump(out/'summary.json',dict(rows=rows,errors=errors,exit_code=int(bool(errors)),seconds=time.time()-tick,planned=len(jobs),same_rules_and_budget=True))
    sys.exit(bool(errors))
if __name__=='__main__':main()
