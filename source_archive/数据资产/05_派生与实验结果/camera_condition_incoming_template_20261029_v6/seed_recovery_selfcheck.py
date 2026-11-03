"""Synthetic mechanisms only; no real-image identity accuracy conclusion."""
from pathlib import Path
import json,hashlib,time
import numpy as np,cv2
from free_seed_recovery import recover_free_seeds

def synthetic(kind):
    h,w=600,800;image=np.zeros((h,w,3),np.uint8);image[:]=(24,90,105);surface=np.zeros((h,w),np.uint8);surface[230:580,520:780]=255
    yy,xx=np.mgrid[:h,:w];d=np.array([1.,.3]);d/=np.linalg.norm(d);n=np.array([-d[1],d[0]]);origin=np.array([40.,140.]);p=np.stack([xx-origin[0],yy-origin[1]],axis=-1);along=p@d;normal=p@n
    if kind!='blank':
        for offset in ([0.,42.] if kind=='two' else [0.]):
            take=(abs(normal-offset)<=9)&(along>=20)&(along<=610)
            value=np.full((h,w),180.) if kind=='plain' else 165+60*np.sin(2*np.pi*(along/18+(normal-offset)/13))
            image[take]=np.clip(value[take,None],0,255).astype(np.uint8)
    return {'original':image,'surface':surface,'pitch':20.},origin,d,n

def main():
    cv2.setNumThreads(1);start=time.time();results={};checks={}
    for kind in ('blank','plain','rope','two'):
        data,origin,d,n=synthetic(kind);before=[hashlib.sha256(data[k].tobytes()).hexdigest() for k in ('original','surface')];r=recover_free_seeds(data)
        checks[kind+'_inputs_unchanged']=before==[hashlib.sha256(data[k].tobytes()).hexdigest() for k in ('original','surface')]
        results[kind]={'count':len(r['candidates']),'status':r['status'],'rejects':[v['evidence']['reason'] for v in r['diagnostics']['rejected']]}
        if kind in ('blank','plain'):checks[kind+'_not_promoted']=not r['candidates']
        else:
            checks[kind+'_retains_candidates']=len(r['candidates'])>=(2 if kind=='two' else 1)
            offsets=[]
            for c in r['candidates']:
                s=c['seed'];q=np.asarray(s['observed_segment_xy']);offset=float((q.mean(0)-origin)@n);offsets.append(offset)
                checks[kind+'_'+c['candidate_id']+'_finite_original_seed']=bool(np.isfinite(q).all() and 0<=q[:,0].min()<800 and 0<=q[:,1].min()<600 and q[:,0].max()<800 and q[:,1].max()<600 and s['coordinate_system']=='original_unmirrored_image_xy_pixels')
                checks[kind+'_'+c['candidate_id']+'_known_axis_recovered']=min(abs(offset-v) for v in ([0.,42.] if kind=='two' else [0.]))<3. and abs(np.asarray(s['direction_xy'])@d)>.999
            results[kind]['offsets']=offsets
            if kind=='two':checks['two_distinct_competitors_preserved']=any(abs(v)<3 for v in offsets) and any(abs(v-42)<3 for v in offsets)
    checks={k:bool(v) for k,v in checks.items()}
    report={'scope':'Synthetic direct-array seed recovery; original plain stripe and textured diagonal-strand strip have identical axis/width/background. No real-image success rate.','checks':checks,'passed':all(checks.values()),'results':results,'seconds':time.time()-start,'source_sha256':hashlib.sha256((Path(__file__).parent/'free_seed_recovery.py').read_bytes()).hexdigest()};path=Path(__file__).parent/'seed_recovery_selfcheck_002.json'
    if path.exists():raise ValueError('Preserve old receipt')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,ensure_ascii=False));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
