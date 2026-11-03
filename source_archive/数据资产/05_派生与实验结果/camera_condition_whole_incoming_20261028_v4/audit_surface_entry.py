"""Independent saved-path search-domain audit; no algorithm/hardware import.

This repeats the declared area/hull rule from numeric protocol constants,
then samples already-saved candidates. Entering this search region is NOT
evidence of rope identity, physical contact or sufficient visible coverage.
"""
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np
import whole_io as io

def independent_domain(source):
    binary=np.where(source>0,255,0).astype(np.uint8)
    count=int(np.count_nonzero(binary));cutoff=max(50.,.0001*count)
    cs,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cs=[c for c in cs if cv2.contourArea(c)>=cutoff]
    mask=np.zeros_like(binary)
    if not cs:return mask
    hull=np.zeros_like(binary);cv2.fillConvexPoly(hull,cv2.convexHull(np.concatenate(cs)),255)
    if np.count_nonzero(hull)/max(1,count)<=1.8:return hull
    cv2.drawContours(mask,cs,-1,255,cv2.FILLED)
    return mask

def inside(domain,points):
    p=np.asarray(points,np.float32).reshape(-1,2)
    if not len(p):return np.zeros(0,bool)
    h,w=domain.shape
    samples=cv2.remap((domain>0).astype(np.float32),p[:,0].reshape(1,-1),p[:,1].reshape(1,-1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel()>.3
    # Initial state uses integer truncation; later states use bilinear >.3.
    samples[0]=bool(domain[int(np.clip(p[0,1],0,h-1)),int(np.clip(p[0,0],0,w-1))])
    return samples

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    if Path(args.run).name!=args.run:raise ValueError('Direct child run name required')
    run=io.HERE/args.run;out=run/'candidate_surface_entry_audit.json'
    if out.exists():raise ValueError('Do not overwrite an audit')
    start=io.read_json(run/'run_start.json');allowed={r['output_file']:r for r in io.selection('all')}
    if any(k not in allowed for k in start['inputs']):raise ValueError('Input outside explicit open400 manifest')
    cv2.setNumThreads(1);began=time.time();cases=[]
    for name in start['inputs']:
        rel=Path(name);r=io.read_json(run/'记录'/rel.parent.name/(rel.stem+'.json'))
        surface_path=io.C4/'roi/surface_masks'/rel.parent.name/rel.name
        if io.sha(surface_path)!=r['record']['input_dependency_sha256'][str(surface_path)]:raise ValueError('Surface dependency changed')
        domain=independent_domain(io.read_image(surface_path,0))
        selected=r['whole'].get('selected_candidate_id');candidates=[]
        for c in r['whole'].get('candidates',[]):
            flags=inside(domain,c['xy'])
            candidates.append({'candidate_id':c['candidate_id'],'selected':c['candidate_id']==selected,'saved_point_count':len(flags),'domain_supported_samples':int(np.count_nonzero(flags)),'passed':bool(np.any(flags))})
        found=selected is None or any(c['selected'] for c in candidates)
        cases.append({'input':name,'status':r['whole']['status'],'selected_candidate_id':selected,'candidates':candidates,'passed':found and all(c['passed'] for c in candidates)})
    chosen=[c for r in cases for c in r['candidates'] if c['selected']]
    report={'run':args.run,'scope':'Independent significant-component search-domain construction and sampling of saved paths; not identity/accuracy evaluation.','case_count':len(cases),'selected_case_count':len(chosen),'candidate_count':sum(len(r['candidates']) for r in cases),'selected_paths_without_surface_entry':sum(not c['passed'] for c in chosen),'all_candidates_without_surface_entry':sum(not c['passed'] for r in cases for c in r['candidates']),'passed':all(c['passed'] for c in cases),'cases':cases,'seconds':time.time()-began,'script_sha256':io.sha(__file__)}
    io.dump(out,report)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False))
    return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
