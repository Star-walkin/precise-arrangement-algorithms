import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import concurrent.futures as cf,shutil
from common20 import *
from probe_fragments23 import process
from correct_roi22_batch import write_gz
from extract_parallel22 import receipt

def analyze(task):
    id,i=task;p=ROOT/'fragments23_trials'/id/f'{i:06d}.json'
    if p.exists():return read(p)
    return process(task)

def main():
    tasks=[]
    for item in plan():
        id=item['id'];cal=read(ROOT/'calibration'/f'{id}.json')
        for i in range(item['planned_frames']):
            with gzip.open(ROOT/'counts_v1'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:row=json.load(f)
            m=row['measurement'];d=m.get('d')
            if m['available'] or d is None or m['status']=='projection_geometry_inconsistent':continue
            phase=next(x for x in cal['epochs'] if x['start']<=i<x['end'])
            assert i not in phase['far_guard_accepted_frames']
            r=record(id,i);polys=r['proposal'].get('surface_contours_xy',[])
            if not polys or not r['track'].get('valid'):continue
            xx=np.concatenate([np.asarray(p)[:,0] for p in polys]);lo,hi=phase['interval']
            if xx.min()>lo-3*PITCH_PRIOR or xx.max()<hi+3*PITCH_PRIOR:tasks.append((id,i))
    dump(ROOT/'fragments23_candidate_frames.json',tasks);print('Candidates',len(tasks),flush=True)
    with cf.ProcessPoolExecutor(max_workers=6) as pool:results=list(pool.map(analyze,tasks))
    promoted=[];rejected=[]
    for x in results:
        id=x['id'];i=x['frame'];cp=ROOT/'counts_v1'/id/'frames'/f'{i:06d}.json.gz';assert sha(cp)==x['old_count_file_sha256']
        with gzip.open(cp,'rt',encoding='utf8') as h:row=json.load(h)
        if not x['measurement']['available']:
            rejected.append(dict(id=id,frame=i,reasons=x['measurement']['reasons']));continue
        oldaxis=row['axis'];newaxis=x['geometry']['axis'];d=x['measurement']['d'];shift=None
        if oldaxis and newaxis:
            ay=lambda a:-(a['line_abc'][0]*d+a['line_abc'][2])/a['line_abc'][1]
            shift=abs(ay(oldaxis)-ay(newaxis))
        # Preserve rope identity while repairing a region failure.
        if shift is None or shift>.35*PITCH_PRIOR:
            rejected.append(dict(id=id,frame=i,reason='axis_identity_change_requires_review',axis_shift_px=shift));continue
        archive=ROOT/'fragments23_archive'/id/f'{i:06d}';archive.mkdir(parents=True,exist_ok=False)
        fp=ROOT/'f3_v2'/id/'records'/f'{i:06d}.json';gp=ROOT/'groups_v2'/id/'frames'/f'{i:06d}.json.gz';ap=ROOT/'geometry_v5'/id/'frames'/f'{i:06d}.json.gz'
        for name,p in [('F3.json',fp),('groups.json.gz',gp),('geometry.json.gz',ap),('count.json.gz',cp)]:shutil.copy2(p,archive/name)
        frontend=ROOT/'frontend_corrections'/id/f'{i:06d}.json.gz';frontend.parent.mkdir(parents=True,exist_ok=True);write_gz(frontend,x['frontend'])
        correction=dict(producer_sha256=sha(__file__),probe_sha256=sha(ROOT/'probe_fragments23.py'),boundary_code_sha256=sha(ROOT/'current_boundary23.py'),segment_code_sha256=sha(ROOT/'region_fragments23.py'),
            evidence_file_sha256=sha(ROOT/'fragments23_trials'/id/f'{i:06d}.json'),previous_count_sha256=sha(cp),axis_shift_px=shift,previous_records=str(archive),
            frontend_override_path=str(frontend),frontend_override_sha256=sha(frontend))
        fr=read(fp);fr.update(roi=x['roi'],mask_sha256=x['mask_sha256'],called=True,skip_reason=None,points=x['points'],fields_path=x['field'],fields_sha256=x['field_sha256'],
            reused_exact_mask_result=False,reuse_provenance=None,fragments23_correction=correction);dump(fp,fr)
        with gzip.open(gp,'rt',encoding='utf8') as h:g=json.load(h)
        g.update(groups=x['groups'],tangent=x['geometry'].get('tangent'),raw_fields_sha256=x['field_sha256'],extraction_record_sha256=sha(fp),exact_input_group_reuse=None,fragments23_correction=correction);write_gz(gp,g)
        a=dict(x['geometry'],frame=i,time_s=row['time_s'],source_pixel_sha256=row['source_pixel_sha256'],analysis_pixel_sha256=row['analysis_pixel_sha256'],
            groups_file_sha256=sha(gp),geometry_code_sha256=sha(ROOT/'geometry22.py'),axis_code_sha256=sha(ROOT/'coil_edge_axis_v4.py'),fragments23_correction=correction);write_gz(ap,a)
        row.update(measurement=x['measurement'],axis=x['geometry']['axis'],current_F3_called=True,F3_record_sha256=sha(fp),F3_fields_sha256=x['field_sha256'],
            groups_file_sha256=sha(gp),geometry_file_sha256=sha(ap),fragments23_correction=correction);write_gz(cp,row)
        promoted.append(dict(id=id,frame=i,N=x['measurement']['N'],axis_shift_px=shift,count_sha256=sha(cp),current_boundary_reinitialized=x['frontend']['boundary'].get('current_only_shape_recovery',{}).get('accepted',False)))
    for id in sorted(set(x['id'] for x in promoted)):receipt(next(x for x in plan() if x['id']==id))
    dump(ROOT/'fragments23_correction_receipt.json',dict(candidates=tasks,promoted=promoted,rejected=rejected,preserved_five_rope_calibration_references=True,code_sha256=sha(__file__)))
    for name in ('f3_v2_receipt.json','group_v2_receipt.json','geometry_v5_receipt.json','count_receipt.json'):
        p=ROOT/name;v=read(p);v['current_fragments23_correction_receipt_sha256']=sha(ROOT/'fragments23_correction_receipt.json');dump(p,v)
    print('PROMOTED',len(promoted),'REJECTED',len(rejected),flush=True)

if __name__=='__main__':main()
