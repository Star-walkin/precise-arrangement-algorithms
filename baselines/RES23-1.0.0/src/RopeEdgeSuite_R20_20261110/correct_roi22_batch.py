import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import concurrent.futures as cf,shutil
from common20 import *
from probe_roi22 import process
from extract_parallel22 import receipt

def write_gz(path,obj):
    tmp=path.with_suffix('.correcting')
    with gzip.open(tmp,'wt',encoding='utf8',compresslevel=2) as f:json.dump(plain(obj),f,ensure_ascii=False,allow_nan=False)
    tmp.replace(path)

def analyze(task):
    id,i=task;path=ROOT/'roi22_corrections'/id/f'{i:06d}.json'
    if path.exists():return read(path)
    return process(id,i)

def main():
    assert read(ROOT/'count_receipt.json')['status']=='complete'
    tasks=[]
    for item in plan():
        id=item['id']
        for i in range(item['planned_frames']):
            with gzip.open(ROOT/'counts_v1'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as h:row=json.load(h)
            m=row['measurement'];d=m.get('d')
            if m['available'] or d is None or m['status']=='projection_geometry_inconsistent':continue
            fr=read(ROOT/'f3_v2'/id/'records'/f'{i:06d}.json')
            if not fr['called'] or fr['roi'].get('extension_reason')!='visible_outline_already_covers_three_quarters_of_current_surface':continue
            if 'active_projection_outside_image' in (m.get('central') or {}).get('reasons',[]):continue
            curve=np.asarray(record(id,i)['boundary']['top_curve_xy']);lo,hi=m['d_interval']
            if lo-3*PITCH_PRIOR<curve[0,0] or hi+3*PITCH_PRIOR>curve[-1,0]:tasks.append((id,i))
    dump(ROOT/'roi22_candidate_frames.json',tasks);print('Candidates',len(tasks),tasks,flush=True)
    with cf.ProcessPoolExecutor(max_workers=6) as pool:results=list(pool.map(analyze,tasks))
    promoted=[];rejected=[]
    for x in results:
        id=x['id'];i=x['frame'];cp=ROOT/'counts_v1'/id/'frames'/f'{i:06d}.json.gz';assert sha(cp)==x['old_count_file_sha256']
        with gzip.open(cp,'rt',encoding='utf8') as h:row=json.load(h)
        oldaxis=row['axis'];newaxis=x['geometry']['axis'];d=x['measurement'].get('d');shift=None
        if oldaxis and newaxis and d is not None:
            ay=lambda a:-(a['line_abc'][0]*d+a['line_abc'][2])/a['line_abc'][1]
            shift=abs(ay(oldaxis)-ay(newaxis))
        if not x['measurement']['available'] or shift is None or shift>.35*PITCH_PRIOR:
            rejected.append(dict(id=id,frame=i,available=x['measurement']['available'],axis_shift_px=shift,reasons=x['measurement']['reasons']));continue
        archive=ROOT/'roi22_corrections_archive'/id/f'{i:06d}';archive.mkdir(parents=True,exist_ok=False)
        fp=ROOT/'f3_v2'/id/'records'/f'{i:06d}.json';gp=ROOT/'groups_v2'/id/'frames'/f'{i:06d}.json.gz';ap=ROOT/'geometry_v5'/id/'frames'/f'{i:06d}.json.gz'
        for name,p in [('F3.json',fp),('groups.json.gz',gp),('geometry.json.gz',ap),('count.json.gz',cp)]:shutil.copy2(p,archive/name)
        correction=dict(producer_sha256=sha(__file__),probe_sha256=sha(ROOT/'probe_roi22.py'),roi_code_sha256=sha(ROOT/'roi22.py'),
            evidence_file_sha256=sha(ROOT/'roi22_corrections'/id/f'{i:06d}.json'),previous_count_sha256=sha(cp),axis_shift_px=shift,previous_records=str(archive))
        fr=read(fp);fr.update(roi=x['roi'],mask_sha256=x['mask_sha256'],points=x['points'],fields_path=x['field'],fields_sha256=x['field_sha256'],
            reused_exact_mask_result=False,reuse_provenance=None,ROI22_correction=correction);dump(fp,fr)
        with gzip.open(gp,'rt',encoding='utf8') as h:g=json.load(h)
        g.update(groups=x['groups'],tangent=x['geometry'].get('tangent'),raw_fields_sha256=x['field_sha256'],extraction_record_sha256=sha(fp),exact_input_group_reuse=None,ROI22_correction=correction);write_gz(gp,g)
        a=dict(x['geometry'],frame=i,time_s=row['time_s'],source_pixel_sha256=row['source_pixel_sha256'],analysis_pixel_sha256=row['analysis_pixel_sha256'],
            groups_file_sha256=sha(gp),geometry_code_sha256=sha(ROOT/'geometry22.py'),axis_code_sha256=sha(ROOT/'coil_edge_axis_v4.py'),ROI22_correction=correction);write_gz(ap,a)
        row.update(measurement=x['measurement'],axis=x['geometry']['axis'],F3_record_sha256=sha(fp),F3_fields_sha256=x['field_sha256'],
            groups_file_sha256=sha(gp),geometry_file_sha256=sha(ap),ROI22_correction=correction);write_gz(cp,row)
        promoted.append(dict(id=id,frame=i,axis_shift_px=shift,N=x['measurement']['N'],N_interval=x['measurement']['N_interval'],count_sha256=sha(cp)))
    for id in sorted(set(x['id'] for x in promoted)):receipt(next(x for x in plan() if x['id']==id))
    result=dict(candidates=tasks,promoted=promoted,rejected=rejected,code_sha256=sha(__file__),roi_code_sha256=sha(ROOT/'roi22.py'),
        preserved_five_rope_calibration_references=True,all_promoted_frames_previously_unavailable=True)
    dump(ROOT/'roi22_correction_receipt.json',result)
    for name in ('f3_v2_receipt.json','group_v2_receipt.json','geometry_v5_receipt.json','count_receipt.json'):
        p=ROOT/name;v=read(p);v['current_ROI22_correction_receipt_sha256']=sha(ROOT/'roi22_correction_receipt.json');dump(p,v)
    print('Promoted',len(promoted),'Rejected',len(rejected),flush=True)

if __name__=='__main__':main()
