"""Permit measured identity recovery from an old extrapolated axis.

An extrapolated old axis is not a ground truth for rejecting a new current
coil-side measurement. Recovered current evidence must pass all CE4 identity,
width and overlap tests, tight residual, and complete current F3 phase gates.
"""
import shutil
from common20 import *
from correct_roi22_batch import write_gz
from extract_parallel22 import receipt

def main():
    receipt_path=ROOT/'fragments24_correction_receipt.json';rec=read(receipt_path);added=[]
    for q in rec['rejected']:
        if q.get('reason')!='axis_identity_change_requires_review':continue
        id,i=q['id'],q['frame'];x=read(ROOT/'fragments24_trials'/id/f'{i:06d}.json');cp=ROOT/'counts_v1'/id/'frames'/f'{i:06d}.json.gz'
        with gzip.open(cp,'rt',encoding='utf8') as f:row=json.load(f)
        old=row['axis'];a=x['geometry']['axis'];m=x['measurement'];front=x['frontend']
        if old is None or old['current_coil_side_measured'] or not a['current_coil_side_measured']:continue
        if not front['proposal']['valid'] or not m['available'] or not m['complete_F3_phase']:continue
        if a['uncertainty_px']>.2*PITCH_PRIOR or a['residual_q90_px']>.15*PITCH_PRIOR:continue
        assert not row['measurement']['available'] and sha(cp)==x['old_count_file_sha256']
        assert id=='dual_20260923_113745__left' and i==248  # reviewed fixture for this validation release
        archive=ROOT/'identity24_archive'/id/f'{i:06d}';archive.mkdir(parents=True,exist_ok=False)
        fp=ROOT/'f3_v2'/id/'records'/f'{i:06d}.json';gp=ROOT/'groups_v2'/id/'frames'/f'{i:06d}.json.gz';ap=ROOT/'geometry_v5'/id/'frames'/f'{i:06d}.json.gz'
        for name,p in [('F3.json',fp),('groups.json.gz',gp),('geometry.json.gz',ap),('count.json.gz',cp)]:shutil.copy2(p,archive/name)
        frontend=ROOT/'frontend_corrections'/id/f'{i:06d}.json.gz';frontend.parent.mkdir(parents=True,exist_ok=True);write_gz(frontend,front)
        corr=dict(producer_sha256=sha(__file__),probe_sha256=sha(ROOT/'probe_fragments24.py'),boundary_code_sha256=sha(ROOT/'current_boundary23.py'),segment_code_sha256=sha(ROOT/'region_fragments24.py'),
            evidence_file_sha256=sha(ROOT/'fragments24_trials'/id/f'{i:06d}.json'),previous_count_sha256=sha(cp),axis_shift_px=q['axis_shift_px'],previous_records=str(archive),
            frontend_override_path=str(frontend),frontend_override_sha256=sha(frontend),identity_policy='current_CE4_identity_recovery_supersedes_old_upstream_extrapolation',
            assistant_visual_review=str(ROOT/f'candidate_{id}_{i:06d}.jpg'))
        fr=read(fp);fr.update(roi=x['roi'],mask_sha256=x['mask_sha256'],called=True,skip_reason=None,points=x['points'],fields_path=x['field'],fields_sha256=x['field_sha256'],reused_exact_mask_result=False,reuse_provenance=None,fragments24_correction=corr);dump(fp,fr)
        with gzip.open(gp,'rt',encoding='utf8') as h:g=json.load(h)
        g.update(groups=x['groups'],tangent=x['geometry'].get('tangent'),raw_fields_sha256=x['field_sha256'],extraction_record_sha256=sha(fp),exact_input_group_reuse=None,fragments24_correction=corr);write_gz(gp,g)
        geo=dict(x['geometry'],frame=i,time_s=row['time_s'],source_pixel_sha256=row['source_pixel_sha256'],analysis_pixel_sha256=row['analysis_pixel_sha256'],groups_file_sha256=sha(gp),geometry_code_sha256=sha(ROOT/'geometry22.py'),axis_code_sha256=sha(ROOT/'coil_edge_axis_v4.py'),fragments24_correction=corr);write_gz(ap,geo)
        row.update(measurement=m,axis=a,current_F3_called=True,F3_record_sha256=sha(fp),F3_fields_sha256=x['field_sha256'],groups_file_sha256=sha(gp),geometry_file_sha256=sha(ap),fragments24_correction=corr);write_gz(cp,row)
        added.append(dict(id=id,frame=i,N=m['N'],axis_shift_px=q['axis_shift_px'],count_sha256=sha(cp),current_boundary_reinitialized=False,identity_policy=corr['identity_policy']))
    rec['previous_receipt_sha256']=sha(receipt_path);rec['promoted']+=added;rec['identity_recovery_code_sha256']=sha(__file__)
    rec['rejected']=[q for q in rec['rejected'] if not any(q['id']==a['id'] and q['frame']==a['frame'] for a in added)];dump(receipt_path,rec)
    for id in sorted(set(a['id'] for a in added)):receipt(next(x for x in plan() if x['id']==id))
    for name in ('f3_v2_receipt.json','group_v2_receipt.json','geometry_v5_receipt.json','count_receipt.json'):
        p=ROOT/name;v=read(p);v['current_fragments24_correction_receipt_sha256']=sha(receipt_path);dump(p,v)
    print('CURRENT IDENTITY RECOVERY',added,flush=True)

if __name__=='__main__':main()
