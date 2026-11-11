"""Verify final products, exact per-frame provenance and count identities."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time,subprocess,collections,concurrent.futures as cf,cv2,imageio_ffmpeg
from common20 import *

def one(item):
    id=item['id'];out=ROOT/'delivery'/id;summary=read(out/'summary.json');rows=read(out/'frames.json');n=item['planned_frames']
    assert len(rows)==n and [x['frame'] for x in rows]==list(range(n))
    cal=read(ROOT/'calibration'/f'{id}.json');calhash=sha(ROOT/'calibration'/f'{id}.json')
    assert sha(summary['video'])==summary['video_sha256']
    assert sha(item['source'])==item['source_sha256']
    tests=collections.Counter();issues=[]
    for phase in cal['epochs']:
        if phase['d'] is None:continue
        accepted=phase['far_guard_accepted_frames'];assert accepted
        audits=phase['current_F3_five_rope_audit']
        for i in accepted:
            audit=[x for x in audits if x['frame']==i and x['phase'].startswith('fixed_column_audit_')][-1]
            assert audit['passed'] and audit['count']['N_interval'][0]>=5
            assert abs(audit['count']['d']-phase['d'])<1e-8
            tests['fixed_d_far_reference_reverified']+=1
        original=4095-phase['d'] if item['camera']=='right' else phase['d']
        assert abs(original-phase['d_original_image'])<1e-8
    for i,row in enumerate(rows):
        cp=ROOT/'counts_v1'/id/'frames'/f'{i:06d}.json.gz'
        with gzip.open(cp,'rt',encoding='utf8') as f:c=json.load(f)
        assert sha(cp)==row['count_record_sha256'] and c['calibration_file_sha256']==calhash
        frp=ROOT/'f3_v2'/id/'records'/f'{i:06d}.json';fr=read(frp)
        gp=ROOT/'groups_v2'/id/'frames'/f'{i:06d}.json.gz';geo=ROOT/'geometry_v5'/id/'frames'/f'{i:06d}.json.gz'
        assert c['F3_record_sha256']==sha(frp) and c['groups_file_sha256']==sha(gp) and c['geometry_file_sha256']==sha(geo)
        assert c['source_pixel_sha256']==row['source_pixel_sha256']==fr['source_pixel_sha256']
        correction=c.get('fragments24_correction') or c.get('fragments23_correction')
        if correction:
            assert sha(correction['frontend_override_path'])==correction['frontend_override_sha256']
            with gzip.open(correction['frontend_override_path'],'rt',encoding='utf8') as f:front=json.load(f)
            original=record(id,i)
            assert front['track']==original['track'] and front['source_pixel_sha256']==c['source_pixel_sha256']
            assert front['boundary']['fit_valid'] and (fr.get('fragments24_correction') or fr.get('fragments23_correction'))==correction
            assert correction['boundary_code_sha256']==sha(ROOT/'current_boundary23.py')
            segment='region_fragments24.py' if c.get('fragments24_correction') else 'region_fragments23.py'
            assert correction['segment_code_sha256']==sha(ROOT/segment)
            tests['current_fragment_repair_provenance_verified']+=1
        if fr['called']:assert sha(fr['fields_path'])==fr['fields_sha256']==c['F3_fields_sha256']
        with gzip.open(gp,'rt',encoding='utf8') as f:g=json.load(f)
        with gzip.open(geo,'rt',encoding='utf8') as f:a=json.load(f)
        assert a['groups_file_sha256']==sha(gp) and g['raw_fields_sha256']==fr['fields_sha256']
        assert a['geometry_code_sha256']==sha(ROOT/'geometry22.py') and a['axis_code_sha256']==sha(ROOT/'coil_edge_axis_v4.py')
        m=c['measurement'];central=m.get('central') or {};curves=central.get('curves',[]);cross=central.get('crossings',[])
        ids=[x['boundary_id'] for x in cross];assert len(ids)==len(set(ids));curveids={x['boundary_id'] for x in curves};assert set(ids)<=curveids
        ownership=collections.Counter(int(k) for x in curves for k in x['point_ids'])
        if any(v>1 for v in ownership.values()):issues.append(dict(frame=i,type='duplicate_point_ownership',points=sum(v>1 for v in ownership.values())))
        if m['available']:
            assert np.isfinite(m['N']) and m['N_interval'][0]<=m['N']<=m['N_interval'][1]
            assert m['d'] is not None and fr['called'] and len(central['crossings'])>=4
            if m['complete_F3_phase']:
                assert central['phase_N'] is not None and not central['unresolved_local_classes']
                # These failures must never masquerade as a complete observed chain.
                assert not any(x in central['reasons'] for x in ('unsupported_count_gap_gt_1_6_pitch','unknown_or_excluded_region_on_count_segment','sub_pitch_competing_grooves_after_deduplication'))
        assert not m['physical_contact_confirmed'] and not row['contact']['physical_contact_confirmed'] and not row['contact']['uses_prediction']
        tests['per_frame_provenance_and_count_contracts']+=1
    # Decode the entire delivered video; metadata alone does not establish that
    # its last frames can be played or that an incomplete encode is usable.
    exe=imageio_ffmpeg.get_ffmpeg_exe()
    proc=subprocess.run([exe,'-hide_banner','-loglevel','error','-nostdin','-threads','2','-i',summary['video'],'-map','0:v:0','-an','-f','null','NUL','-progress','pipe:1','-nostats'],capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert proc.returncode==0 and not proc.stderr.strip(),proc.stderr
    frames=[int(x.split('=')[1]) for x in proc.stdout.splitlines() if x.startswith('frame=')];assert frames[-1]==n
    cap=cv2.VideoCapture(summary['video']);fps=cap.get(cv2.CAP_PROP_FPS);w=cap.get(3);h=cap.get(4);cap.release()
    assert abs(fps-item['fps'])<1e-3 and (w,h)==(1920,1080)
    result=dict(id=id,frames=n,decoded_video_frames=frames[-1],fps=fps,checks=dict(tests),issues=issues,passed=not issues)
    dump(out/'audit.json',result);print(id,'AUDIT',result['passed'],flush=True);return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    if args.wait:
        while not (ROOT/'delivery_receipt.json').exists():time.sleep(5)
    assert read(ROOT/'delivery_receipt.json')['status']=='complete'
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(one,plan()))
    dump(ROOT/'final_audit.json',dict(passed=all(x['passed'] for x in results),videos=results,frames=sum(x['frames'] for x in results),
        scope='technical_integrity_not_independent_recognition_accuracy',code_sha256=sha(__file__)))
