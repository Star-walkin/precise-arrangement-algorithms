import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from common21 import *
import concurrent.futures as cf,subprocess,time,collections,imageio_ffmpeg
from row_overlap21 import observe,summarize

def inspect(item):
    id=item['id'];out=ROOT/'delivery'/id;new=read(out/'frames.json');old=read(R20/'delivery'/id/'frames.json');summary=read(out/'summary.json')
    assert len(new)==len(old)==item['planned_frames'];changed=[];lost=[];duplicates=[];overlaps=[];same_fixed=0;old_contact_changes=[];unknown_dynamic=[]
    for n,b in zip(new,old):
        i=n['frame'];assert n['source_pixel_sha256']==b['source_pixel_sha256'];cp=Path(n['measurement_record_path']);assert sha(cp)==n['measurement_record_sha256']
        row=gzread(cp);m=row['measurement'];c=m.get('central') or {}
        assert m.get('physical_contact_confirmed') is False
        if b['available'] and not n['available']:lost.append(i)
        if n['fixed_d'] is not None:
            for key in ('available','N','N_interval','d','d_interval','complete_F3_phase','observed_grooves'):
                assert n.get(key)==b.get(key),(id,i,key,n.get(key),b.get(key))
            same_fixed+=1
            if n['contact']['state']!=b['contact']['state']:old_contact_changes.append(i)
        elif n['available']:
            assert not m['fixed_d_calibrated'];assert m['measurement_mode']=='current_dynamic_column';assert row['axis']['current_coil_side_measured']
            assert len((row.get('dynamic_diagnostic') or {}).get('gate_reasons',[]))==0
            changed.append(i)
        crossings=c.get('crossings',[])
        if crossings:
            ids=[str(x['boundary_id']) for x in crossings];assert len(ids)==len(set(ids)),(id,i,'duplicate crossing')
            assert set(ids)<={str(x['boundary_id']) for x in c['curves']}
            point_ids=[int(p) for curve in c['curves'] for p in curve.get('point_ids',[])]
            assert len(point_ids)==len(set(point_ids)),(id,i,'point assigned to multiple curves')
        if n['near_event']['current_evidence']:
            assert n['available'] and n['current_coil_side_axis'] and not n['boundary_occluded']
        assert not n['near_event']['physical_contact_confirmed'];assert not n['contact']['physical_contact_confirmed']
        overlaps.append(dict(frame=i,time_s=n['time_s'],evidence=observe(m,row['axis'])))
    candidates=summarize(overlaps,item['fps']);dump(ROOT/'row_overlap'/f'{id}.json',dict(candidates=candidates,frames=overlaps,physical_layer_change_confirmed=False))
    video=Path(summary['video']);assert sha(video)==summary['video_sha256']
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-v','error','-i',str(video),'-map','0:v:0','-f','null','-','-progress','pipe:1','-nostats']
    result=subprocess.run(command,capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));assert result.returncode==0,result.stderr
    frames=[int(x.split('=')[1]) for x in result.stdout.splitlines() if x.startswith('frame=')];assert frames[-1]==item['planned_frames'],(id,frames[-1])
    iterator=imageio_ffmpeg.read_frames(str(video));meta=next(iterator);iterator.close();assert meta['size']==(1920,1080) and abs(meta['fps']-item['fps'])<.01
    result=dict(id=id,frames=len(new),old_available=sum(r['available'] for r in old),new_available=sum(r['available'] for r in new),
        fixed_frames_exactly_equal=same_fixed,lost_available_frames=lost,added_dynamic_frames=changed,old_contact_state_changes=old_contact_changes,
        physical_confirmation_claims=0,source_frame_hashes_equal=True,full_output_decode=True,video_frames=frames[-1],
        near_events=summary['near_events'],first_row_overlap_candidates=candidates)
    assert not lost and not old_contact_changes
    dump(ROOT/'audit'/f'{id}.json',result);print('audit passed',id,'added',len(changed),flush=True);return result

if __name__=='__main__':
    jobs=list(plan());pending={};done_rows=[]
    with cf.ProcessPoolExecutor(max_workers=2) as pool:
        while jobs or pending:
            for item in list(jobs):
                if len(pending)>=2:break
                sp=ROOT/'delivery'/item['id']/'summary.json'
                # final_measurements distinguishes final rendering from earlier diagnostic render.
                if sp.exists() and (ROOT/'final_measurements'/item['id']/'summary.json').exists():
                    new=read(ROOT/'delivery'/item['id']/'frames.json')
                    if all('measurement_record_path' in n for n in new):jobs.remove(item);pending[pool.submit(inspect,item)]=item['id']
            if not pending:time.sleep(2);continue
            done,_=cf.wait(pending,timeout=2,return_when=cf.FIRST_COMPLETED)
            for f in done:pending.pop(f);done_rows.append(f.result())
    ref=read(ROOT/'human_reference.json');target=next(r for r in done_rows if r['id']==ref['id']);event=next(e for e in target['near_events'] if e['departure_s'] is not None)
    evaluation=dict(reference=ref,near_onset_s=event['start_s'],near_confirmation_s=event['entry_confirmed_s'],departure_s=event['departure_s'],
        near_confirmation_error_s=event['entry_confirmed_s']-ref['near_s'],departure_error_s=event['departure_s']-ref['leave_s'],
        layer_entry_confirmed=False,first_row_overlap_candidates=target['first_row_overlap_candidates'],
        human_times_used_by_detector=False,independent_layer_change_ground_truth_available=False)
    dump(ROOT/'regression_audit.json',dict(status='passed_technical_regression',frames=sum(r['frames'] for r in done_rows),videos=done_rows,
        evaluation=evaluation,all_fixed_frames_unchanged=True,all_old_available_frames_preserved=True,
        framewise_ground_truth_accuracy_verified=False,physical_layer_change_confirmed=False))
