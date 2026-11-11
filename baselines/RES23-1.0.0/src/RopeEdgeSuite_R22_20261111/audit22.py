import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import concurrent.futures as cf,subprocess,cv2,imageio_ffmpeg
from common22 import *
from events22 import analyze as motion_analyze
from motor22 import analyze as motor_analyze

def inspect(item):
    id=item['id'];old=read(R21/'delivery'/id/'frames.json');new=read(ROOT/'delivery'/id/'frames.json')
    summary=read(ROOT/'delivery'/id/'summary.json');rr=read(ROOT/'rotation'/id/'receipt.json')
    assert len(old)==len(new)==item['planned_frames']==rr['decoded_pixel_verified']
    for i,(a,b) in enumerate(zip(old,new)):
        for k,v in a.items():assert b[k]==v,(id,i,k)
        s=b['state22']
        if s['evidence_mode']=='history_hold':
            assert 0<s['age_s']<=.5+1e-8 and not s['current_evidence'] and s['uses_prediction']
        if s['current_evidence']:assert a['available'] and a['current_coil_side_axis'] and not a['boundary_occluded']
        assert not s['physical_contact_confirmed']
        if b['motor22'].get('confirmation_s') is not None:assert b['time_s']>=b['motor22']['confirmation_s']
    before=read(R21/'events'/f'{id}.json')['events'];after=summary['near_events']
    assert [{k:v for k,v in e.items() if k!='pose_transition'} for e in before]==after
    features=np.load(rr['features_path'])['features'];overlap=read(R21/'row_overlap'/f'{id}.json')
    for count in sorted(set([min(len(new),x) for x in [30,130,200,251,260,274,278,281,350,len(new)]])):
        prefix=new[:count];motion=motion_analyze(prefix,item['fps'])
        assert motion['frames']==[x['state22'] for x in prefix],(id,count,'motion causality')
        motor=motor_analyze(prefix,overlap,motion,features[:count],item['fps'])
        assert motor['frames']==[x['motor22'] for x in prefix],(id,count,'motor causality')
    video=summary['video'];assert sha(video)==summary['video_sha256']
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-v','error','-i',video,'-map','0:v:0','-f','null','-','-progress','pipe:1','-nostats']
    proc=subprocess.run(cmd,capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));assert proc.returncode==0,proc.stderr
    counts=[int(x.split('=')[1]) for x in proc.stdout.splitlines() if x.startswith('frame=')];assert counts[-1]==len(new)
    cap=cv2.VideoCapture(video);assert abs(cap.get(cv2.CAP_PROP_FPS)-item['fps'])<.01;assert (cap.get(3),cap.get(4))==(1920,1080)
    base=cv2.VideoCapture(summary['baseline_video']);psnrs=[]
    for i in sorted(set([0,len(new)//2,len(new)-1])):
        cap.set(1,i);base.set(1,i);ok,a=cap.read();ok2,b=base.read();assert ok and ok2
        aa=a[144:867,852:1892];bb=b[144:867,852:1892]
        psnr=float(cv2.PSNR(aa,bb));assert psnr>35,(id,i,psnr);psnrs.append(psnr)
    cap.release();base.release()
    result=dict(id=id,frames=len(new),all_R21_measurement_fields_exactly_equal=True,source_pixel_verified=rr['decoded_pixel_verified'],
        near_event_timestamps_unchanged=True,causal_prefix_checks_passed=True,full_video_decode=True,video_sha256=summary['video_sha256'],
        image_panel_sample_psnr_min=min(psnrs),old_changes=summary['old_state_changes'],new_changes=summary['new_state_changes'],
        motor_recurrence_candidates=sum(e['recurrence_supported'] for e in summary['motor_events']))
    dump(ROOT/'audit'/f'{id}.json',result);print('audit passed',id,flush=True);return result

def tests():
    scripts=[ROOT/'test_temporal22.py',ROOT/'test_motor22.py',R21/'test_contracts21.py',R20/'test_measurement_contracts.py']
    result=[]
    for script in scripts:
        proc=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,errors='replace')
        (ROOT/(script.stem+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf8')
        assert proc.returncode==0,(str(script),proc.stdout,proc.stderr)
        result.append(dict(script=str(script),passed=True,log=str(ROOT/(script.stem+'.log'))))
    return result

if __name__=='__main__':
    cv2.setNumThreads(1);t=tests()
    with cf.ThreadPoolExecutor(3) as pool:results=list(pool.map(inspect,plan()))
    audit=dict(status='passed',frames=sum(x['frames'] for x in results),videos=results,tests=t,
        old_state_changes=sum(x['old_changes'] for x in results),new_state_changes=sum(x['new_changes'] for x in results),
        new_module_prefix_causality_verified=True,whole_pipeline_live_calibration_verified=False,
        physical_first_turn_ground_truth_available=False,framewise_ground_truth_accuracy_verified=False)
    dump(ROOT/'regression_audit.json',audit)
    print('ALL',audit['frames'],audit['old_state_changes'],audit['new_state_changes'],flush=True)
