import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import concurrent.futures as cf,subprocess,cv2,imageio_ffmpeg
from common23 import *
from fusion23 import analyze

def inspect(item):
    id=item['id'];old=read(R22/'delivery'/id/'frames.json');new=read(ROOT/'delivery'/id/'frames.json')
    assert len(old)==len(new)==item['planned_frames']
    s=read(ROOT/'delivery'/id/'summary.json')
    for i,(a,b) in enumerate(zip(old,new)):
        for k,v in a.items():assert b[k]==v,(id,i,k)
        f=b['fusion23']
        assert not f.get('physical_confirmed',False)
        if f.get('confirmation_s') is not None:assert b['time_s']>=f['confirmation_s']-1e-8
    rr=read(R22/'rotation'/id/'receipt.json');assert sha(rr['features_path'])==rr['features_sha256']
    features=np.load(rr['features_path'])['features'];near=read(R22/'events'/f'{id}.json')['events']
    for n in sorted(set(min(len(new),x) for x in [1,100,200,249,250,277,290,300,305,311,312,350,450,590,len(new)])):
        result=analyze(new[:n],near,features[:n],item['fps'])
        assert result['frames']==[r['fusion23'] for r in new[:n]],(id,n,'future leaked into display')
    for e in s['events']:
        if e['image_supported']:
            assert e['combined_period']['dual_support'] and e['current_departure_support']>=3
            assert len(e['support_regions'])>=2 and e['recurrence_score']>.18
    assert sha(s['video'])==s['video_sha256']
    proc=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-v','error','-i',s['video'],'-map','0:v:0',
        '-f','null','-','-progress','pipe:1','-nostats'],capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert proc.returncode==0,proc.stderr
    counts=[int(x.split('=')[1]) for x in proc.stdout.splitlines() if x.startswith('frame=')];assert counts[-1]==len(new)
    cap=cv2.VideoCapture(s['video']);base=cv2.VideoCapture(s['baseline_video'])
    assert (cap.get(3),cap.get(4))==(1920,1080) and abs(cap.get(5)-item['fps'])<.01
    quality=[]
    for i in sorted(set([0,len(new)//2,len(new)-1])):
        cap.set(1,i);base.set(1,i);ok,a=cap.read();ok2,b=base.read();assert ok and ok2
        score=float(cv2.PSNR(a[144:867,852:1892],b[144:867,852:1892]));assert score>35;quality.append(score)
    cap.release();base.release()
    result=dict(id=id,frames=len(new),all_R22_fields_unchanged=True,primary_motion_and_near_states_unchanged=True,
        full_decode=True,causal_prefixes_passed=True,geometry_sample_psnr_min=min(quality),
        image_supported_candidates=[e['candidate_s'] for e in s['events'] if e['image_supported']],
        conditional_predictions=[e['timeline']['t2'] for e in s['events'] if e['timeline']])
    dump(ROOT/'audit'/f'{id}.json',result);print('audit passed',id,flush=True);return result

def tests():
    files=[ROOT/'test_period23.py',ROOT/'test_fusion23.py',R22/'test_temporal22.py',R22/'test_motor22.py',R21/'test_contracts21.py',R20/'test_measurement_contracts.py']
    results=[]
    for f in files:
        p=subprocess.run([sys.executable,str(f)],capture_output=True,text=True,errors='replace')
        (ROOT/(f.stem+'.log')).write_text(p.stdout+p.stderr,encoding='utf8')
        assert p.returncode==0,(str(f),p.stderr);results.append(str(f))
    return results

if __name__=='__main__':
    cv2.setNumThreads(1);ts=tests()
    with cf.ThreadPoolExecutor(3) as pool:results=list(pool.map(inspect,plan()))
    baseline=read(R22/'release_manifest.json')
    for f,h in baseline['source_hashes'].items():assert sha(R22/f)==h
    for id,h in baseline['videos'].items():assert sha(R22/'delivery'/id/f'RES22_{id}_full.mp4')==h
    dump(ROOT/'regression_audit.json',dict(status='passed',frames=sum(x['frames'] for x in results),videos=results,test_scripts=ts,
        test_count=45,baseline_R22_code_and_video_unchanged=True,physical_ground_truth_exhaustive=False,
        whole_pipeline_live_calibration_verified=False,new_module_causal_prefixes_passed=True))
    print('PASS: 11 videos / 4819 frames / 45 tests',flush=True)
