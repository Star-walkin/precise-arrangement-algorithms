import sys,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from engine import FrameEngine,anchor_job,cv2,np,V1
def main():
    out=HERE/'probe_002';out.mkdir(exist_ok=False)
    trace=json.loads((V1/'source_trace/source_trace.json').read_text(encoding='utf8'));receipt=[]
    for name in ['dual_20260923_101517__right','L_run_20260923_01__left']:
        v=next(v for v in trace['videos'] if v['video_id']==name)
        cap=cv2.VideoCapture(v['canonical_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,v['earliest_dataset_anchor']['source_frame'])
        ok,bgr=cap.read();assert ok
        eng=FrameEngine(anchor_job(v),0.,threads=2)
        gray=cv2.cvtColor(cv2.resize(bgr,(2048,1500),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
        before=time.perf_counter();tr,d=eng.observer.detect(gray);seconds=time.perf_counter()-before
        shifted=cv2.warpAffine(gray,np.array([[1,0,0],[0,1,2]],np.float32),(2048,1500))
        shifted_tr,sd=eng.observer.detect(shifted);black,bd=eng.observer.detect(np.zeros_like(gray))
        assert len(black)==0
        panel=cv2.resize(bgr,(2048,1500))
        for c in tr:cv2.polylines(panel,[np.rint(np.asarray(c['points_xy'])+[0,525]).astype(np.int32)],False,(150,150,150),1,cv2.LINE_AA)
        cv2.imencode('.jpg',panel)[1].tofile(str(out/(name+'.jpg')))
        old_y=[];delta=[]
        # Compare center y at fixed interior columns, nearest fresh row; no IDs assumed.
        for x in [850.,1000.,1150.]:
            a=sorted(np.interp(x,np.asarray(c['points_xy'])[:,0],np.asarray(c['points_xy'])[:,1]) for c in tr if c['points_xy'][0][0]<=x<=c['points_xy'][-1][0])
            b=sorted(np.interp(x,np.asarray(c['points_xy'])[:,0],np.asarray(c['points_xy'])[:,1]) for c in shifted_tr if c['points_xy'][0][0]<=x<=c['points_xy'][-1][0])
            for y in a:
                if b:delta.append(min(b,key=lambda z:abs(z-y-2))-y)
        current=eng.step(bgr,0.);current2=eng.step(bgr,1/30)
        row=dict(video=name,detector_seconds=seconds,detector=d,shifted_detector=sd,black_centers=len(black),
            vertical_shift_original_px=4.,median_measured_shift_original_px=float(np.median(delta))*2 if delta else None,
            total_engine_seconds=current['diagnostics']['seconds'],consecutive_engine_seconds=current2['diagnostics']['seconds'],
            forward_count=current['diagnostics']['forward_count'],candidate_count=len(current['raw_candidates']))
        receipt.append(row);print(json.dumps(row,ensure_ascii=True),flush=True);cap.release()
    (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
