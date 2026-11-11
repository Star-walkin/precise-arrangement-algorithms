import time,hashlib,csv
import cv2
from ecw_v2 import ROOT,SOURCE,measure_v2,dump

def main():
    out=ROOT/'v2_full';out.mkdir(exist_ok=True);cap=cv2.VideoCapture(str(SOURCE));rows=[];t0=time.perf_counter()
    stamp=list(csv.DictReader((SOURCE.parent/'left_frames.csv').open(encoding='utf-8-sig')))
    assert len(stamp)==601 and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==601
    for f in range(601):
        ok,im=cap.read();assert ok
        r=measure_v2(im,f);r['host_time_s']=float(stamp[f]['host_time_s']);rows.append(r)
        if f%50==0:
            print(f,'valid',sum(r['valid'] for r in rows),'elapsed',round(time.perf_counter()-t0,1),flush=True)
    cap.release();dump(out/'records.json',rows)
    dump(out/'run_receipt.json',dict(frames=len(rows),source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                                  elapsed_seconds=time.perf_counter()-t0,code_sha256=hashlib.sha256((ROOT/'ecw_v2.py').read_bytes()).hexdigest()))
if __name__=='__main__':main()
