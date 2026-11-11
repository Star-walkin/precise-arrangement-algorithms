import time,hashlib
import cv2
from ecw_v3 import ROOT,SOURCE,measure_v3,read,dump
def main():
    out=ROOT/'v3_full';out.mkdir(exist_ok=True);old=read(ROOT/'v2_full/records.json');cap=cv2.VideoCapture(str(SOURCE));rows=[];t=time.perf_counter()
    for f,v2 in enumerate(old):
        ok,a=cap.read();assert ok and hashlib.sha256(a.tobytes()).hexdigest()==v2['source_pixel_sha256']
        r=measure_v3(a,f,rope=v2.get('rope'));r['host_time_s']=v2['host_time_s'];rows.append(r)
        if f%100==0:print(f,'valid',sum(x['valid'] for x in rows),'elapsed',round(time.perf_counter()-t,1),flush=True)
    cap.release();dump(out/'records.json',rows)
    dump(out/'run_receipt.json',dict(frames=len(rows),elapsed_s=time.perf_counter()-t,source=str(SOURCE),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),v2_rope_cache_sha256=hashlib.sha256((ROOT/'v2_full/records.json').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256((ROOT/'ecw_v3.py').read_bytes()).hexdigest()))
if __name__=='__main__':main()
