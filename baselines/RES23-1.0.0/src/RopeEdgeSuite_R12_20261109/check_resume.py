from run_frontend import *

def main():
    cv2.setNumThreads(1)
    out=ROOT/'frontend/dual_20260923_113745_v2';items=[record(out,i) for i in range(110)]
    cap=cv2.VideoCapture(e.read(out/'binding.json')['source']);cap.set(1,109);ok,im=cap.read();assert ok and e.pixelsha(im)==items[-1]['source_pixel_sha256']
    mask=cv2.imdecode(np.fromfile(out/'regions/000109.png',np.uint8),0);a=Tracker(mask)
    restore_current(a,im,items);ok,im=cap.read();assert ok
    old=record(out,110);mask=cv2.imdecode(np.fromfile(out/'regions/000110.png',np.uint8),0)
    a.set_region(dict(record=old['region'],surface_mask=mask));s=a.step(im,11.,110);before=old['track']
    checks={k:s[k]==before[k] for k in ['valid','line_abc','line_points_xy','supported_segments_xy','seed_xy_original']}
    assert all(checks.values()),checks
    cap.release();dump(ROOT/'resume_parity_receipt.json',dict(passed=True,restored_frame=109,replayed_frame=110,checks=checks))
    print(json.dumps(checks))
if __name__=='__main__':main()
