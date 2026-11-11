"""Validate new-video plumbing against sealed reference outputs before transfer."""
import time
import cv2,numpy as np
import transfer_engine as e
def compare(a,b,path=''):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a:compare(a[k],b[k],path+'/'+k)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),(path,len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert np.isclose(a,b,rtol=1e-9,atol=1e-8),(path,a,b)
    else:assert a==b,(path,a,b)
def main():
    root=e.ROOT.parent;out=e.ROOT/'parity';out.mkdir(exist_ok=True);checks=[]
    e.verify_baseline()
    for f in [126,240,260]:
        old=e.read(root/'LC1_boundary_BG3_point_global_20261107_v2/full_001/frames'/f'{f:06d}.json')
        reference=e.read(root/'LC1_boundary_BG4_local_temporal_20261107_v1/full_003/frames'/f'{f:06d}.json')
        with np.load(root/'LC1_boundary_BG3_point_global_20261107_v2/cache'/f'{f:06d}.npz',allow_pickle=False) as data:data={k:data[k] for k in data.files}
        got=e.plain(e.spatial_bg4(old,data,reference['camera_motion']))
        for key in ['observation','reasons','BG4','intersection_y_px','region','s1']:compare(got[key],reference[key],key)
        checks.append(dict(frame=f,stage='BG4_numeric_tail',passed=True));print('BG4 parity',f,flush=True)
    # One source-frame fresh extraction validates masks, point coordinates and all
    # supplied-01 raw fields against the previously byte-bound reference cache.
    source=e.ROOT.parents[3]/'双相机同步/dual_20260924_111500/left_DA9627065.avi'
    cap=cv2.VideoCapture(str(source));cap.set(1,260);ok,im=cap.read();assert ok;cap.release()
    r=e.read(root/'LC1_boundary_BG4_local_temporal_20261107_v1/full_003/frames/000260.json')
    tick=time.perf_counter();region=e.segment(im);fresh=e.extract_current(im,region['surface_mask'],r['s1'])
    with np.load(root/'LC1_boundary_BG3_point_global_20261107_v2/cache/000260.npz',allow_pickle=False) as npz:
        for key in ['points','surface','exclusion','broad','clean','body','x0','y0','pitch']:
            assert np.array_equal(fresh[key],npz[key]),key
    checks.append(dict(frame=260,stage='fresh_supplied_01_extractor_all_fields',passed=True,seconds=time.perf_counter()-tick))
    ec=e.measure_v3(im,260,record=r)
    ref=e.read(root/'LC1_ECW_20261107_v1/v3_full/records.json')[260]
    for key in ['rope','boundary','valid','center_gap_proxy_px','center_gap_width','sensitivity_bound_px']:compare(ec[key],ref[key],key)
    checks.append(dict(frame=260,stage='ECW_v3_current_pixels',passed=True))
    e.dump(out/'receipt.json',dict(passed=True,checks=checks,baseline_manifest_sha256=e.sha(e.ROOT/'baseline/manifest.json'),adapter_sha256=e.sha(e.ROOT/'transfer_engine.py')))
    print('All parity checks passed',flush=True)
if __name__=='__main__':main()
