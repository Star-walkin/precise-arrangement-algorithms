"""Read-only diagnosis of the user-named open-development counterexample.
The wider profile scan is a diagnostic candidate pool, never a target label.
"""
from pathlib import Path
import gzip,hashlib,importlib.util,json,sys
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
H2=BASE/"postprocess_b_20261030_v2";H3=BASE/"postprocess_b_20261031_v3"
ID="R_static_20260922_01__f000095__right"
def load(p):
    with (gzip.open(p,"rt",encoding="utf-8") if str(p).endswith(".gz") else open(p,encoding="utf-8-sig")) as f:return json.load(f)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def save(p,a):cv2.imencode(p.suffix,a)[1].tofile(str(p))
def sample(a,p):
    p=np.asarray(p,np.float32);sh=p.shape[:-1]
    # Preserve a 2-D sampling grid; flattening this strip exceeds OpenCV32767.
    if p.ndim==2:p=p[None]
    return cv2.remap(a,p[...,0],p[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(sh)
def panel(im,title,subtitle):
    out=Image.new("RGB",(im.shape[1],im.shape[0]+100),(245,247,250));out.paste(Image.fromarray(im[:,:,::-1]),(0,100));d=ImageDraw.Draw(out)
    d.text((16,8),title,font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",29),fill=(20,30,40))
    d.text((16,54),subtitle,font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",23),fill=(50,65,80))
    return np.asarray(out)[:,:,::-1].copy()
def draw_poly(im,p,color,width=2):
    cv2.polylines(im,[np.rint(np.asarray(p)).astype(np.int32)],False,color,width,cv2.LINE_AA)

def main():
    config=load(H3/"inputs.json");job=next(j for j in config["jobs"] if j["id"]==ID)
    assert job["dataset_use_status"]=="unlabeled_open_development_no_new_test_split" and "20260925" not in job["image_path"]
    records=[load(v/"all_001/记录"/(ID+".json.gz")) for v in (H2,H3)]
    entries=[r["entries"][r["selected_entry"]] for r in records]
    assert entries[0]["seed"]==entries[1]["seed"]
    original=Path(job["image_path"]);assert sha(original)==job["image_sha256"]
    bgr=cv2.imdecode(np.fromfile(original,np.uint8),cv2.IMREAD_COLOR)
    fullhalf=cv2.resize(bgr,(2048,1500),interpolation=cv2.INTER_AREA);crop=fullhalf[525:]
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    rp=Path(entries[0]["response_file"]);before=sha(rp);assert before==entries[1]["response_sha256"]
    with np.load(rp,allow_pickle=False) as z:heat=z["response"].copy()
    # This module contains only image/numeric processing, inspected before import.
    source=H2/"free_candidates.py";spec=importlib.util.spec_from_file_location("_readonly_H2_free",source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    a7p=BASE.parents[1]/"数据资产/05_派生与实验结果/camera_condition_free_straight_20261030_v7/all_001/记录/right_camera"/(ID+".json")
    a7=load(a7p)["result"];f2,f3=[e["free"] for e in entries]
    candidate=next(c for c in a7["candidates"] if c["candidate_id"]==f2["a7_candidate_id"])
    seed=(np.asarray(job["seeds"][0]["xy"])-[0,1050])*.5
    direction=np.asarray(f2["axis_direction_xy"]);co=(np.asarray(candidate["origin_xy"])-[0,1050])*.5
    origin=co+((seed[0]-co)@direction)*direction;normal=np.array([-direction[1],direction[0]])
    width=float(f2["width_half_px"]);t=np.arange(0,1000.0001,3.);axis=origin+t[:,None]*direction
    hf=cv2.GaussianBlur(heat,(0,0),.7);gf=cv2.GaussianBlur(gray,(0,0),.7);valid=np.ones_like(heat)
    scans={}
    for name,shifts in [("H2_narrow_search",np.linspace(-.32,.32,7)*width),("diagnostic_wide_search",np.arange(-2*width,2*width+.01,.5))]:
        proposals=(axis[:,None]+normal[None,None]*shifts[None,:,None]).reshape(-1,2)
        ev=m._profiles(gf,hf,valid,proposals,direction,width)
        scores=ev["score"].reshape(len(t),-1)-.08*(shifts[None]/width)**2
        choice=scores.argmax(1);ix=np.arange(len(t))*len(shifts)+choice
        measured=proposals[ix];selected={k:np.asarray(v)[ix] for k,v in ev.items()}
        joint=selected["heat_profile_supported"]&(selected["raw_score"]>=.25)
        scans[name]=dict(points_xy=measured.tolist(),normal_offset_original_px=(shifts[choice]*2).tolist(),
                         joint_supported=joint.tolist(),evidence={k:v.tolist() for k,v in selected.items()})
    axis3=np.asarray(f3["points_xy"])[0]+t[:,None]*np.asarray(f3["axis_direction_xy"])
    ev3=m._profiles(gf,hf,valid,axis3,np.asarray(f3["axis_direction_xy"]),width)
    scans["H3_fixed_axis"]=dict(points_xy=axis3.tolist(),joint_supported=(ev3["heat_profile_supported"]&(ev3["raw_score"]>=.25)).tolist(),evidence={k:np.asarray(v).tolist() for k,v in ev3.items()})
    windows=[]
    for lo,hi in [(0,600),(600,828),(840,1200),(1200,1600),(1600,2000)]:
        use=(t*2>=lo)&(t*2<hi)
        stats={}
        for name,s in scans.items():
            e=s["evidence"]
            stats[name]=dict(center_mean=float(np.mean(np.asarray(e["center"])[use])),
                             joint_supported_fraction=float(np.mean(np.asarray(s["joint_supported"])[use])),
                             raw_score_mean=float(np.mean(np.asarray(e["raw_score"])[use])))
            if "normal_offset_original_px" in s:stats[name]["median_normal_offset_original_px"]=float(np.median(np.asarray(s["normal_offset_original_px"])[use]))
        windows.append(dict(progress_original_px=[lo,hi],profiles=stats))
    # Original/response/H2/H3 share the same unmodified full crop.
    pictures=[crop.copy(),cv2.applyColorMap(np.rint(heat*255).astype(np.uint8),cv2.COLORMAP_TURBO)]
    for e in entries:
        im=crop.copy();draw_poly(im,e["free"]["points_xy"],(40,230,40),2)
        c=e["connection"]["candidates"][0];draw_poly(im,c["points_xy"],(255,50,230),2)
        cv2.circle(im,tuple(np.rint(e["free"]["points_xy"][-1]).astype(int)),7,(20,240,240),2)
        if c.get("join_start_xy"):cv2.circle(im,tuple(np.rint(c["join_start_xy"]).astype(int)),8,(255,255,255),2)
        pictures.append(im)
    # Fixed 0..1 scale; never per-image minmax.
    bar=cv2.applyColorMap(np.linspace(255,0,300).astype(np.uint8)[:,None],cv2.COLORMAP_TURBO)
    pictures[1][40:340,1930:1960]=np.repeat(bar,30,axis=1)
    for y,label in [(50,"1"),(190,".5"),(335,"0")]:cv2.putText(pictures[1],label,(1970,y),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    labels=[("原图","完整原图搜索ROI，保留真实零灰度"),("冻结B响应","响应强度0–1；高响应不是身份真值"),("H2保存路径","绿：自由候选；紫：主路径；黄圈：自由末点"),("H3保存路径","白圈：提前接桥起点；黄圈：自由候选末点")]
    save(HERE/(ID+"__original_B_H2_H3.jpg"),np.concatenate([panel(im,*title) for im,title in zip(pictures,labels)],axis=1))
    # Fixed spatial crop covers the visible diagonal rope and premature join.
    box=[1400,2100,3300,2600];x0,y0,x1,y1=box;qbox=[x0//2,(y0-1050)//2,x1//2,(y1-1050)//2]
    local=[im[qbox[1]:qbox[3],qbox[0]:qbox[2]] for im in pictures]
    save(HERE/(ID+"__fixed_original_crop.jpg"),np.concatenate([panel(im,*title) for im,title in zip(local,labels)],axis=1))
    # Straightened-axis strip is a diagnostic sampling display, not a new input.
    offsets=np.arange(-40,40.01,.5);q=axis[:,None]+normal[None,None]*(offsets[None,:,None]*.5)
    strip=sample(heat,q).T
    vis=cv2.applyColorMap(np.rint(np.clip(strip,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    vis=cv2.resize(vis,(1600,480),interpolation=cv2.INTER_NEAREST)
    cy=240;cv2.line(vis,(0,cy),(1599,cy),(245,245,245),1)
    corridor=.32*width*2
    for offset in [-corridor,corridor]:
        yy=int(round((offset+40)/80*479));cv2.line(vis,(0,yy),(1599,yy),(100,100,100),1)
    xx=int(round(828/2000*1599));cv2.line(vis,(xx,0),(xx,479),(255,255,255),2)
    save(HERE/(ID+"__B_normal_strip.png"),panel(vis,"B法向诊断：横轴沿旧轴0–2000原px；纵轴法向-40–40原px","白竖线：旧自由末端828px；横线：旧轴及±0.32宽搜索范围；色标固定0–1"))
    c3=entries[1]["connection"]["candidates"][0]
    diagnostic=dict(id=ID,command=sys.argv,code_sha256=sha(__file__),original_image=str(original),original_sha256=job["image_sha256"],
        response_file=str(rp),response_sha256=before,H2_record_sha256=sha(H2/"all_001/记录"/(ID+".json.gz")),H3_record_sha256=sha(H3/"all_001/记录"/(ID+".json.gz")),
        original_coordinate_mapping="original=2*crop_half+[0,1050]",a7_status=a7["status"],chosen_h2_a7_candidate_id=candidate["candidate_id"],
        chosen_candidate_accepted=candidate["accepted"],chosen_candidate_plausible=candidate["plausible"],h2_free_point_count=len(f2["points_xy"]),h3_free_point_count=len(f3["points_xy"]),
        h2_free_end_original_xy=(np.asarray(f2["points_xy"][-1])*2+[0,1050]).tolist(),h3_free_end_original_xy=(np.asarray(f3["points_xy"][-1])*2+[0,1050]).tolist(),
        H3_axis_source=f3["straightening"]["axis_source"],H3_angle_from_seed=f3["straightening"]["axis_angle_from_seed_degrees"],
        H3_join_start_original_xy=(np.asarray(c3["join_start_xy"])*2+[0,1050]).tolist(),H3_join_end_original_xy=(np.asarray(c3["join_end_xy"])*2+[0,1050]).tolist(),
        H3_bridge_starts_before_free_end_original_px=float(2*np.linalg.norm(np.asarray(c3["join_start_xy"])-np.asarray(f3["points_xy"][-1]))),
        H3_reported_abandoned_fraction=c3["metrics"]["abandoned_verified_free_fraction"],profile_windows=windows,
        scan_progress_original_px=(t*2).tolist(),scans=scans,diagnostic_crop_original_xyxy=box,
        diagnosis=["H2 scans a fixed plausible but unaccepted A7 axis with only plus/minus0.32 optical widths local correction.",
                   "H3 uses a TLS fallback because no accepted A7 match exists; it retains the inherited139-point extent and cannot extend it.",
                   "The H3 winning bridge starts before that extent ends; its distance-to-observed-prefix tolerance is not proof of target continuation.",
                   "Wider normal candidates and strong B strips are diagnostic evidence only; same-rope identity and final endpoint remain unverified."],
        first_diagnostic_attempt_failure="OpenCV remap32767-column limit from flattened display sampling grid; corrected by preserving2D grid, no core/response change",
        labels_read=False,reserved_data_accessed=False,network_inference=False,independent_identity_reference=False)
    dump(HERE/(ID+"__diagnosis.json"),diagnostic)
    assert sha(original)==job["image_sha256"] and sha(rp)==before
    print(json.dumps(dict(exit_code=0,windows=windows,H3_early_bridge_original_px=diagnostic["H3_bridge_starts_before_free_end_original_px"]),ensure_ascii=True))
if __name__=="__main__":main()
