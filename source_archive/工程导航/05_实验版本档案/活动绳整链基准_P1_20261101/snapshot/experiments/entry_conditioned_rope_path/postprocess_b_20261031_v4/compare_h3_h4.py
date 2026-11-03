"""Saved H3/H4 same-condition free-ridge comparison; no labels or inference.

One predeclared diagnostic gate is applied to both versions' saved numerical
profiles. Heat morphology, original-image corroboration and identity stay separate.
"""
from pathlib import Path
from collections import Counter
import argparse,gzip,hashlib,importlib.util,json,sys,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent;H3=HERE.parent/"postprocess_b_20261031_v3"
EXAMPLE="R_static_20260922_01__f000095__right"
SETTINGS=dict(heat_center_minimum=.28,side_drop_minimum=.045,side_asymmetry_maximum=.45,
              raw_score_minimum=.25,raw_contrast_minimum=.008,path_retention_tolerance_widths=.40,
              common_free_minimum_original_px=120.,profile_sampling_original_px=6.,
              purpose="same saved profile diagnostic gates, not an identity or accuracy metric")
def load(p):
    with (gzip.open(p,"rt",encoding="utf-8") if str(p).endswith(".gz") else open(p,encoding="utf-8-sig")) as f:return json.load(f)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def main_candidate(e):
    cs=e["connection"]["candidates"];ix=e["connection"].get("selected_index")
    return cs[ix if ix is not None else 0] if cs else None
def flags(e):
    heat=(np.asarray(e["valid"],bool)&(np.asarray(e["center"])>=.28)&(np.asarray(e["side_drop"])>=.045)&(np.asarray(e["side_distance_asymmetry_widths"])<=.45))
    joint=heat&(np.asarray(e["raw_score"])>=.25)&(np.asarray(e["raw_contrast"])>=.008)
    return heat,joint
def poly_dist(p,q):
    if len(q)<2:return np.full(len(p),1.e9)
    a=q[:-1];d=np.diff(q,axis=0);den=np.maximum((d*d).sum(1),1.e-12);out=[]
    for chunk in np.array_split(p,max(1,int(np.ceil(len(p)/128)))):
        if not len(chunk):continue
        t=np.clip(((chunk[:,None]-a[None])*d[None]).sum(2)/den[None],0,1)
        dist=np.linalg.norm(chunk[:,None]-a[None]-t[:,:,None]*d[None],axis=2)
        out.extend(dist.min(1).tolist())
    return np.asarray(out)
def weighted(v,seg):
    if not len(seg) or seg.sum()==0:return None
    v=np.asarray(v,float);return float(np.sum((v[:-1]+v[1:])*.5*seg)/seg.sum())

def measure(e,c):
    f=e["free"];p=np.asarray(f["points_xy"],float).reshape(-1,2)
    out=dict(free_available=len(p)>=2,free_length_original_px=0.,heat_supported_length_original_px=0.,
             joint_supported_length_original_px=0.,heat_only_length_original_px=0.,unknown_length_original_px=0.,
             selected_path_length_original_px=0.,selected_joint_length_original_px=0.,only_free=False,
             kind=c["kind"] if c else "no_candidate",heat_supported_abandoned_length_original_px=0.)
    if c:
        q=np.asarray(c["points_xy"]);ds=2*np.linalg.norm(np.diff(q,axis=0),axis=1);s=np.asarray(c["supported"],bool)
        out.update(selected_path_length_original_px=float(ds.sum()),selected_joint_length_original_px=float(ds[s[:-1]&s[1:]].sum()),
                   only_free=all(x=="free" for x in c["stage"]),termination=c["termination"])
    if len(p)<2:return out
    ev=f["evidence"];heat,joint=flags(ev);seg=2*np.linalg.norm(np.diff(p,axis=0),axis=1)
    # The input entry condition is measured for consistency, but excluded from
    # newly supported-length claims here.
    states=np.asarray(f["point_status"]);post=states!="entry_condition"
    hm=heat[:-1]&heat[1:]&post[:-1]&post[1:];jm=joint[:-1]&joint[1:]&post[:-1]&post[1:]
    out.update(free_length_original_px=float(seg.sum()),heat_supported_length_original_px=float(seg[hm].sum()),
               joint_supported_length_original_px=float(seg[jm].sum()),heat_only_length_original_px=float(seg[hm&~jm].sum()),
               unknown_length_original_px=float(seg[post[:-1]&post[1:]&~hm].sum()),
               heat_center_mean=weighted(ev["center"],seg),left_low_mean=weighted(ev["left_low"],seg),
               right_low_mean=weighted(ev["right_low"],seg),side_drop_mean=weighted(ev["side_drop"],seg),
               raw_score_mean=weighted(ev["raw_score"],seg),raw_contrast_mean=weighted(ev["raw_contrast"],seg),
               width_original_px=2*float(f["width_half_px"]),free_end_original_xy=(p[-1]*2+[0,1050]).tolist())
    if c:
        q=np.asarray(c["points_xy"]);dist=poly_dist(p,q);tol=.4*float(f["width_half_px"])
        lost=(dist[:-1]>tol)|(dist[1:]>tol)
        out["heat_supported_abandoned_length_original_px"]=float(seg[hm&lost].sum())
        out["heat_supported_retained_fraction"]=float(seg[hm&~lost].sum()/max(seg[hm].sum(),1.e-9))
        free_indices=np.flatnonzero(np.asarray(c["stage"])=="free")
        heat_indices=np.flatnonzero(heat&post)
        if len(free_indices) and len(heat_indices):
            d=(p[-1]-p[0]);d/=max(np.linalg.norm(d),1.e-9)
            lead=float(2*((p[heat_indices[-1]]-q[free_indices[-1]])@d))
            out["join_lead_before_last_heat_support_original_px"]=max(0.,lead) if not out["only_free"] else 0.
            out["early_join_is_proxy_not_verified_jump_error"]=True
        out["recorded_abandoned_fraction"]=c.get("metrics",{}).get("abandoned_verified_free_fraction")
    return out

def common_profiles(e3,e4):
    fs=[e3["free"],e4["free"]]
    if any(len(f["points_xy"])<2 for f in fs):return dict(eligible=False,reason="one_free_unavailable")
    seed=np.asarray(e3["seed"]["xy"]);u=seed[-1]-seed[0];u/=np.linalg.norm(u);origin=(seed[0]-[0,1050])*.5
    ps=[np.asarray(f["points_xy"]) for f in fs];ts=[(p-origin)@u for p in ps]
    if any(np.any(np.diff(t)<=0) for t in ts):return dict(eligible=False,reason="nonmonotonic_free_progress")
    lo=max(60.,ts[0][0],ts[1][0]);hi=min(ts[0][-1],ts[1][-1])
    if 2*(hi-lo)<120:return dict(eligible=False,reason="common_free_post_seed_under120px")
    grid=np.linspace(lo,hi,max(3,int((hi-lo)/3)+1));out=[]
    for f,t in zip(fs,ts):
        ev=f["evidence"];v={k:np.interp(grid,t,np.asarray(ev[k],float)) for k in ["center","left_low","right_low","side_drop","side_distance_asymmetry_widths","raw_score","raw_contrast","valid"]}
        heat,joint=flags(v)
        out.append(dict(center_mean=float(v["center"].mean()),side_drop_mean=float(v["side_drop"].mean()),
                        left_low_mean=float(v["left_low"].mean()),right_low_mean=float(v["right_low"].mean()),
                        heat_profile_fraction=float(heat.mean()),joint_profile_fraction=float(joint.mean())))
    return dict(eligible=True,common_post_entry_original_px=2*(hi-lo),H3=out[0],H4=out[1],
                delta_center=out[1]["center_mean"]-out[0]["center_mean"],delta_side_drop=out[1]["side_drop_mean"]-out[0]["side_drop_mean"])

def aggregate(rows):
    out=dict(pairs=len(rows))
    for v in ("H3","H4"):
        ms=[r[v] for r in rows]
        keys=["free_length_original_px","heat_supported_length_original_px","joint_supported_length_original_px",
              "heat_only_length_original_px","unknown_length_original_px","selected_path_length_original_px",
              "selected_joint_length_original_px","heat_supported_abandoned_length_original_px"]
        out[v]=dict(kind_counts=dict(Counter(m["kind"] for m in ms)),only_free_count=sum(m["only_free"] for m in ms),
                    free_available_count=sum(m["free_available"] for m in ms),
                    sums={k:float(sum(m[k] for m in ms)) for k in keys},
                    medians={k:float(np.median([m[k] for m in ms])) for k in keys},
                    early_join_proxy_count=sum(m.get("join_lead_before_last_heat_support_original_px",0)>12 for m in ms),
                    early_join_proxy_note="stage ends over12 original px before last diagnostic heat support; not true identity error")
    ok=[r["common_profiles"] for r in rows if r["common_profiles"]["eligible"]]
    out["common_profiles"]=dict(eligible=len(ok),ineligible_reasons=dict(Counter(r["common_profiles"]["reason"] for r in rows if not r["common_profiles"]["eligible"])),
                                 median_delta_center=float(np.median([r["delta_center"] for r in ok])) if ok else None,
                                 median_delta_side_drop=float(np.median([r["delta_side_drop"] for r in ok])) if ok else None)
    out["selection_transitions"]=dict(Counter(("free" if r["H3"]["only_free"] else "connected" if r["H3"]["kind"]!="no_candidate" else "none")+" -> "+("free" if r["H4"]["only_free"] else "connected" if r["H4"]["kind"]!="no_candidate" else "none") for r in rows))
    out["H4_free_longer_count"]=sum(r["H4"]["free_length_original_px"]>r["H3"]["free_length_original_px"]+1 for r in rows)
    out["H4_free_shorter_count"]=sum(r["H4"]["free_length_original_px"]<r["H3"]["free_length_original_px"]-1 for r in rows)
    return out

def visual(oldrun,newrun,identity,dest):
    paths=[oldrun/"四栏检测结果"/(identity+".jpg"),newrun/"四栏检测结果"/(identity+".jpg")]
    with Image.open(paths[0]) as im3,Image.open(paths[1]) as im4:
        assert im3.size==im4.size==(8192,1655)
        crops=[im4.crop((0,680,2048,1655)).convert("RGB"),im3.crop((6144,680,8192,1655)).convert("RGB"),im4.crop((6144,680,8192,1655)).convert("RGB")]
    board=Image.new("RGB",(6144,1065),(246,248,251));d=ImageDraw.Draw(board)
    font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",30);small=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",23)
    for k,(title,im) in enumerate(zip(["原图（固定搜索ROI）","H3：旧直轴/长度继承","H4：B热图重估自由直段"],crops)):
        x=k*2048;d.text((x+20,8),title,font=font,fill=(20,30,40));d.text((x+20,50),identity,font=small,fill=(50,65,80));board.paste(im,(x,90))
    board.save(dest,quality=93,subsampling=0)
    return dict(id=identity,paths=list(map(str,paths)),source_sha256=[sha(p) for p in paths],output=str(dest),output_sha256=sha(dest))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--h3-run",default="all_001");ap.add_argument("--h4-run",default="pilot_001");ap.add_argument("--out",required=True);a=ap.parse_args()
    oldrun=H3/a.h3_run;newrun=HERE/a.h4_run;out=HERE/a.out;out.mkdir(exist_ok=False)
    cfg=load(HERE/"inputs.json");summary=load(newrun/"summary.json");ids=[r["id"] for r in summary["rows"]];allowed={j["id"]:j for j in cfg["jobs"]}
    assert len(ids)==len(set(ids)) and set(ids)<=set(allowed)
    entries=[];images=[];sources=[];start=time.monotonic()
    with (out/"paired_entries.jsonl").open("x",encoding="utf-8") as log:
        for identity in ids:
            paths=[r/"记录"/(identity+".json.gz") for r in (oldrun,newrun)];r3,r4=[load(p) for p in paths]
            assert r3["source_image_sha256"]==r4["source_image_sha256"]==allowed[identity]["image_sha256"]
            assert len(r3["entries"])==len(r4["entries"])
            for k,(e3,e4) in enumerate(zip(r3["entries"],r4["entries"])):
                assert e3["seed"]==e4["seed"] and e3["response_sha256"]==e4["response_sha256"]
                row=dict(id=identity,entry_index=k,H3=measure(e3,main_candidate(e3)),H4=measure(e4,main_candidate(e4)),common_profiles=common_profiles(e3,e4))
                entries.append(row);log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
            ei3,ei4=r3["selected_entry"],r4["selected_entry"];e3,e4=r3["entries"][ei3],r4["entries"][ei4]
            c3=e3["connection"]["candidates"][r3["selected_candidate"]] if r3["selected_candidate"] is not None else None
            c4=e4["connection"]["candidates"][r4["selected_candidate"]] if r4["selected_candidate"] is not None else None
            common=common_profiles(e3,e4) if e3["seed"]==e4["seed"] else dict(eligible=False,reason="selected_entry_changed")
            images.append(dict(id=identity,selected_entry_H3=ei3,selected_entry_H4=ei4,H3=measure(e3,c3),H4=measure(e4,c4),common_profiles=common))
            sources.append(dict(id=identity,H3_sha256=sha(paths[0]),H4_sha256=sha(paths[1])))
    vd=out/"固定13图_原图_H3_H4";vd.mkdir();visuals=[]
    for identity in cfg["pilot_ids"]:
        if identity in ids:visuals.append(visual(oldrun,newrun,identity,vd/(identity+".jpg")))
    report=dict(command=sys.argv,comparison_source_sha256=sha(__file__),settings=SETTINGS,H3_run=str(oldrun),H4_run=str(newrun),
                H3_summary_sha256=sha(oldrun/"summary.json"),H4_summary_sha256=sha(newrun/"summary.json"),by_image=aggregate(images),by_entry=aggregate(entries),
                selected_entry_changed=sum(r["selected_entry_H3"]!=r["selected_entry_H4"] for r in images),
                specified_counterexample=next((r for r in images if r["id"]==EXAMPLE),None),
                fixed_visual_count=len(visuals),seconds=time.monotonic()-start,labels_read=False,new_inference=False,
                claims="More/larger heat-supported straight output is not same-rope identity accuracy. Connected-to-free is retained as a change, not scored automatically as success.")
    save(out/"summary.json",report);save(out/"paired_images.json",images);save(out/"sources.json",sources);save(out/"visuals.json",visuals)
    print(json.dumps(report,ensure_ascii=True),flush=True)
if __name__=="__main__":main()
