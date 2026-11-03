"""Paired geometry/support comparison of saved H2/H3 results; no labels.

Direction is measured only over a shared downstream progress interval.
Length, support and missing comparisons remain explicit so truncation cannot
masquerade as a smoother complete path.
"""
from pathlib import Path
from collections import Counter
import argparse, gzip, hashlib, json, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
H2=HERE.parent/"postprocess_b_20261030_v2"
SETTINGS=dict(coordinate_scale=.5, minimum_common_progress_half_px=60.,
              progress_sample_step_half_px=4., direction_chord_half_px=24.,
              common_start="maximum of the two last-free progress positions",
              progress_axis="same unchanged entry seed direction",
              length_support_rule="segment counted supported only when both endpoints supported",
              same_selected_entry_required_for_image_direction_comparison=True)

def load(p):
    f=gzip.open(p,"rt",encoding="utf-8") if str(p).endswith(".gz") else open(p,encoding="utf-8-sig")
    with f:return json.load(f)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

def direction(points, parameter=None):
    p=np.asarray(points,float)
    if len(p)<3:return None
    if parameter is None:parameter=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    parameter=np.asarray(parameter,float)
    keep=np.r_[True,np.diff(parameter)>1e-8];p=p[keep];parameter=parameter[keep]
    if len(p)<3 or parameter[-1]-parameter[0]<8:return None
    grid=np.linspace(parameter[0],parameter[-1],max(3,int((parameter[-1]-parameter[0])/4)+1))
    lo=np.maximum(parameter[0],grid-12);hi=np.minimum(parameter[-1],grid+12)
    a=np.c_[np.interp(lo,parameter,p[:,0]),np.interp(lo,parameter,p[:,1])]
    b=np.c_[np.interp(hi,parameter,p[:,0]),np.interp(hi,parameter,p[:,1])]
    delta=b-a;angles=np.unwrap(np.arctan2(delta[:,1],delta[:,0]))
    return dict(amplitude_degrees=float(np.degrees(np.ptp(angles))),
                total_variation_degrees=float(np.degrees(np.abs(np.diff(angles)).sum())),
                chord_half_px=24.,samples=len(grid))

def straightness(p):
    p=np.asarray(p,float)
    if len(p)<2:return None
    origin=p.mean(0);_,_,vt=np.linalg.svd(p-origin,full_matrices=False)
    normal=np.array([-vt[0,1],vt[0,0]]);r=np.abs((p-origin)@normal)*2
    d=np.diff(p,axis=0);d=d[np.linalg.norm(d,axis=1)>1e-8]
    d/=np.maximum(np.linalg.norm(d,axis=1)[:,None],1e-8)
    turns=np.degrees(np.arccos(np.clip((d[1:]*d[:-1]).sum(1),-1,1))) if len(d)>1 else np.zeros(0)
    return dict(points=len(p),TLS_orthogonal_rms_original_px=float(np.sqrt(np.mean(r*r))),
                TLS_orthogonal_p95_original_px=float(np.percentile(r,95)),TLS_orthogonal_max_original_px=float(r.max()),
                raw_vertex_turn_mean_degrees=float(turns.mean()) if len(turns) else 0.,
                raw_vertex_turn_max_degrees=float(turns.max()) if len(turns) else 0.,
                chord_direction=direction(p))

def main_candidate(entry):
    cs=entry["connection"]["candidates"];i=entry["connection"].get("selected_index",0)
    return cs[i if i is not None else 0] if cs else None

def path_metrics(candidate):
    if candidate is None:return dict(available=False,kind="no_candidate",length_original_px=0.,supported_length_original_px=0.,weak_length_original_px=0.,only_free=False,stage_lengths_original_px={},free_geometry=None)
    p=np.asarray(candidate["points_xy"],float);sup=np.asarray(candidate["supported"],bool);stage=np.asarray(candidate["stage"])
    seg=np.linalg.norm(np.diff(p,axis=0),axis=1)*2;ss=sup[:-1]&sup[1:]
    lengths={str(s):float(seg[stage[1:]==s].sum()) for s in sorted(set(stage))}
    return dict(available=True,kind=candidate["kind"],length_original_px=float(seg.sum()),
                supported_length_original_px=float(seg[ss].sum()),weak_length_original_px=float(seg[~ss].sum()),
                supported_length_fraction=float(seg[ss].sum()/max(seg.sum(),1e-9)),
                only_free=bool(np.all(stage=="free")),stage_lengths_original_px=lengths,
                free_geometry=straightness(p[stage=="free"]),termination=candidate["termination"],
                track_id=candidate.get("track_id"),identity_verified=False)

def common_direction(a,b,seed,same_condition=True):
    out=dict(eligible=False)
    if not same_condition:return out|dict(reason="selected_entry_changed")
    if a is None or b is None:return out|dict(reason="one_or_both_have_no_candidate")
    if all(s=="free" for s in a["stage"]) or all(s=="free" for s in b["stage"]):
        return out|dict(reason="one_or_both_only_free")
    xy=np.asarray(seed["xy"],float);u=xy[-1]-xy[0];u/=np.linalg.norm(u);origin=(xy[0]-[0,1050])*.5
    paths=[np.asarray(c["points_xy"],float) for c in (a,b)]
    progress=[(p-origin)@u for p in paths]
    if any(np.any(np.diff(t)<=1e-7) for t in progress):
        return out|dict(reason="nonmonotonic_progress_no_reordering_or_shortcut")
    exits=[]
    for c,t in zip((a,b),progress):
        ids=np.flatnonzero(np.asarray(c["stage"])=="free")
        if not len(ids):return out|dict(reason="no_free_stage")
        exits.append(float(t[ids[-1]]))
    lo=max(exits+[float(t[0]) for t in progress]);hi=min(float(t[-1]) for t in progress);span=hi-lo
    out.update(common_progress_interval_original_px=[2*lo,2*hi],common_progress_length_original_px=max(0.,2*span),
               free_exit_progress_original_px=[2*v for v in exits],
               downstream_available_progress_original_px=[2*max(0.,float(t[-1])-x) for t,x in zip(progress,exits)],
               total_progress_original_px=[2*float(t[-1]-t[0]) for t in progress])
    if span<SETTINGS["minimum_common_progress_half_px"]:return out|dict(reason="common_downstream_interval_under_120_original_px")
    grid=np.linspace(lo,hi,max(3,int(span/4)+1));measures=[];coverage=[];support=[]
    for c,p,t,x in zip((a,b),paths,progress,exits):
        q=np.c_[np.interp(grid,t,p[:,0]),np.interp(grid,t,p[:,1])]
        measures.append(direction(q,grid))
        coverage.append(float(span/max(float(t[-1])-x,1e-9)))
        segment_support=np.interp(grid,t,np.asarray(c["supported"],float))>.99999
        ds=np.linalg.norm(np.diff(q,axis=0),axis=1)*2
        support.append(float(ds[segment_support[:-1]&segment_support[1:]].sum()))
    out.update(eligible=True,reason="same_entry_common_downstream_interval",
               H2=measures[0],H3=measures[1],common_fraction_of_own_downstream=coverage,
               image_supported_length_in_common_original_px=support,
               delta_H3_minus_H2_amplitude_degrees=measures[1]["amplitude_degrees"]-measures[0]["amplitude_degrees"],
               delta_H3_minus_H2_variation_degrees=measures[1]["total_variation_degrees"]-measures[0]["total_variation_degrees"])
    return out

def source_constraint(entry):
    checks=[];mapping=entry.get("candidate_track_geometry",{})
    for i,c in enumerate(entry["connection"]["candidates"]):
        ids=np.flatnonzero(np.asarray(c["stage"])=="surface")
        if not len(ids):continue
        track=c.get("track_id");source=mapping.get(track);indices=c.get("surface_source_indices",[])
        valid=source is not None and len(indices)==len(ids)
        deviation=None
        if valid:
            src=np.asarray(source,float);idx=np.asarray(indices,int)
            valid=bool(np.all((idx>=0)&(idx<len(src))))
            if valid:
                actual=np.asarray(c["points_xy"])[ids];ref=src[idx]
                deviation=float(np.max(np.linalg.norm(actual-ref,axis=1)))
                valid=bool(np.array_equal(actual,ref))
        checks.append(dict(candidate=i,track_id=track,surface_points=len(ids),exact_match=bool(valid),maximum_deviation_original_px=2*deviation if deviation is not None else None))
    return checks

def aggregate(rows):
    out=dict(pairs=len(rows),H2={},H3={})
    for version in ("H2","H3"):
        metrics=[r[version] for r in rows]
        total=[m["length_original_px"] for m in metrics];supported=[m["supported_length_original_px"] for m in metrics]
        fg=[m["free_geometry"] for m in metrics if m.get("free_geometry") is not None]
        out[version]=dict(path_count=sum(m["available"] for m in metrics),only_free_count=sum(m["only_free"] for m in metrics),
                          kind_counts=dict(Counter(m["kind"] for m in metrics)),
                          length_sum_original_px=float(sum(total)),length_median_original_px=float(np.median(total)) if total else None,
                          supported_length_sum_original_px=float(sum(supported)),supported_length_median_original_px=float(np.median(supported)) if supported else None,
                          free_geometry_count=len(fg),free_TLS_rms_median_original_px=float(np.median([x["TLS_orthogonal_rms_original_px"] for x in fg])) if fg else None,
                          free_raw_turn_max_median_degrees=float(np.median([x["raw_vertex_turn_max_degrees"] for x in fg])) if fg else None)
    ok=[r for r in rows if r["common_direction"]["eligible"]]
    out["common_direction"]=dict(eligible_count=len(ok),not_eligible_reasons=dict(Counter(r["common_direction"]["reason"] for r in rows if not r["common_direction"]["eligible"])))
    if ok:
        out["common_direction"].update(
            median_delta_amplitude_degrees=float(np.median([r["common_direction"]["delta_H3_minus_H2_amplitude_degrees"] for r in ok])),
            median_delta_variation_degrees=float(np.median([r["common_direction"]["delta_H3_minus_H2_variation_degrees"] for r in ok])),
            both_direction_metrics_lower_count=sum(r["common_direction"]["delta_H3_minus_H2_amplitude_degrees"]<0 and r["common_direction"]["delta_H3_minus_H2_variation_degrees"]<0 for r in ok),
            H3_shorter_path_count=sum(r["H3"]["length_original_px"]<r["H2"]["length_original_px"] for r in ok),
            H3_lower_supported_length_count=sum(r["H3"]["supported_length_original_px"]<r["H2"]["supported_length_original_px"] for r in ok),
            median_common_progress_original_px=float(np.median([r["common_direction"]["common_progress_length_original_px"] for r in ok])),
            median_coverage_H2=float(np.median([r["common_direction"]["common_fraction_of_own_downstream"][0] for r in ok])),
            median_coverage_H3=float(np.median([r["common_direction"]["common_fraction_of_own_downstream"][1] for r in ok])))
    return out

def image_pair(oldrun,newrun,identity,dest):
    # Fixed search ROI in last panel: full original crop [0,1050,4096,3000]
    # shown at50%; no result-dependent crop and no contrast manipulation.
    paths=[r/"四栏检测结果"/(identity+".jpg") for r in (oldrun,newrun)]
    images=[]
    for p in paths:
        with Image.open(p) as im:
            if im.size!=(8192,1655):raise ValueError("Unexpected four-panel layout: "+str(p))
            images.append(im.crop((6144,680,8192,1655)).convert("RGB"))
    board=Image.new("RGB",(4096,1051),(245,247,250));draw=ImageDraw.Draw(board)
    f=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",28)
    draw.text((20,8),"H2 原后处理 | "+identity,font=f,fill=(20,30,40))
    draw.text((2068,8),"H3 直自由段＋既有中心轨迹约束",font=f,fill=(20,30,40))
    draw.text((20,43),"同一固定坐标窗口；橙虚线仍是弱证据/推断；没有独立身份真值",font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",23),fill=(70,70,70))
    board.paste(images[0],(0,76));board.paste(images[1],(2048,76));board.save(dest,quality=93,subsampling=0)
    return dict(id=identity,source_paths=list(map(str,paths)),source_sha256=[sha(p) for p in paths],output=str(dest),output_sha256=sha(dest),crop_in_four_panel_xyxy=[6144,680,8192,1655])

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--h2-run",default="all_001");ap.add_argument("--h3-run",default="pilot_001");ap.add_argument("--out",required=True);a=ap.parse_args()
    oldrun=H2/a.h2_run;newrun=HERE/a.h3_run;out=HERE/a.out;out.mkdir(exist_ok=False)
    cfg=load(HERE/"inputs.json");sm2=load(oldrun/"summary.json");sm3=load(newrun/"summary.json")
    ids=[r["id"] for r in sm3["rows"]];assert len(ids)==len(set(ids)) and set(ids)<={r["id"] for r in sm2["rows"]}
    allowed={j["id"]:j for j in cfg["jobs"]};assert set(ids)<=set(allowed)
    entry_rows=[];image_rows=[];source_checks=[];sources=[];start=time.monotonic()
    with (out/"paired_entries.jsonl").open("x",encoding="utf-8") as log:
        for identity in ids:
            p2=oldrun/"记录"/(identity+".json.gz");p3=newrun/"记录"/(identity+".json.gz")
            r2,r3=load(p2),load(p3);assert r2["source_image_sha256"]==r3["source_image_sha256"]==allowed[identity]["image_sha256"]
            assert len(r2["entries"])==len(r3["entries"])
            for i,(e2,e3) in enumerate(zip(r2["entries"],r3["entries"])):
                assert e2["seed"]==e3["seed"] and e2["response_sha256"]==e3["response_sha256"]
                c2,c3=main_candidate(e2),main_candidate(e3)
                checks=source_constraint(e3);source_checks.extend([dict(id=identity,entry_index=i,**x) for x in checks])
                row=dict(id=identity,entry_index=i,seed_id=e2["seed"]["entry_id"],H2=path_metrics(c2),H3=path_metrics(c3),
                         common_direction=common_direction(c2,c3,e2["seed"]),surface_source_constraint=checks,
                         H3_straightening=e3["free"].get("straightening",{}))
                entry_rows.append(row);log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
            ei2,ei3=r2["selected_entry"],r3["selected_entry"];ci2,ci3=r2["selected_candidate"],r3["selected_candidate"]
            e2,e3=r2["entries"][ei2],r3["entries"][ei3]
            c2=e2["connection"]["candidates"][ci2] if ci2 is not None else None
            c3=e3["connection"]["candidates"][ci3] if ci3 is not None else None
            image_rows.append(dict(id=identity,selected_entry_H2=ei2,selected_entry_H3=ei3,H2=path_metrics(c2),H3=path_metrics(c3),
                                   common_direction=common_direction(c2,c3,e2["seed"],same_condition=e2["seed"]==e3["seed"])))
            sources.append(dict(id=identity,H2_record_sha256=sha(p2),H3_record_sha256=sha(p3)))
    visuals=[];vd=out/"固定12图末栏ROI";vd.mkdir()
    for identity in cfg["pilot_ids"]:
        if identity in ids:visuals.append(image_pair(oldrun,newrun,identity,vd/(identity+".jpg")))
    summary=dict(command=sys.argv,source_sha256=sha(__file__),settings=SETTINGS,H2_run=str(oldrun),H3_run=str(newrun),
                 H2_summary_sha256=sha(oldrun/"summary.json"),H3_summary_sha256=sha(newrun/"summary.json"),
                 by_image=aggregate(image_rows),by_entry=aggregate(entry_rows),
                 selected_entry_changed_count=sum(r["selected_entry_H2"]!=r["selected_entry_H3"] for r in image_rows),
                 surface_constraint=dict(candidates_with_surface=len(source_checks),exact_matches=sum(x["exact_match"] for x in source_checks),
                                         failures=[x for x in source_checks if not x["exact_match"]]),
                 fixed_visual_count=len(visuals),seconds=time.monotonic()-start,
                 identity_accuracy_measured=False,labels_read=False,
                 interpretation="Geometric regularity and heuristic image support only; all length/coverage losses are retained. A lower direction change does not verify same-rope identity or endpoint correctness.")
    save(out/"summary.json",summary);save(out/"paired_images.json",image_rows);save(out/"source_records.json",sources);save(out/"visuals.json",visuals)
    print(json.dumps(summary,ensure_ascii=True),flush=True)

if __name__=="__main__":main()
