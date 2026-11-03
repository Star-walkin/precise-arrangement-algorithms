"""Independent H2 artifact audit. No detector/model/label/device imports.

Passing means numerical and artifact consistency, never verified rope identity.
Failed checks are retained; no source or prior result is modified.
"""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
import argparse, ast, csv, gzip, hashlib, json, math, sys, time, traceback
import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
H1 = BASE / "postprocess_b_20261030_v1"
ASSETS = ROOT / "数据资产/05_派生与实验结果"
DATA = ASSETS / "camera_condition_images_20261009_v2"
MODEL_SHA = "ed4bfa12c5b3081a19d056265b053d516bdba5ec05dce658e1e32339b47fb882"

def number(s):
    value = float(s)
    if not math.isfinite(value): raise ValueError("nonfinite JSON number: " + s)
    return value

def bad_number(s): raise ValueError("nonfinite JSON constant: " + s)

def load(p):
    opener = gzip.open if Path(p).suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8-sig") as f:
        return json.load(f, parse_float=number, parse_constant=bad_number)

def norm(p): return str(Path(p).resolve())

@lru_cache(maxsize=12000)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576), b""): h.update(b)
    return h.hexdigest()

class Checks:
    def __init__(self): self.failures=[]; self.count=0
    def test(self, ok, code, detail=""):
        self.count += 1
        if not bool(ok): self.failures.append(dict(code=code, detail=str(detail)))
        return bool(ok)
    def hash(self, p, expected, code):
        try: return self.test(sha(norm(p))==expected, code, p)
        except Exception as e: return self.test(False, code, str(p)+": "+str(e))

def points(c, values, tag, empty=False):
    p=np.asarray(values,float)
    if not p.size and empty: return np.empty((0,2))
    if not c.test(p.ndim==2 and p.shape[1]==2 and len(p)>=2,"path_shape",tag): return np.empty((0,2))
    c.test(np.isfinite(p).all(),"path_nonfinite",tag)
    c.test(np.all((p[:,0]>=0)&(p[:,0]<2048)&(p[:,1]>=0)&(p[:,1]<975)),"path_crop_bounds",tag)
    c.test(np.all(np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-7),"zero_length_segment",tag)
    return p

def aligned(c, obj, keys, n, tag):
    for key in keys: c.test(key in obj and len(obj[key])==n,"array_length_alignment",f"{tag}.{key}: {n}")

def seed_raster(xy):
    q=(np.asarray(xy,float)-[0,1050])*.5
    mask=np.zeros((975,2048),np.uint8)
    cv2.polylines(mask,[np.rint(q).astype(np.int32)],False,1,1,cv2.LINE_8)
    d=cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return np.exp(-.5*(d/2.)**2).astype(np.float32)

def dash_contract(source):
    tree=ast.parse(Path(source).read_text(encoding="utf-8-sig"))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="draw_final")
    env=dict(np=np,cv2=cv2)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),"exec"),env)
    p=np.c_[np.arange(100,421,8),np.full(41,100.)]
    candidate=dict(points_xy=p.tolist(),supported=[False]*len(p),stage=["surface"]*len(p))
    background=np.zeros((1500,2048,3),np.uint8)
    weak=env["draw_final"](background,candidate,True)
    candidate["supported"]=[True]*len(p)
    strong=env["draw_final"](background,candidate,False)
    wo=np.any(weak[625,120:395]!=0,axis=1);so=np.any(strong[625,120:395]!=0,axis=1)
    return dict(weak_visible_pixels=int(wo.sum()),weak_gap_pixels=int((~wo).sum()),
                supported_continuous=bool(so.all()),passed=bool(wo.any() and (~wo).sum()>=20 and so.all()),
                scope="frozen_final_renderer_behavior_not_identity_verification")

def audit_job(job,row,run,srow,new_receipt):
    c=Checks(); identity=job["id"]
    result=dict(id=identity,entries=[],candidates=0,unsupported_points=0)
    try:
        original=(DATA/row["output_file"]).resolve()
        c.test(norm(job["image_path"])==str(original) and job["image_sha256"]==row["output_sha256"],"source_manifest_binding",identity)
        c.hash(original,row["output_sha256"],"source_bytes_hash")
        c.test(job["shape"]==[3000,4096] and job["bbox"]==[0,1050,4096,3000],"source_crop_mapping",identity)
        c.test(job["source_group"]==row["source_group"],"source_group",identity)
        rp=run/"记录"/(identity+".json.gz");r=load(rp)
        c.test(r["id"]==identity and norm(r["source_image"])==str(original) and r["source_image_sha256"]==row["output_sha256"],"record_source_binding",identity)
        c.test(r["coordinate_system"]=="crop_half_xy; original=2*xy+[0,1050]","record_coordinates",identity)
        c.test(not r["labels_read"] and not r["device_connected"] and not r["complete_identity_verified"] and not r["response_entry_conditions_merged"],"record_scope_flags",identity)
        c.test(len(r["entries"])==len(job["seeds"])==len(job["response_files"]),"entry_count_alignment",identity)
        c.test([e["entry_index"] for e in r["entries"]]==list(range(len(job["seeds"]))),"entry_indices",identity)
        for path,expected in r["input_dependency_sha256"].items():
            pp=Path(path).resolve()
            if c.test(pp.is_relative_to(ASSETS.resolve()) and "20260925" not in str(pp),"dependency_allowed",pp):
                c.hash(pp,expected,"dependency_hash")
        old=load(H1/"responses"/identity/"inference.json")
        c.test(old["model_sha256"]==MODEL_SHA,"old_model_binding",identity)
        c.test(old["image_sha256"]==row["output_sha256"] and old["bbox"]==job["bbox"],"old_image_mapping",identity)
        cache=H1/"input_cache"/(identity+".npz")
        c.hash(cache,old["input_cache_sha256"],"old_input_cache_hash")
        with np.load(cache,allow_pickle=False) as z:
            cached=z["seeds"]
            c.test(z["image"].shape==(975,2048) and cached.shape==(len(job["original_seeds"]),975,2048),"old_cache_shapes",identity)
        all_candidates=[]; seen=set()
        for k,(e,seed,response) in enumerate(zip(r["entries"],job["seeds"],job["response_files"])):
            tag=f"{identity}/E{k+1}"
            c.test(e["seed"]==seed,"record_seed_binding",tag)
            xy=np.asarray(seed["xy"],float); q=(xy-[0,1050])*.5
            c.test(xy.ndim==2 and xy.shape[1]==2 and len(xy)>=2 and np.isfinite(xy).all(),"seed_shape",tag)
            c.test(np.all((q[:,0]>=0)&(q[:,0]<2048)&(q[:,1]>=0)&(q[:,1]<975)),"seed_crop_bounds",tag)
            c.test(np.allclose(2*q+[0,1050],xy,atol=1e-9,rtol=0),"seed_roundtrip",tag)
            c.test(0<np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()<=120.00001,"seed_length",tag)
            key=tuple(xy.ravel());c.test(key not in seen,"duplicate_seed_condition",tag);seen.add(key)
            c.test(Path(seed["source"]).stem==identity and Path(seed["source"]).resolve().is_relative_to(ASSETS.resolve()),"seed_source_binding",tag)
            c.test(norm(e["response_file"])==norm(response),"response_path_binding",tag)
            oi=seed.get("old_entry_index")
            parent=(H1/"responses"/identity) if oi is not None else (HERE/"responses"/identity)
            if not c.test(Path(response).resolve().parent==parent.resolve(),"response_allowed_path",tag):continue
            c.hash(response,e["response_sha256"],"response_hash")
            with np.load(response,allow_pickle=False) as z:
                heat=z["response"]
                c.test(heat.shape==(975,2048) and heat.dtype==np.float32,"response_shape_dtype",tag)
                c.test(np.isfinite(heat).all() and heat.min()>=0 and heat.max()<=1,"response_range",tag)
                heat_range=[float(heat.min()),float(heat.max())]
            if oi is not None:
                m=old["entry_conditions"][oi]
                c.test(m["entry_index"]==oi and m["seed"]["xy"]==seed["xy"],"old_seed_condition",tag)
                c.test(norm(m["response_path"])==norm(response) and m["response_sha256"]==e["response_sha256"],"old_response_receipt",tag)
                c.test(np.array_equal(cached[oi],seed_raster(seed["xy"])),"old_seed_raster_mapping",tag)
            else:
                matches=[x for x in new_receipt["rows"] if norm(x["response_file"])==norm(response)]
                c.test(len(matches)==1,"new_receipt_count",tag)
                if matches:c.test(matches[0]["seed"]==seed and matches[0]["response_sha256"]==e["response_sha256"],"new_response_binding",tag)
            f=e["free"];c.test(f["actual_seed_xy"]==seed["xy"],"free_seed_binding",tag)
            fp=points(c,f["points_xy"],tag+"/free",True)
            aligned(c,f,("supported","point_status"),len(fp),tag+"/free")
            c.test(len(f["segment_status"])==max(0,len(fp)-1),"free_segment_alignment",tag)
            for i,status in enumerate(f["point_status"]):
                if status=="ambiguous_gap":c.test(not f["supported"][i],"unknown_free_supported",tag)
            for gap in f["gap_hypotheses"]:c.test(not gap["supported"] and not gap["filled_as_measurement"],"gap_promoted_to_observation",tag)
            for i,part in enumerate(f["disconnected_pieces"]):
                pp=points(c,part["points_xy"],f"{tag}/disconnected{i}")
                aligned(c,part,("supported","a7_raw_supported"),len(pp),tag)
                c.test(not part["reachable_from_entry"] and not part["gap_filled"] and not part["entry_identity_confirmed"],"disconnected_identity_claim",tag)
            connection=e["connection"];candidates=connection["candidates"]
            c.test(len(candidates)<=3,"candidate_limit",tag)
            c.test(connection.get("selected_index")== (0 if candidates else None),"entry_selected_index",tag)
            scores=[x["rank_score"] for x in candidates];c.test(scores==sorted(scores,reverse=True),"candidate_order",tag)
            for ci,ca in enumerate(candidates):
                name=f"{tag}/candidate{ci}";p=points(c,ca["points_xy"],name)
                aligned(c,ca,("width_half_px","supported","stage","point_status"),len(p),name)
                widths=np.asarray(ca["width_half_px"],float)
                c.test(np.isfinite(widths).all() and np.all(widths>0),"candidate_width",name)
                c.test(not ca["identity_verified"],"identity_claim",name)
                for i,status in enumerate(ca["point_status"]):
                    if status in ("inferred_or_weak","conflicting_entry","ambiguous_gap"):
                        c.test(not ca["supported"][i],"unknown_candidate_solid_flag",name)
                result["unsupported_points"]+=int((~np.asarray(ca["supported"],bool)).sum())
                result["candidates"]+=1;all_candidates.append((ca["rank_score"],k,ci,ca))
            result["entries"].append(dict(entry_index=k,seed_id=seed["entry_id"],response_sha256=e["response_sha256"],response_range=heat_range,candidate_count=len(candidates),seed_crop_xy=q.tolist()))
        selected=r["selected_entry"]
        c.test(isinstance(selected,int) and 0<=selected<len(r["entries"]),"selected_entry_index",identity)
        best=sorted(all_candidates,key=lambda v:-v[0])[0] if all_candidates else None
        c.test(r["selected_candidate"]==(best[2] if best else None),"selected_candidate_index",identity)
        c.test(selected==(best[1] if best else 0),"selected_entry_best",identity)
        c.test(len(r["displayed_candidates"])<=3,"display_candidate_limit",identity)
        for item in r["displayed_candidates"]:
            ei,ci=item["entry"],item["candidate"]
            c.test(0<=ei<len(r["entries"]) and 0<=ci<len(r["entries"][ei]["connection"]["candidates"]),"display_candidate_index",identity)
        c.test(srow["entry_count"]==len(r["entries"]) and srow["selected_entry"]==selected,"summary_selection",identity)
        if best:
            length=float(2*np.linalg.norm(np.diff(best[3]["points_xy"],axis=0),axis=1).sum())
            c.test(abs(length-srow["length_original_px"])<1e-6 and length>0,"summary_path_length",identity)
            c.test(srow["kind"]==best[3]["kind"] and abs(srow["score"]-best[0])<1e-9,"summary_best_binding",identity)
        else:c.test(not srow["has_path"] and srow["length_original_px"]==0,"summary_no_path",identity)
        surface=r["surface_tracks"];tracks=surface["tracks"]
        c.test(len({t["track_id"] for t in tracks})==len(tracks),"surface_id_duplicate",identity)
        c.test(surface["diagnostics"]["extrapolated_points"]==0 and surface["diagnostics"]["fixed_spacing_lines_generated"]==0,"surface_fill",identity)
        for t in tracks:
            name=identity+"/"+t["track_id"];p=points(c,t["points_xy"],name)
            aligned(c,t,("midpoint_xy","upper_boundary_xy","lower_boundary_xy","spacing_px","refine_offset_px","f3_pair_supported","f3_pair_observed","f3_pair_inferred","supported","raw_score","heat_score","heat_level"),len(p),name)
            mid=np.asarray(t["midpoint_xy"]);width=np.asarray(t["spacing_px"])
            c.test(np.all(np.linalg.norm(p-mid,axis=1)<=.15001*width),"surface_refine_bound",name)
            c.test(len(set(t["boundary_ids"]))==2 and np.all(np.diff(mid[:,0])==3),"surface_adjacent_columns",name)
            c.test(np.all(~np.asarray(t["f3_pair_inferred"],bool)|~np.asarray(t["supported"],bool)),"inferred_boundary_supported",name)
        jpg=run/"四栏检测结果"/(identity+".jpg")
        c.test(norm(srow["output"])==norm(jpg),"gallery_path_binding",identity);c.hash(jpg,srow["sha256"],"gallery_hash")
        with Image.open(jpg) as im:
            c.test(im.size==(8192,1655) and im.mode=="RGB","gallery_size_channels",identity);im.load()
        result["record_sha256"]=sha(norm(rp));result["gallery_sha256"]=sha(norm(jpg))
    except Exception as e:c.test(False,"audit_exception",f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    result.update(passed=not c.failures,check_count=c.count,failures=c.failures)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",default="all_001");ap.add_argument("--out",default="audit_001");ap.add_argument("--workers",type=int,default=2);a=ap.parse_args()
    run=(HERE/a.run).resolve()
    if run.parent!=HERE.resolve() or a.run!="all_001":raise ValueError("This audit is for frozen all_001 only")
    out=run/a.out
    if out.parent!=run:raise ValueError("Output must be inside all_001")
    # Replay historical prepare_inference.prepare's two-thread precise EDT.
    # OpenCV's one-thread branch differs by about 1e-8, so it is unsuitable
    # for a bit-exact cache audit even though the seed geometry is identical.
    out.mkdir(exist_ok=False);cv2.setNumThreads(2);started=time.monotonic();c=Checks();results=[];dash=None
    try:
        config=load(HERE/"inputs.json");summary=load(run/"summary.json");start=load(run/"run_start.json");freeze=load(HERE/"freeze_final.json");new=load(HERE/"new_inference_receipt.json")
        manifest=list(csv.DictReader((DATA/"manifest.csv").open(encoding="utf-8-sig")));lookup={Path(r["output_file"]).stem:r for r in manifest}
        jobs=config["jobs"];ids=[j["id"] for j in jobs]
        c.test(len(manifest)==len(lookup)==400,"manifest400")
        c.test(len(ids)==len(set(ids))==292 and set(ids)<=set(lookup),"292_unique_open_images")
        c.test(sum(len(j["seeds"]) for j in jobs)==config["entry_count"]==301 and config["old_entry_count"]==299,"301_entries")
        unresolved={Path(r["file"]).stem for r in config["unresolved"]}
        c.test(len(unresolved)==108 and unresolved==set(lookup)-set(ids),"108_missing_entries_preserved")
        c.test(config["unresolved_is_not_absent_rope"],"missing_is_not_no_rope")
        c.test(all(r["dataset_use_status"]=="unlabeled_open_development_no_new_test_split" and "20260925" not in r["source_file"]+r["output_file"] for r in manifest),"allowed400_only")
        c.test(config["no_training"] and not config["labels_read"] and not config["reserved_data_accessed"] and not config["device_connected"],"input_scope")
        c.test(summary["completed"]==summary["planned"]==292 and summary["exit_code"]==0 and not summary["failures"],"all_run_complete",summary.get("failures"))
        sids=[r["id"] for r in summary["rows"]];c.test(len(sids)==len(set(sids))==292 and set(sids)==set(ids),"summary_ids")
        c.test(set(start["input_ids"])==set(ids) and len(start["input_ids"])==292,"start_ids")
        record_ids={p.name.removesuffix(".json.gz") for p in (run/"记录").glob("*.json.gz")};jpg_ids={p.stem for p in (run/"四栏检测结果").glob("*.jpg")}
        c.test(record_ids==jpg_ids==set(ids),"exact_record_gallery_ids")
        c.test(config["model_B"]["sha256"]==MODEL_SHA and new["model_sha256"]==MODEL_SHA,"frozen_model_binding");c.hash(config["model_B"]["path"],MODEL_SHA,"model_bytes_hash")
        c.test(new["forward_calls"]==2 and new["exit_code"]==0,"new_A7_inference")
        c.hash(HERE/"inputs.json",start["input_protocol_sha256"],"run_protocol_hash");c.hash(HERE/"inputs.json",freeze["protocol_sha256"],"freeze_protocol_hash");c.hash(H1/"protocol_inputs.json",config["input_protocol_sha256"],"H1_protocol_hash")
        c.test(start["sources"]==freeze["sources"],"core_freeze_equality")
        for p,expected in freeze["sources"].items():
            c.hash(p,expected,"current_core_hash");c.hash(run/"source_snapshot"/Path(p).name,expected,"snapshot_core_hash")
        c.test(summary["source_unchanged"],"source_unchanged")
        dash=dash_contract(run/"source_snapshot/run_h2.py");c.test(dash["passed"],"unknown_segments_dashed",dash)
        for added in config["new_entries"]:
            c.hash(added["cache"],added["cache_sha256"],"new_cache_hash")
            with np.load(added["cache"],allow_pickle=False) as z:
                seeds=z["seeds"];c.test(seeds.shape==(len(added["seeds"]),975,2048),"new_seed_cache_shape")
                for i,seed in enumerate(added["seeds"]):c.test(np.array_equal(seeds[i],seed_raster(seed["xy"])),"new_seed_raster_mapping",added["id"])
        smap={r["id"]:r for r in summary["rows"]}
        with (out/"checks.jsonl").open("x",encoding="utf-8") as log,ThreadPoolExecutor(a.workers) as ex:
            futures={ex.submit(audit_job,j,lookup[j["id"]],run,smap[j["id"]],new):j["id"] for j in jobs if j["id"] in lookup and j["id"] in smap}
            for future in as_completed(futures):
                row=future.result();results.append(row);log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");log.flush()
                if len(results)%12==0 or len(results)==len(futures):print(json.dumps(dict(audited=len(results),failed=sum(not r["passed"] for r in results),total=len(futures),seconds=round(time.monotonic()-started,1))),flush=True)
        c.test(len(results)==292 and sum(len(r["entries"]) for r in results)==301,"audited_denominator")
    except Exception as e:c.test(False,"global_audit_exception",f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    failed=[dict(id=r["id"],failures=r["failures"]) for r in results if not r["passed"]]
    report=dict(command=sys.argv,utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,audit_source_sha256=sha(norm(__file__)),exit_code=int(bool(c.failures or failed)),global_check_count=c.count,global_failures=c.failures,failed_images=failed,audited_images=len(results),audited_entries=sum(len(r["entries"]) for r in results),passed_images=sum(r["passed"] for r in results),render_dash_contract=dash,total_candidates=sum(r["candidates"] for r in results),unsupported_candidate_points=sum(r["unsupported_points"] for r in results),failure_codes=dict(Counter(f["code"] for r in results for f in r["failures"])),elapsed_seconds=time.monotonic()-started,labels_read=False,device_connected=False,identity_accuracy_verified=False,limitation="File, coordinate, response and renderer integrity only; no independent complete path or terminal identity verification.")
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("exit_code","audited_images","audited_entries","passed_images","elapsed_seconds")}),flush=True);sys.exit(report["exit_code"])

if __name__=="__main__":main()
