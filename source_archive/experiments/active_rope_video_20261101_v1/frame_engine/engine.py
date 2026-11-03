"""Causal video adapter around frozen B750, A7 and H6 (CPU, no devices).

Only an already-open source's earliest dataset image supplies scene geometry.
Subsequent target responses are new B forwards on the current decoded frame.
F3 geometry is a history prior, never advertised as freshly rerun F3.
Coordinates in the public interface are ORIGINAL unmirrored image pixels.
"""
from __future__ import annotations
import copy, csv, hashlib, importlib.util, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.append(str(ROOT/'vision_prototype/.venv/Lib/site-packages'))
import cv2
import numpy as np
import torch
import torch.nn.functional as F

H = ROOT/'experiments/entry_conditioned_rope_path'
H6 = H/'postprocess_b_20261031_v6'
A = ROOT/'数据资产/05_派生与实验结果'
A7 = A/'camera_condition_free_straight_20261030_v7'
MODEL = H/'runs/run_20261030_002/branches/B/model/checkpoints/step_0750.pt'
PARAMETERS = dict(entry_refresh_s=3.0, seed_length_original_px=120.,
    maximum_entry_conditions=2, entry_lk_fb_max_half_px=1.5,
    entry_lk_max_displacement_pitch_per_s=1., entry_raw_dual_min=.005,
    entry_raw_valid_fraction=.30, entry_lk_min_fraction=.60,
    scene_lk_fb_max_half_px=1.5, scene_lk_min_fraction=.25,
    scene_max_shift_original_px_per_s=8., scene_cumulative_max_pitch=2.,
    normalization_std_floor=.08, normalization_domain='entire established model crop',
    output_coordinates='original_unmirrored_image_xy_pixels',
    scene_source='causal propagation of earliest allowed saved SAM2/F3; not fresh F3',
    target_identity_confirmed=False)

def js(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

_TOOLS=None
def frozen_tools():
    global _TOOLS
    if _TOOLS is not None:return _TOOLS
    frozen=js(H6/'freeze_final.json')
    for p,v in (frozen['sources']|frozen['reused_sources']).items():
        if sha(p)!=v:raise ValueError('Frozen H6 source changed: '+p)
    names=['common','surface_tracks','topology','heat_free','connect_paths']
    previous={k:sys.modules.get(k) for k in names}
    for k in names:sys.modules.pop(k,None)
    sys.path.insert(0,str(H6))
    try:
        common=module('_video_frozen_h6_common',H6/'common.py');sys.modules['common']=common
        gallery=module('_video_frozen_h6_gallery',H6/'run_h6.py')
        surface=sys.modules['surface_tracks']
    finally:
        sys.path.remove(str(H6))
        for k in names:
            if previous[k] is None:sys.modules.pop(k,None)
            else:sys.modules[k]=previous[k]
    sys.path.insert(0,str(A7))
    try:detector=module('_video_frozen_a7',A7/'straight_detector.py')
    finally:sys.path.remove(str(A7))
    legacy=module('_video_frozen_b_model',H/'model.py')
    cp=torch.load(MODEL,map_location='cpu',weights_only=False)
    net=legacy.EntryUNet(cp['base_channels']);net.load_state_dict(cp['model_state']);net.eval()
    _TOOLS=(gallery,surface,detector,net)
    return _TOOLS

def anchor_job(video):
    """Metadata only; builds an allowed scene initializer even without old entry."""
    anchor=video['earliest_dataset_anchor'];ident=anchor['id']
    manifest=A/'camera_condition_images_20261009_v2/manifest.csv'
    with manifest.open(encoding='utf-8-sig',newline='') as f:
        row=next(r for r in csv.DictReader(f) if Path(r['output_file']).stem==ident)
    return dict(id=ident,image_path=str(manifest.parent/row['output_file']),
        image_sha256=row['output_sha256'],shape=[3000,4096],seeds=[],
        camera_role=row['camera_role'],source_group=row['source_group'])

def sample(a,xy):
    p=np.asarray(xy,np.float32)
    return cv2.remap(a,p[...,0].reshape(1,-1),p[...,1].reshape(1,-1),
        cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel().reshape(p.shape[:-1])

def seed_map(xy):
    mask=np.zeros((975,2048),np.uint8)
    q=np.rint((np.asarray(xy)-[0,1050])*.5).astype(np.int32)
    cv2.polylines(mask,[q],False,1,1,cv2.LINE_8)
    if not mask.any():raise ValueError('Entry is outside the established B crop')
    d=cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return np.exp(-.5*(d/2.)**2).astype(np.float32)

def lk(old,new,points):
    p=np.asarray(points,np.float32).reshape(-1,1,2)
    if not len(p):return p.reshape(-1,2),np.zeros(0,bool),np.zeros(0)
    q,s,e=cv2.calcOpticalFlowPyrLK(old,new,p,None,winSize=(25,25),maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.02))
    back,sb,eb=cv2.calcOpticalFlowPyrLK(new,old,q,None,winSize=(25,25),maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.02))
    fb=np.linalg.norm(back[:,0]-p[:,0],axis=1)
    ok=s.ravel().astype(bool)&sb.ravel().astype(bool)&np.isfinite(q[:,0]).all(1)
    return q[:,0],ok,fb

class FrameEngine:
    def __init__(self,anchor_job,anchor_time=0.,threads=2):
        cv2.setNumThreads(1);torch.set_num_threads(threads)
        self.gallery,self.surface,self.a7,self.net=frozen_tools()
        self.anchor_job=copy.deepcopy(anchor_job);self.anchor_time=float(anchor_time)
        self.priors=self.gallery.load_surface_priors(dict(anchor_job,seeds=[]))
        self.pitch=float(self.priors['pitch_px'])
        self.anchor_half=self.priors['original_half'].copy()
        # Only scene geometry survives initialization. Historical target paths,
        # old A6/A7 target candidates and old B maps are never fed to video.
        self.priors.update(a6_entries=[],a7_candidates=[],seed_source_records={})
        gray=cv2.cvtColor(self.anchor_half[525:],cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        zero=np.zeros(gray.shape,np.float32)
        self.anchor_tracks=self.gallery.build_tracks(self.priors,zero,gray)['tracks']
        self.tracks=copy.deepcopy(self.anchor_tracks)
        self.anchor_masks={k:self.priors[k].copy() for k in ['surface_half','sam2_surface_half']}
        self.scene_offset=np.zeros(2);self.scene_ok=True
        self.prev_gray=None;self.prev_t=None;self.seeds=[];self.last_entry_refresh=-float('inf')
        self.forward_total=0;self.frame_number=0;self.last_entry_diagnostic={}
        self.receipt=dict(anchor_id=anchor_job['id'],anchor_time_s=self.anchor_time,
            input_dependency_sha256=self.priors['input_dependency_sha256'],
            model_path=str(MODEL),model_sha256=sha(MODEL),
            historical_target_paths_used=False,historical_response_used=False)

    def _scene(self,gray,t):
        """One common bounded displacement preserves saved center-lane ordering."""
        diag=dict(source=PARAMETERS['scene_source'],age_s=float(t-self.anchor_time),
            new_f3_measurement=False,scene_shift_half_xy=[0.,0.],lk_fraction=None)
        if self.prev_gray is not None and self.tracks:
            # Material texture rotates; it must not drag the scene tracks.
            # Register only static upper machinery outside the spool/search ROI.
            mask=np.zeros(self.prev_gray.shape,np.uint8);mask[25:500,25:2023]=255
            corners=cv2.goodFeaturesToTrack(self.prev_gray,maxCorners=400,
                qualityLevel=.01,minDistance=18,mask=mask,blockSize=7)
            pts=np.empty((0,2),np.float32) if corners is None else corners[:,0]
            q,ok,fb=lk(self.prev_gray,gray,pts)
            dt=max(1e-3,float(t-self.prev_t));disp=q-pts
            ok&=(fb<=PARAMETERS['scene_lk_fb_max_half_px'])
            ok&=np.linalg.norm(disp,axis=1)<=max(1.,PARAMETERS['scene_max_shift_original_px_per_s']*dt*.5)
            frac=float(ok.mean()) if len(ok) else 0.;diag['lk_fraction']=frac
            diag['lk_domain']='static_upper_machinery_original_y_50_to_1000_outside_model_crop'
            diag['lk_sample_count']=len(pts)
            if frac>=PARAMETERS['scene_lk_min_fraction']:
                shift=np.median(disp[ok],axis=0)
                # Texture rotates; robust scene motion is only a scene prior.
                self.scene_offset+=shift;self.scene_ok=True
                diag['scene_shift_half_xy']=shift.tolist()
            else:self.scene_ok=False
        if np.linalg.norm(self.scene_offset)*2>PARAMETERS['scene_cumulative_max_pitch']*self.pitch:
            self.scene_ok=False;diag['cumulative_motion_limit_exceeded']=True
        self.tracks=copy.deepcopy(self.anchor_tracks)
        for tr in self.tracks:
            for k in ['points_xy','midpoint_xy','upper_boundary_xy','lower_boundary_xy']:
                tr[k]=(np.asarray(tr[k])+self.scene_offset).tolist()
            tr['video_source_age_s']=float(t-self.anchor_time)
            tr['video_scene_geometry_supported']=bool(self.scene_ok)
        transform=np.array([[1,0,self.scene_offset[0]],[0,1,self.scene_offset[1]]],np.float32)
        for k,m in self.anchor_masks.items():
            self.priors[k]=cv2.warpAffine(m,transform,(2048,1500),flags=cv2.INTER_NEAREST)
        diag.update(scene_geometry_supported=bool(self.scene_ok),cumulative_shift_original_xy=(2*self.scene_offset).tolist())
        return diag

    def _raw_seed_check(self,gray,xy,width):
        p=np.linspace(np.asarray(xy)[0],np.asarray(xy)[-1],24)*.5
        d=p[-1]-p[0];d/=max(np.linalg.norm(d),1e-6);n=np.array([-d[1],d[0]])
        offsets=np.array([-1.,-.6,0,.6,1.])*(width*.5)
        v=sample(gray.astype(np.float32)/255,p[:,None,:]+n*offsets[None,:,None])
        dual=np.minimum(v[:,2]-v[:,:2].mean(1),v[:,2]-v[:,3:].mean(1))
        # Both bright and dark optical bodies can seed, but require two sides.
        dark=np.minimum(v[:,:2].mean(1)-v[:,2],v[:,3:].mean(1)-v[:,2])
        support=np.maximum(dual,dark)>=PARAMETERS['entry_raw_dual_min']
        return float(support.mean())

    def _propagate_entries(self,gray,t):
        kept=[];diagnostics=[]
        if self.prev_gray is None:return kept,diagnostics
        dt=max(.001,t-self.prev_t)
        for seed in self.seeds:
            xy=np.asarray(seed['xy']);pts=np.linspace(xy[0],xy[-1],16)*.5
            q,ok,fb=lk(self.prev_gray,gray,pts)
            ok&=fb<=PARAMETERS['entry_lk_fb_max_half_px']
            ok&=np.linalg.norm(q-pts,axis=1)*2<=max(4.,self.pitch*dt*PARAMETERS['entry_lk_max_displacement_pitch_per_s'])
            frac=float(ok.mean());accepted=frac>=PARAMETERS['entry_lk_min_fraction']
            if accepted:
                shift=np.median((q-pts)[ok],axis=0)*2
                newxy=xy+shift
                raw=self._raw_seed_check(gray,newxy,seed['width_px'])
                accepted=raw>=PARAMETERS['entry_raw_valid_fraction']
            else:raw=0.
            diagnostics.append(dict(entry_id=seed['entry_id'],lk_fraction=frac,current_raw_support_fraction=raw,accepted=bool(accepted)))
            if accepted:
                s=copy.deepcopy(seed);s.update(xy=newxy.tolist(),source='causal_LK_entry_with_current_raw_dual_check',
                    last_current_image_check_time_s=float(t),entry_age_s=float(t-seed['detected_time_s']))
                kept.append(s)
        return kept,diagnostics

    def _capture_entries(self,bgr,gray,t):
        surface=cv2.resize(self.priors['sam2_surface_half'],(4096,3000),interpolation=cv2.INTER_NEAREST)
        result=self.a7.detect(dict(original=bgr,surface=surface,pitch=self.pitch))
        self.priors['a7_candidates']=result['candidates']
        # Distance to the actual scene ROI, not image/patch boundary, determines
        # which end of this *observed finite segment* approaches the spool.
        distance=cv2.distanceTransform((surface==0).astype(np.uint8),cv2.DIST_L2,3)
        seeds=[]
        for c in result['candidates']:
            if not c.get('plausible') or c.get('anchor_span_px',0)<100:continue
            seg=np.asarray(c['anchor_segment_xy'],float)
            dist=sample(distance,seg)
            if len(seg)<2 or max(dist)<20:continue
            if dist[0]<dist[-1]:seg=seg[::-1]
            d=seg[-1]-seg[0];length=np.linalg.norm(d)
            if length<60:continue
            d/=length;start=seg[0]+d*min(12.,length*.05)
            end=start+d*min(PARAMETERS['seed_length_original_px'],length-20)
            xy=np.stack([start,end]);raw=self._raw_seed_check(gray,xy,float(c['width_px']))
            if raw<PARAMETERS['entry_raw_valid_fraction']:continue
            if np.any(xy[:,1]<1050) or np.any(xy[:,1]>=3000) or np.any(xy[:,0]<0) or np.any(xy[:,0]>=4096):continue
            seeds.append(dict(xy=xy.tolist(),source='current_raw_frame_A7',entry_id='video_'+c['candidate_id'],
                a7_candidate_id=c['candidate_id'],width_px=float(c['width_px']),automatic=True,
                detected_time_s=float(t),entry_age_s=0.,last_current_image_check_time_s=float(t),
                current_raw_support_fraction=raw,direction_ambiguous=bool(abs(float(dist[0]-dist[-1]))<20),
                direction_basis='finite_observed_outside_segment_toward_SAM2_scene_ROI',identity_confirmed=False))
            if len(seeds)>=PARAMETERS['maximum_entry_conditions']:break
        return seeds,dict(status=result['status'],candidate_count=len(result['candidates']),valid_entries=len(seeds),
            old_A1_A2_A6_candidates_used=False,source='fresh_current_raw_A7')

    def _forward(self,gray,seed):
        # Exact B grayscale ordering: original gray first, then area reduction.
        g=torch.from_numpy(gray.copy());s=torch.from_numpy(seed_map(seed['xy']))
        mean=g.mean();std=g.std(unbiased=False).clamp_min(.08)
        x=F.pad(torch.stack([(g-mean)/std,s]),(0,0,0,17))[None]
        with torch.inference_mode():heat=self.net(x)[0,0,:975,:2048].numpy().copy()
        self.forward_total+=1
        return heat,dict(mean=float(mean),std=float(std),min=float(gray.min()),max=float(gray.max()))

    def _measured_tracks(self,gray,heat):
        tracks=copy.deepcopy(self.tracks);old=self.surface.OLD
        for tr in tracks:
            p=np.asarray(tr['points_xy']);d=np.gradient(p,axis=0);d/=np.maximum(np.linalg.norm(d,axis=1)[:,None],1e-6)
            n=np.c_[-d[:,1],d[:,0]];w=np.asarray(tr['width_px'])
            raw,hs,hl,rd,hd=old._appearance(gray,heat,p,n,w)
            ro=raw>=old.PARAMETERS['raw_support_threshold']
            ho=(hs>=old.PARAMETERS['heat_support_threshold'])&(hl>=old.PARAMETERS['heat_level_support_threshold'])
            observed=np.asarray(tr['f3_pair_observed'],bool)&self.scene_ok
            tr.update(raw_score=raw.tolist(),heat_score=hs.tolist(),heat_level=hl.tolist(),
                raw_signed_center_minus_sides=rd.tolist(),heat_signed_center_minus_sides=hd.tolist(),
                image_supported=ro.tolist(),heat_supported=ho.tolist(),supported=(ro&ho&observed).tolist(),
                score=float(np.mean(.45*raw+.35*hs+.2*hl)),
                video_current_image_support_fraction=float(ro.mean()),
                video_source='historical_F3_geometry_causal_bounded_scene_LK_current_raw_and_B_profiles',
                historical_F3_pair_flags_are_not_fresh_boundary_detection=True)
        return tracks

    def step(self,bgr,t,refresh_entry=False):
        start=time.perf_counter();t=float(t)
        if bgr.shape!=(3000,4096,3) or bgr.dtype!=np.uint8:raise ValueError('Expected original 4096x3000 uint8 BGR')
        if t+1e-7<self.anchor_time:raise ValueError('Scene anchor is in the future')
        if self.prev_t is not None and t<=self.prev_t:raise ValueError('Causal strictly increasing frame time required')
        fullgray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        halfgray=cv2.resize(fullgray,(2048,1500),interpolation=cv2.INTER_AREA)
        self.priors['original_half']=cv2.resize(bgr,(2048,1500),interpolation=cv2.INTER_AREA)
        diag={'scene':self._scene(halfgray,t)}
        propagated,entrychecks=self._propagate_entries(halfgray,t)
        refresh=refresh_entry or not propagated or t-self.last_entry_refresh>=PARAMETERS['entry_refresh_s']
        self.priors['a7_candidates']=[] # Never reuse old per-point A7 observed flags.
        if refresh:
            tick=time.perf_counter();fresh,entrydiag=self._capture_entries(bgr,halfgray,t)
            self.last_entry_refresh=t;diag['entry_refresh_seconds']=time.perf_counter()-tick
            self.seeds=fresh if fresh else propagated
            diag['entry_capture']=entrydiag
        else:self.seeds=propagated
        diag['entry_propagation']=entrychecks
        gray=cv2.resize(fullgray[1050:].astype(np.float32)/255,(2048,975),interpolation=cv2.INTER_AREA)
        entries=[];heats=[];allranked=[];tracksets=[];before=self.forward_total
        for ei,seed in enumerate(self.seeds):
            tick=time.perf_counter();heat,stats=self._forward(gray,seed)
            infer_time=time.perf_counter()-tick
            tracks=self._measured_tracks(gray,heat)
            tick=time.perf_counter();full=self.gallery.describe_free(self.priors,seed,heat)
            free,projection=self.gallery.prepare_task_free(full,tracks)
            # A failed scene registration cannot authorize a new attachment to
            # old geometry. Keep image-supported free observations separately.
            connection_tracks=tracks if self.scene_ok else []
            linked=self.gallery.connect(gray,heat,free,connection_tracks,[],self.priors['surface_half'][525:])
            entry=dict(seed=seed,free=free,connection=linked,normalization=stats,
                forward_seconds=infer_time,h6_seconds=time.perf_counter()-tick,
                free_projection=projection)
            entries.append(entry);heats.append(heat);tracksets.append(tracks)
        if entries:self.gallery.task_direction_feasibility(entries,self.priors['sam2_surface_half'][525:])
        for ei,e in enumerate(entries):
            for c in e['connection']['candidates']:
                allranked.append((e['task_direction']['selection_allowed'],c['rank_score'],ei,c))
        allranked.sort(key=lambda x:(x[0],x[1],x[3]['metrics'].get('free',{}).get('quality',0)),reverse=True)
        selected=allranked[0][2] if allranked else 0
        public=[]
        for allowed,score,ei,c in allranked[:3]:
            p=copy.deepcopy(c);p['points_xy']=(np.asarray(c['points_xy'])*2+[0,1050]).tolist()
            p.update(coordinate_system=PARAMETERS['output_coordinates'],entry_index=ei,
                task_direction_allowed=bool(allowed),seed_xy=self.seeds[ei]['xy'],
                response_is_identity_probability=False)
            public.append(p)
        selected_tracks=tracksets[selected] if tracksets else self.tracks
        simplified=[dict(track_id=tr['track_id'],points_xy=(np.asarray(tr['points_xy'])*2+[0,1050]).tolist(),
            inferred=tr['f3_pair_inferred'] if self.scene_ok else [True]*len(tr['points_xy']),
            current_image_supported=tr.get('image_supported',[False]*len(tr['points_xy'])),
            source_age_s=float(t-self.anchor_time),source=PARAMETERS['scene_source']) for tr in selected_tracks]
        self.prev_gray=halfgray;self.prev_t=t;self.frame_number+=1
        diag.update(seconds=time.perf_counter()-start,frame_number=self.frame_number,
            forward_count=self.forward_total-before,forward_total=self.forward_total,
            entry_count=len(entries),track_count=len(simplified),candidate_count=len(public),
            selected_entry=selected if entries else None,
            entries=[{k:e[k] for k in ['normalization','forward_seconds','h6_seconds','task_direction']} for e in entries],
            scene_anchor_id=self.anchor_job['id'],future_records_used=False,
            historical_response_reused=False,complete_identity_verified=False)
        return dict(t=t,heat=heats[selected] if heats else None,heatmaps_by_entry=heats,
            normalization_by_entry=[e['normalization'] for e in entries],seeds=copy.deepcopy(self.seeds),
            raw_candidates=public,surface_tracks=simplified,pitch_original_px=self.pitch,
            diagnostics=diag,status='candidate' if public else 'entry_without_path' if heats else 'unknown_no_entry')
