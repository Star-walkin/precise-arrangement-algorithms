"""SG2: current-evidence segmented S1, bounded memory, transactional identity.

Frozen S1 remains unmodified. Search-memory is never an output measurement.
Current BG5 is a search prior, not proof of active-rope identity or contact.
"""
from pathlib import Path
import sys, time, math
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
RES=ROOT.parent/'RopeEdgeSuite_R1_20261107'
FROZEN=RES/'baseline/source'
sys.path.insert(0,str(FROZEN/'LC1_fixed_d2238p7_full_20261106_v1'))
sys.path.insert(0,str(FROZEN/'LC1_locked_column_20261105_v1'))
from live_s1_tracker_anchored import LiveS1Tracker as Anchored
from lc1_s1_adapter import S1Adapter
from live_s1_tracker import pixel_sha256

PARAMETERS=dict(memory_ttl_s=.5,minimum_span_px=400.,minimum_points=24,
    minimum_support_fraction=.65,max_gap_widths=4.,max_gap_px=120.,
    width_ratio=[.65,1.45],max_angle_change_deg=2.,max_residual_widths=.18,
    candidate_confirmations=2,local_seed_x_offsets=[0.,160.,320.])

def line_y(abc,x):return float(-(abc[0]*x+abc[2])/abc[1])
def angle(abc):return math.atan2(-abc[0],abc[1])

class SegmentedS1(S1Adapter):
    def __init__(self):
        super().__init__(retain_step_records=True)
        self.current_width=24.

    def step(self,image,seed_xy,*,seed_source):
        gap=min(PARAMETERS['max_gap_px'],PARAMETERS['max_gap_widths']*self.current_width)
        self.config['max_unobserved_gap_px']=gap
        self.overrides={'max_unobserved_gap_px':gap}
        out=super().step(image,seed_xy,seed_source=seed_source)
        diag=out['diagnostics'];p=diag.get('s1_prediction',{})
        rows=p.pop('step_records',[])
        seg=dict(valid=False,reasons=[],gap_limit_px=gap,current_supported_points_only=True,
                 extrapolated_points_used_in_fit=False,bridges=[])
        diag['SG2_segments']=seg
        if not out['valid']:return out
        ids=np.array([i for i,v in enumerate(rows) if v['supported']],int)
        pts=np.asarray(out['line_points_xy'],float)
        if len(ids)!=len(pts):raise ValueError('S1 record/point correspondence changed')
        width=float(p['width_px']);v=np.asarray(seed_xy)[1]-np.asarray(seed_xy)[0];v/=np.linalg.norm(v)
        span=float(np.ptp(pts@v));fraction=len(ids)/max(1,ids[-1]-ids[0]+1)
        residual=np.abs(pts@np.asarray(out['line_abc'])[:2]+out['line_abc'][2])
        q90=float(np.percentile(residual,90))
        if span<PARAMETERS['minimum_span_px']:seg['reasons'].append('current_support_span_insufficient')
        if len(ids)<PARAMETERS['minimum_points']:seg['reasons'].append('current_support_points_insufficient')
        if fraction<PARAMETERS['minimum_support_fraction']:seg['reasons'].append('current_support_fraction_insufficient')
        if q90>max(2.,PARAMETERS['max_residual_widths']*width):seg['reasons'].append('current_line_residual_excessive')
        for j in np.flatnonzero(np.diff(ids)>1):
            before=ids[max(0,j-3):j+1];after=ids[j+1:j+5]
            a=float(np.median([rows[k]['measured_width_px'] for k in before]))
            b=float(np.median([rows[k]['measured_width_px'] for k in after]))
            pol1=int(np.median([rows[k]['polarity'] for k in before]));pol2=int(np.median([rows[k]['polarity'] for k in after]))
            missing=(ids[j+1]-ids[j]-1)*float(self.config.get('step_px',4.))
            ratio=b/max(a,1e-9)
            okay=bool(missing<=gap and .65<=ratio<=1.45 and pol1==pol2)
            seg['bridges'].append(dict(missing_px=float(missing),before_width_px=a,after_width_px=b,
                polarity_consistent=pol1==pol2,valid=okay))
            if not okay:seg['reasons'].append('gap_end_evidence_inconsistent')
        seg.update(valid=not seg['reasons'],support_span_px=span,support_fraction=fraction,
                   residual_q90_px=q90,observed_count=len(pts),width_px=width)
        if not seg['valid']:
            out['valid']=False;diag['adapter_gate']='SG2_segment_validation_failed'
        return out

class Tracker(Anchored):
    def __init__(self,mask,reacquirer=None):
        super().__init__(mask,s1_adapter=SegmentedS1(),reacquire_callback=None)
        self.reacquirer=reacquirer;self.trusted=None;self.pending=None;self.history=[]
        self.current_region=None;self.attempts=[];self.entry_seed=None;self.entry_witness=None

    def set_region(self,region):
        self.current_region=region
        if region['record']['valid']:
            self.mask=region['surface_mask']>0
            self.outside_distance=cv2.distanceTransform((~self.mask).astype(np.uint8),cv2.DIST_L2,3)

    def identity(self,result,seed,witness,t):
        why=[];metrics={}
        if not result['valid']:return False,['optical_support_rejected'],metrics
        abc=result['line_abc'];width=float(witness['width_px'])
        if self.trusted and t-self.trusted['t']<=PARAMETERS['memory_ttl_s']:
            old=self.trusted;dt=t-old['t'];ratio=width/old['width']
            da=abs(math.degrees(angle(abc)-angle(old['abc'])))
            x=old['reference_x'];pred=line_y(old['abc'],x)+self.velocity()*dt
            innovation=abs(line_y(abc,x)-pred)
            metrics.update(width_ratio=ratio,angle_delta_deg=da,innovation_px=innovation,dt_s=dt)
            if not .65<=ratio<=1.45:why.append('rope_width_identity_mismatch')
            if da>PARAMETERS['max_angle_change_deg']:why.append('rope_angle_identity_mismatch')
            if innovation>max(16.,1.3*old['width']+20*dt):why.append('rope_position_identity_mismatch')
        else:
            # Cold/reinitialization must have measurable support outside the CURRENT
            # coil, and a trajectory pointing toward its current horizontal extent.
            region=self.current_region
            if region is None or not region['record']['valid']:
                why.append('cold_start_current_region_unavailable')
            else:
                pts=np.asarray(result['line_points_xy'],float)
                q=np.rint(pts).astype(int);h,w=self.mask.shape
                good=(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
                outside=np.zeros(len(pts),bool)
                outside[good]=self.outside_distance[q[good,1],q[good,0]]>=12.
                free=pts[outside]
                free_span=float(np.ptp(free[:,0])) if len(free)>1 else 0.
                metrics['current_free_support_x_span_px']=free_span
                entry_ok=False
                ew=self.entry_witness
                if ew and ew.get('valid'):
                    ep=np.asarray(ew['points_xy'],float);eq=np.rint(ep).astype(int)
                    ev=(eq[:,0]>=0)&(eq[:,0]<w)&(eq[:,1]>=0)&(eq[:,1]<h)
                    outside_entry=np.zeros(len(ep),bool)
                    outside_entry[ev]=self.outside_distance[eq[ev,1],eq[ev,0]]>=12.
                    error=float(np.median(np.abs(ep@np.asarray(abc)[:2]+abc[2])))
                    ratio=width/ew['width_px']
                    entry_ok=bool(outside_entry.sum()>=4 and np.ptp(ep[outside_entry,0])>=60.
                        and error<.75*width and .65<=ratio<=1.45)
                    metrics.update(entry_anchor_current_valid=entry_ok,entry_line_residual_px=error,entry_width_ratio=ratio)
                if (len(free)<24 or free_span<160) and not entry_ok:why.append('cold_start_free_support_insufficient')
                top=np.asarray(region['record'].get('top_curve_xy',[]),float)
                if len(top):
                    center=float(np.median(top[:,0]));direction=seed[1]-seed[0]
                    if (center-seed[0,0])*direction[0]<=0:why.append('candidate_points_away_from_current_coil')
        return not why,why,metrics

    def velocity(self):
        if len(self.history)<2:return 0.
        rates=[(b[1]-a[1])/(b[0]-a[0]) for a,b in zip(self.history[:-1],self.history[1:]) if 0<b[0]-a[0]<=.5]
        return float(np.clip(np.median(rates),-50,50)) if rates else 0.

    def trial(self,image,seed,source,t):
        self.entry_witness=self._seed_witness(image,self.entry_seed) if self.entry_seed is not None else None
        seed=np.asarray(seed,float).reshape(2,2);witness=self._seed_witness(image,seed)
        log=dict(source=source,seed_xy=seed.tolist(),seed_valid=bool(witness.get('valid')),accepted=False)
        self.attempts.append(log)
        if not witness.get('valid'):
            log['reasons']=[witness.get('reason','seed_bilateral_failure')];return None
        self.s1.current_width=float(witness['width_px'])
        result=self.s1.step(image,seed.tolist(),seed_source=source)
        valid,why,identity=self.identity(result,seed,witness,t)
        log.update(reasons=why,identity=identity,segments=result['diagnostics'].get('SG2_segments'),
                   seed_width_px=witness['width_px'],accepted=valid)
        return (result,seed,witness,source,identity) if valid else None

    def step(self,image,t,i):
        tick=time.perf_counter();t=float(t);i=int(i);self.attempts=[]
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);got=None;tracking={};reacquire_log=None;forwards=0
        self.entry_seed=None;self.entry_witness=None
        short=bool(self.trusted and 0<t-self.trusted['t']<=PARAMETERS['memory_ttl_s'])
        if short:
            seed,wit,tracking=self._track_seed(image,gray,t,i)
            if seed is not None:got=self.trial(image,seed,'SG2_current_LK_fixed_anchor',t)
            if got is None:
                # Bounded spatial windows, not material-point marching. All seed
                # witnesses and every fitted support point are remeasured now.
                old=self.trusted;abc=old['abc'];xbase=old['reference_x'];dt=t-old['t']
                direction=np.array([abc[1],-abc[0]]);direction/=np.linalg.norm(direction)
                if old['direction'][0]<0:direction=-direction
                for dx in PARAMETERS['local_seed_x_offsets']:
                    if got:break
                    x=xbase+dx*np.sign(direction[0])
                    for shift in (0.,-8.,8.,-16.,16.):
                        y=line_y(abc,x)+self.velocity()*dt+shift
                        seed=np.array([[x,y],[x,y]+120.*direction])
                        got=self.trial(image,seed,'SG2_bounded_current_memory_search',t)
                        if got:break
        elif self.trusted:
            self.prev=None;tracking['reason']='search_memory_expired'
        # A pending new identity is checked again on new pixels before committing.
        if got is None and self.pending and i==self.pending['frame']+1:
            self.entry_seed=self.pending.get('entry_seed')
            got=self.trial(image,self.pending['seed'],'SG2_pending_current_revalidation',t)
        if got is None and self.reacquirer:
            candidate=self.reacquirer(image,t,i,self.current_region)
            if candidate:
                reacquire_log=candidate.get('diagnostics');forwards=candidate.get('model_forward_count',0)
                if candidate['source_pixel_sha256']!=pixel_sha256(image):raise ValueError('noncurrent seed')
                for proposal in candidate.get('seeds',[]):
                    seed=proposal['seed_xy'];self.entry_seed=proposal['entry_seed']
                    got=self.trial(image,seed,'SG2_current_BG5_P1_candidate',t)
                    if got:break
        accepted=bool(got)
        if got and not short:
            result,seed,witness,source,identity=got
            abc=result['line_abc'];p=self.pending
            consistent=bool(p and i==p['frame']+1 and abs(math.degrees(angle(abc)-angle(p['abc'])))<2
                and abs(line_y(abc,seed[0,0])-line_y(p['abc'],seed[0,0]))<30
                and .65<witness['width_px']/p['width']<1.45)
            count=p['count']+1 if consistent else 1
            self.pending=dict(frame=i,seed=seed.tolist(),abc=abc,width=witness['width_px'],count=count,
                entry_seed=None if self.entry_seed is None else np.asarray(self.entry_seed).tolist())
            accepted=count>=PARAMETERS['candidate_confirmations']
        if accepted:
            result,seed,witness,source,identity=got;abc=result['line_abc']
            pts=np.asarray(result['line_points_xy'],float)
            expected=result['diagnostics'].get('s1_prediction',{}).get('seed_measurement',{})
            matches=bool(witness['support_count']==expected.get('supported_profiles')
                and abs(witness['width_px']-expected.get('width_px',-999))<.01
                and witness['polarity']==expected.get('selected_polarity'))
            support=np.vstack([pts,witness['points_xy']]) if matches else pts
            direction=seed[1]-seed[0];direction/=np.linalg.norm(direction)
            if not short:
                reference_x=float(seed[0,0]);self.history=[]
            else:reference_x=self.trusted['reference_x']
            self.trusted=dict(t=t,frame=i,abc=abc,width=witness['width_px'],reference_x=reference_x,direction=direction)
            self.prev=dict(gray=gray,tracking_support_xy=support,direction=direction,t=t,frame_index=i)
            # Commit only AFTER current support + identity validation.
            if self.anchor_x_original!=reference_x:self._set_anchor([[reference_x,line_y(abc,reference_x)],seed[1]],source,i,'SG2_accepted_current_identity')
            self.history=(self.history+[(t,line_y(abc,reference_x))])[-9:];self.pending=None
            reason='SG2_current_segmented_identity_accepted';confidence=min(1.,len(pts)/100.)
        else:
            result=got[0] if got else dict(valid=False,line_abc=None,line_points_xy=[],supported_segments_xy=[],diagnostics={})
            seed=got[1] if got else None;witness=got[2] if got else None;source=got[3] if got else None
            abc=None;confidence=0.
            reason='SG2_identity_confirmation_pending' if got else ('SG2_short_loss_search_memory_retained' if short else 'SG2_reacquisition_required')
        return dict(valid=accepted,current_line_abc=abc,line_abc=abc,
            line_points_xy=result['line_points_xy'] if accepted else [],
            supported_segments_xy=result.get('supported_segments_xy',[]) if accepted else [],
            confidence=confidence,confidence_is_uncalibrated_heuristic=True,source='RES-1.2/SG2',
            seed_xy_original=None if seed is None else seed.tolist(),seed_source=source,S1=result,
            diagnostics=dict(accept_reason=reason,tracking=tracking,reacquire=reacquire_log,attempts=self.attempts,
                seed_validation=witness,current_region_valid=bool(self.current_region and self.current_region['record']['valid']),
                trusted_frame=None if self.trusted is None else self.trusted['frame'],
                search_memory_ttl_s=PARAMETERS['memory_ttl_s'],history_ABC_held_as_measurement=False,
                seed_witness_matches_primary_S1_seed_measurement=bool(accepted and matches),
                spatial_anchor=dict(x_original=self.anchor_x_original,established_frame=self.anchor_frame,
                    reset_count=self.anchor_reset_count,committed_only_after_current_identity_validation=True),
                total_seconds=time.perf_counter()-tick),model_forward_count=forwards,
            same_rope_identity_confirmed=False,temporal_identity_consistent=accepted,whole_path_candidate_supported=False)

class DynamicReacquire:
    def __init__(self,pitch):self.pitch=pitch;self.adapter=None;self.last=-100;self.calls=0
    def __call__(self,image,t,i,region):
        if i-self.last<5 or region is None or not region['record']['valid']:return None
        self.last=i;self.calls+=1
        from lc1_p1_adapter import P1H6Adapter
        surface=region['surface_mask']
        if self.adapter is None:
            self.adapter=P1H6Adapter(surface,self.pitch,FROZEN/'LC1_locked_column_20261105_v1/supplied_assets/step_0750.pt',
                threads=2,camera='left',surface_mode='entry_band')
        else:self.adapter.surface=(surface>0).astype(np.uint8)*255
        p=self.adapter.step(image)
        seeds=[]
        if p.get('seed_xy'):seeds.append(p['seed_xy'])
        for entry in p.get('diagnostics',{}).get('entry_summary',[]):
            q=entry['seed']['xy']
            if q not in seeds:seeds.append(q)
        proposals=[]
        for q in seeds:
            base=np.asarray(q,float);v=base[1]-base[0];v/=np.linalg.norm(v)
            for distance in (0.,160.,320.,480.,640.,800.):
                trial=base+v*distance
                if np.all(trial[:,0]>=40) and np.all(trial[:,0]<image.shape[1]-40):
                    proposals.append(dict(seed_xy=trial.tolist(),entry_seed=q))
        return dict(seeds=proposals,source_pixel_sha256=pixel_sha256(image),model_forward_count=p.get('model_forward_count',0),
            diagnostics=dict(source='current_BG5_surface_into_A7_B750_H6',current_surface_sha256=pixel_sha256(surface),
                status=p['status'],reasons=p['reasons'],candidate_count=len(seeds),
                whole_path_supported=p['whole_path_candidate_supported']))
