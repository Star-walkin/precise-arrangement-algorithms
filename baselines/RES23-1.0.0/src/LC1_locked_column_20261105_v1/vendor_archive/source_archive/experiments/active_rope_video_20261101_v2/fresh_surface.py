"""Fresh current-frame groove observations and adjacent-boundary centers.

F3/D4 principles: ROI weighted local contrast, tangent integration to suppress
fine strands, original-grey two shoulders, finite ordered grooves, and exact
adjacent-boundary midpoints. This is a lighter video observer, not a replay of
the original full F3 fusion. No saved center/boundary coordinates enter detect.
Only a scene ROI, uncalibrated pitch and broad C3 quadratic direction guide do.
"""
import cv2,numpy as np,time,hashlib
P=dict(column_step_half_px=4,normal_sigma_pitch=.07,tangent_sigma_pitch=.75,
    local_sigma_pitch=.9,std_floor_gray=4.,weak_raw_contrast=.20,strong_raw_contrast=.60,
    minimum_z_drop=.14,minimum_gap_width_pitch=.12,minimum_peak_spacing_pitch=.55,
    search_half_pitch=.42,maximum_gap_columns=2,minimum_boundary_span_original_px=48.,
    minimum_center_span_original_px=40.,pair_spacing_pitch=[.62,1.48],
    minimum_observed_pair_fraction=.65,boundary_smoothing_columns=1.0)

def runs(v):
    d=np.diff(np.r_[False,v,False].astype(np.int8));return zip(np.flatnonzero(d==1),np.flatnonzero(d==-1))
def remap(a,x,y):return cv2.remap(a,np.asarray(x,np.float32),np.asarray(y,np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
def wblur(a,m,sx,sy):
    den=cv2.GaussianBlur(m,(0,0),sx,sigmaY=sy)
    return cv2.GaussianBlur(a*m,(0,0),sx,sigmaY=sy)/np.maximum(den,1e-5),den

class FreshSurface:
    def __init__(self,mask_half,pitch_original,reference):
        self.mask=(mask_half[525:]>0).astype(np.float32);self.pitch=float(pitch_original)*.5
        self.ref=reference or {};self.calls=0
        # ROI is a search-domain prior. All line locations below come from the
        # current image, including newly appearing rows and changed curvature.
        yy,xx=np.nonzero(self.mask)
        self.x0=int(xx.min()) if len(xx) else 0;self.x1=int(xx.max())+1 if len(xx) else 0
        self.y0=int(yy.min()) if len(yy) else 0;self.y1=int(yy.max())+1 if len(yy) else 0
    def detect(self,gray_half):
        tick=time.perf_counter();self.calls+=1;p=self.pitch;step=P['column_step_half_px']
        raw=gray_half[525:].astype(np.float32)
        if not self.x1:return [],dict(status='empty_roi',fresh_detection=True,calls=self.calls)
        xs=np.arange(self.x0,self.x1,step,dtype=np.float32)
        ref=self.ref;u=(2*xs-float(ref.get('center_x_px',2048.)))/max(1.,float(ref.get('scale_x_px',1000.)))
        guide=.5*(float(ref.get('b_px_per_u',0.))*u+float(ref.get('c_px_per_u2',0.))*u*u)
        gy=np.arange(max(0,self.y0-int(3*p)),min(975,self.y1+int(3*p)),dtype=np.float32)
        xx=np.broadcast_to(xs[None,:],(len(gy),len(xs)));yy=gy[:,None]+guide[None,:]
        flat=remap(raw,xx,yy);mask=remap(self.mask,xx,yy)
        tangent,coverage=wblur(flat,mask,max(.6,P['tangent_sigma_pitch']*p/step),max(.6,P['normal_sigma_pitch']*p))
        mu,_=wblur(tangent,mask,max(.6,P['local_sigma_pitch']*p/step),P['local_sigma_pitch']*p)
        second,_=wblur(tangent*tangent,mask,max(.6,P['local_sigma_pitch']*p/step),P['local_sigma_pitch']*p)
        std=np.sqrt(np.maximum(second-mu*mu,0));z=(tangent-mu)/np.maximum(std,P['std_floor_gray'])
        shoulder=max(2,int(round(.34*p)))
        upper=np.roll(tangent,shoulder,axis=0);lower=np.roll(tangent,-shoulder,axis=0)
        weak=np.minimum(upper-tangent,lower-tangent);strong=np.maximum(upper-tangent,lower-tangent)
        support=(mask>.98)&(coverage>.95)&(np.roll(mask,shoulder,0)>.98)&(np.roll(mask,-shoulder,0)>.98)
        dark=(z < -P['minimum_z_drop'])&support&(weak>=P['weak_raw_contrast'])&(strong>=P['strong_raw_contrast'])
        dark[:shoulder]=False;dark[-shoulder:]=False
        width_kernel=max(2,int(np.ceil(P['minimum_gap_width_pitch']*p)))
        opened=cv2.morphologyEx(dark.astype(np.uint8),cv2.MORPH_OPEN,np.ones((width_kernel,1),np.uint8)).astype(bool)
        score=np.maximum(0,-z)*np.clip(weak/2,0,1)
        boundaries=[];active=[];point_count=0;current_peak_columns=[]
        for k,x in enumerate(xs):
            peaks=[]
            for a,b in runs(opened[:,k]):
                values=score[a:b,k];j=int(a+np.argmax(values));weight=np.maximum(values,0)
                y=float(np.dot(np.arange(a,b),weight)/max(weight.sum(),1e-9)) if weight.sum()>0 else float(j)
                peaks.append((float(score[j,k]),y))
            chosen=[]
            for q in sorted(peaks,reverse=True):
                if not any(abs(q[1]-v[1])<P['minimum_peak_spacing_pitch']*p for v in chosen):chosen.append(q)
            chosen.sort(key=lambda q:q[1]);point_count+=len(chosen);current_peak_columns.append(len(chosen))
            active=[i for i in active if k-boundaries[i][-1][0]<=P['maximum_gap_columns']+1]
            edges=[]
            if active and chosen:
                pred=[]
                for i in active:
                    q=boundaries[i];tail=q[-5:];slope=0.
                    if len(tail)>1:
                        slope=float(np.median([(b[1]-a[1])/(b[0]-a[0]) for a,b in zip(tail[:-1],tail[1:])]))
                    pred.append(q[-1][1]+np.clip(slope,-.5*p,.5*p)*(k-q[-1][0]))
                cost=abs(np.asarray(pred)[:,None]-np.asarray([q[1] for q in chosen])[None,:])
                ai,pi=np.where(cost<P['search_half_pitch']*p)
                edges=sorted((float(cost[a,j]),active[a],int(j)) for a,j in zip(ai,pi))
            used_i=set();used_j=set();matched=[]
            for _,i,j in edges:
                if i in used_i or j in used_j:continue
                if any((boundaries[i][-1][1]-boundaries[ii][-1][1])*(chosen[j][1]-chosen[jj][1])<0 for ii,jj in matched):continue
                matched.append((i,j));used_i.add(i);used_j.add(j)
            for i,j in matched:boundaries[i].append((k,chosen[j][1]))
            for j,q in enumerate(chosen):
                if j not in used_j:active.append(len(boundaries));boundaries.append([(k,q[1])])
        rows=[]
        for q in boundaries:
            if len(q)<4 or (q[-1][0]-q[0][0])*step*2<P['minimum_boundary_span_original_px']:continue
            ix=np.asarray([v[0] for v in q]);ys=np.asarray([v[1] for v in q],np.float32)
            row=np.full(len(xs),np.nan);observed=np.zeros(len(xs),bool)
            rr=np.arange(ix[0],ix[-1]+1);interpolated=np.interp(rr,ix,ys).astype(np.float32)
            smooth=cv2.GaussianBlur(interpolated[None,:],(0,0),P['boundary_smoothing_columns']).ravel()
            row[rr]=smooth+gy[0]+guide[rr];observed[ix]=True
            rows.append(dict(y=row,observed=observed))
        pairs={}
        for k in range(len(xs)):
            present=sorted((r['y'][k],i) for i,r in enumerate(rows) if np.isfinite(r['y'][k]))
            gaps=np.diff([v[0] for v in present])
            for j,((ya,a),(yb,b)) in enumerate(zip(present[:-1],present[1:])):
                gap=yb-ya
                nearby=gaps[max(0,j-3):j+4];plausible=nearby[(nearby>=.5*p)&(nearby<=1.6*p)]
                local=float(np.median(plausible)) if len(plausible)>=3 else p
                if not P['pair_spacing_pitch'][0]<=gap/local<=P['pair_spacing_pitch'][1]:continue
                if not .50*p<=gap<=1.7*p:continue
                cy=(ya+yb)/2
                if cy<0 or cy>=975 or self.mask[int(round(min(974,cy))),int(x:=xs[k])]<.99:continue
                pairs.setdefault((a,b),[]).append(k)
        tracks=[]
        for (a,b),kk in sorted(pairs.items()):
            kk=np.asarray(kk);cuts=np.flatnonzero(np.diff(kk)>1)+1
            for seq,ix in enumerate(np.split(kk,cuts)):
                if len(ix)<4 or (xs[ix[-1]]-xs[ix[0]])*2<P['minimum_center_span_original_px']:continue
                observed=rows[a]['observed'][ix]&rows[b]['observed'][ix]
                if observed.mean()<P['minimum_observed_pair_fraction']:continue
                upper=np.c_[xs[ix],rows[a]['y'][ix]];lower=np.c_[xs[ix],rows[b]['y'][ix]];pts=(upper+lower)*.5
                tangent=np.gradient(pts,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-6)
                width=np.maximum(1.,(lower[:,1]-upper[:,1])*abs(tangent[:,0]))
                states=np.where(observed,'observed','inferred_gap').tolist();tid=f'current_pair_{a}_{b}_{seq}'
                tracks.append(dict(track_id=tid,boundary_ids=[f'current_boundary_{a}',f'current_boundary_{b}'],
                    points_xy=pts.tolist(),midpoint_xy=pts.tolist(),upper_boundary_xy=upper.tolist(),lower_boundary_xy=lower.tolist(),
                    width_px=width.tolist(),spacing_px=width.tolist(),vertical_spacing_px=(lower[:,1]-upper[:,1]).tolist(),
                    f3_pair_supported=[True]*len(ix),f3_pair_observed=observed.tolist(),f3_pair_inferred=(~observed).tolist(),
                    upper_boundary_inferred=(~rows[a]['observed'][ix]).tolist(),lower_boundary_inferred=(~rows[b]['observed'][ix]).tolist(),
                    point_status=states,segment_status=['observed' if x and y else 'inferred_gap' for x,y in zip(observed[:-1],observed[1:])],
                    refine_offset_px=[0.]*len(ix),length_px=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()),
                    width_is_calibrated_rope_diameter=False,identity='unknown',same_material_identity_confirmed=False,
                    source_boundary_lineage_ids=[f'current_boundary_{a}',f'current_boundary_{b}'],
                    current_frame_boundaries=True,attached_center_is_exact_current_boundary_midpoint=True))
        diagnostic=dict(status='detected' if tracks else 'no_supported_current_centers',fresh_detection=True,
            source='current_frame_LCN_grooves_and_adjacent_boundary_midpoints',calls=self.calls,
            observed_points=point_count,boundary_rows=len(rows),center_tracks=len(tracks),
            historical_boundary_coordinates_used=False,old_center_fallback=False,
            original_pixel_pitch_prior=2*p,parameters=P,seconds=time.perf_counter()-tick)
        return tracks,diagnostic

