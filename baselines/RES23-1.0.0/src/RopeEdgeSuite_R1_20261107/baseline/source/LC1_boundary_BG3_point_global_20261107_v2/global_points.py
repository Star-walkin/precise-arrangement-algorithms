"""BG3: global ordered classes of raw yellow points -> one continuous model/class.

No fitted-fragment input, no expiration of a class at an image evidence gap.
Models span their extreme retained observations; no displayed extrapolation.
"""
import math
import numpy as np
import cv2
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix,hstack,vstack,diags
from scipy.sparse.linalg import lsmr
from scipy.optimize import minimize,LinearConstraint

PARAMETERS=dict(version='BG3-point-global-v2',shape_knot_pitch=8.,row_knot_pitch=10.,
    assignment_radius_pitch=.42,assignment_ambiguity_pitch=.10,
    minimum_class_points=16,minimum_class_span_pitch=6.,fit_residual_q90_pitch=.28,
    minimum_curve_separation_pitch=.45,minimum_image_support_fraction=.35,
    maximum_count_point_gap_pitch=6.,maximum_inferred_count_point_gap_pitch=2.,
    iterations=5,display_extrapolation=False)

def basis(x,lo,hi,step):
    n=max(1,int(math.ceil((hi-lo)/step)))
    knots=np.r_[np.repeat(lo,4),np.linspace(lo,hi,n+1)[1:-1],np.repeat(hi,4)]
    return BSpline.design_matrix(np.clip(x,lo,hi),knots,3).toarray(),knots

def shared_shape(points,pitch):
    x=points[:,1]/pitch;y=points[:,2]/pitch
    tids=points[:,7].astype(int);use=[]
    for tid in np.unique(tids):
        ids=np.flatnonzero(tids==tid)
        if len(ids)>=8 and np.ptp(x[ids])>=3:use.extend(ids)
    use=np.asarray(use,int)
    if len(use)<40:raise ValueError('insufficient_shared_shape_points')
    xx=x[use];yy=y[use];track=np.unique(tids[use],return_inverse=True)[1]
    B,knots=basis(xx,float(x.min()),float(x.max()),PARAMETERS['shape_knot_pitch'])
    anchor=float(np.median(xx));b0=BSpline.design_matrix([anchor],knots,3).toarray()[0]
    B-=b0;nb=B.shape[1];nt=int(track.max())+1
    T=csr_matrix((np.ones(len(use)),(np.arange(len(use)),track)),shape=(len(use),nt))
    A=hstack([csr_matrix(B),T],format='csr')
    D=np.diff(np.eye(nb),n=2,axis=0)
    penalty=hstack([csr_matrix(D*.8),csr_matrix((len(D),nt))],format='csr')
    base=np.clip(points[use,3],.15,1.);w=base.copy();coef=np.zeros(nb+nt)
    for _ in range(4):
        Aw=diags(np.sqrt(w))@A
        coef=lsmr(vstack([Aw,penalty]),np.r_[yy*np.sqrt(w),np.zeros(len(D))],
                  atol=1e-7,btol=1e-7,maxiter=500)[0]
        residual=yy-A@coef
        w=base*np.minimum(1.,.12/np.maximum(np.abs(residual),1e-6))
    # Perspective changes the visible pitch across x. A translation-only
    # flattening can switch to the next rope even when its curves look smooth.
    # Jointly estimate y = center(x) + track_offset * pitch_scale(x).
    Bfull=B+b0;Bc=B
    offsets=coef[nb:]-np.median(coef[nb:]);w=base.copy()
    penalty2=np.zeros((2*len(D)+nb,2*nb))
    penalty2[:len(D),:nb]=D*.8
    penalty2[len(D):2*len(D),nb:]=D*30.
    penalty2[2*len(D):,nb:]=np.eye(nb)*.5
    for _ in range(8):
        design=np.c_[Bfull,Bc*offsets[track,None]]
        mat=design.T@(design*w[:,None])+penalty2.T@penalty2+np.eye(2*nb)*1e-7
        cc=np.linalg.solve(mat,design.T@(w*(yy-offsets[track])))
        center=Bfull@cc[:nb];scale=1+Bc@cc[nb:]
        scale=np.clip(scale,.65,1.5)
        for k in range(nt):
            ii=track==k;offsets[k]=np.median((yy[ii]-center[ii])/scale[ii])
        residual=yy-center-scale*offsets[track]
        w=base*np.minimum(1.,.10/np.maximum(np.abs(residual),1e-6))
    spline=BSpline(knots,cc[:nb],3);stretch=BSpline(knots,cc[nb:],3)
    stretch0=float(stretch(anchor))
    def predict(v):return spline(np.clip(np.asarray(v)/pitch,x.min(),x.max()))*pitch
    def scale_fn(v):return np.clip(1+stretch(np.clip(np.asarray(v)/pitch,x.min(),x.max()))-stretch0,.65,1.5)
    predict.scale=scale_fn
    return predict,dict(knots_processing=(knots*pitch).tolist(),coefficients=(cc[:nb]*pitch).tolist(),
        stretch_coefficients=cc[nb:].tolist(),stretch_anchor=stretch0,
        pitch_scale_range=[float(np.min(scale)),float(np.max(scale))],
        anchor_x_processing=anchor*pitch,
        training_point_count=len(use),training_local_track_count=nt,
        residual_q90_pitch=float(np.percentile(np.abs(residual),90)),
        local_track_ids_used_only_as_nuisance_intercepts=True)

class Row:
    def __init__(self,center,shape,pitch):
        self.center=float(center);self.shape=shape;self.pitch=pitch;self.spline=None
        self.lo=None;self.hi=None;self.ids=np.empty(0,int);self.stats={}
    def predict(self,x):
        x=np.asarray(x,dtype=float)
        if self.spline is None:return self.shape(x)+self.shape.scale(x)*self.center
        z=np.clip(x/self.pitch,self.lo,self.hi)
        # Outside data is only a temporary association hypothesis. The final
        # returned curve is clipped to observed endpoints, never extrapolated.
        return self.shape(x)+self.shape.scale(x)*self.spline(z)*self.pitch
    def fit(self,points,ids):
        p=self.pitch;xy=points[ids,1:3];u=xy[:,0]/p
        residual=(xy[:,1]-self.shape(xy[:,0]))/self.shape.scale(xy[:,0])/p
        self.lo=float(u.min());self.hi=float(u.max())
        B,knots=basis(u,self.lo,self.hi,PARAMETERS['row_knot_pitch'])
        D=np.diff(np.eye(B.shape[1]),n=2,axis=0)
        base=np.clip(points[ids,3],.15,1.);w=base.copy()
        for _ in range(4):
            M=B.T@(B*w[:,None])+8.*D.T@D+np.eye(B.shape[1])*1e-7
            c=np.linalg.solve(M,B.T@(w*residual))
            rr=residual-B@c;w=base*np.minimum(1.,.08/np.maximum(np.abs(rr),1e-6))
        self.spline=BSpline(knots,c,3);self.ids=ids
        self.stats=dict(residual_q90_pitch=float(np.percentile(np.abs(rr),90)),
            residual_median_pitch=float(np.median(np.abs(rr))),point_count=len(ids),
            span_pitch=self.hi-self.lo)
        self.model=dict(residual_knots_processing=(knots*p).tolist(),
            residual_coefficients_processing=(c*p).tolist(),degree=3)

    def constrain(self,points,neighbors):
        """Small constrained refit; do not translate curves or merge rope rows."""
        p=self.pitch;ids=self.ids;x=points[ids,1]
        B=BSpline.design_matrix(x/p,self.spline.t,3).toarray()
        y=(points[ids,2]-self.shape(x))/self.shape.scale(x)/p
        D=np.diff(np.eye(B.shape[1]),n=2,axis=0)
        res=y-B@self.spline.c
        w=np.clip(points[ids,3],.15,1.)*np.minimum(1.,.08/np.maximum(np.abs(res),1e-6))
        H=B.T@(B*w[:,None])+8.*D.T@D+np.eye(B.shape[1])*1e-7;g=B.T@(w*y)
        matrices=[];lower=[];upper=[]
        for other in neighbors:
            xy=other['xy'];lo=max(x.min(),xy[0,0]);hi=min(x.max(),xy[-1,0])
            if hi<lo:continue
            xs=np.arange(math.ceil(lo),math.floor(hi)+1)
            predicted=self.predict(xs);target=np.interp(xs,xy[:,0],xy[:,1])
            if np.min(np.abs(predicted-target))>.8*p:continue
            C=BSpline.design_matrix(xs/p,self.spline.t,3).toarray()*self.shape.scale(xs)[:,None]
            rhs=(target-self.shape(xs))/p
            gap=PARAMETERS['minimum_curve_separation_pitch']*self.shape.scale(xs)+.002
            matrices.append(C)
            if np.median(predicted-target)<0:
                lower.extend(np.full(len(xs),-np.inf));upper.extend(rhs-gap)
            else:lower.extend(rhs+gap);upper.extend(np.full(len(xs),np.inf))
        if not matrices:return False
        prior=self.spline.c.copy()
        result=minimize(lambda z:.5*z@H@z-g@z,prior,jac=lambda z:H@z-g,
            constraints=[LinearConstraint(np.vstack(matrices),np.asarray(lower),np.asarray(upper))],
            method='SLSQP',options=dict(maxiter=60,ftol=1e-9))
        if not result.success:return False
        allx=np.arange(math.ceil(x.min()),math.floor(x.max())+1)
        prior_y=self.predict(allx)
        self.spline=BSpline(self.spline.t,result.x,3)
        change=float(np.max(np.abs(self.predict(allx)-prior_y)))/p
        error=np.abs(points[ids,2]-self.predict(x))/p
        if change>.20 or np.percentile(error,90)>PARAMETERS['fit_residual_q90_pitch']:
            self.spline=BSpline(self.spline.t,prior,3);return False
        self.stats.update(constraint_adjustment_max_pitch=change,residual_q90_pitch=float(np.percentile(error,90)))
        self.model['residual_coefficients_processing']=(result.x*p).tolist()
        return True

def initial_rows(points,shape,p):
    xs,counts=np.unique(points[:,1],return_counts=True)
    middle=np.median(xs)
    anchor=float(xs[np.argmax(counts-.02*np.abs(xs-middle)/p)])
    take=np.abs(points[:,1]-anchor)<=2*p
    z=(points[take,2]-shape(points[take,1]))/shape.scale(points[take,1]);step=.05*p
    bins=np.arange(z.min()-p,z.max()+p+step,step)
    hist,_=np.histogram(z,bins,weights=np.clip(points[take,3],.15,1.))
    sm=gaussian_filter1d(hist.astype(float),2.)
    peaks,_=find_peaks(sm,distance=max(2,round(.55*p/step)),
                       prominence=max(.3,float(np.max(sm))*.08))
    centers=(bins[peaks]+bins[peaks+1])/2
    return [Row(c,shape,p) for c in centers],dict(anchor_x=anchor,initial_class_count=len(centers))

def assign(points,rows,p):
    if not rows:return np.full(len(points),-1,int),np.zeros(len(points),bool)
    pred=np.vstack([r.predict(points[:,1]) for r in rows]).T
    costs=np.abs(points[:,2,None]-pred)
    order=np.argsort(costs,axis=1);labels=order[:,0]
    best=costs[np.arange(len(points)),labels]
    ambiguous=(costs[np.arange(len(points)),order[:,1]]-best<PARAMETERS['assignment_ambiguity_pitch']*p) if len(rows)>1 else np.zeros(len(points),bool)
    labels[(best>PARAMETERS['assignment_radius_pitch']*p)|ambiguous]=-1
    # At most one original point per class and image column. Membership is on
    # raw point IDs, not on fitted samples or copied historical curve IDs.
    for x in np.unique(points[:,1]):
        ii=np.flatnonzero((points[:,1]==x)&(labels>=0))
        for k in np.unique(labels[ii]):
            jj=ii[labels[ii]==k]
            if len(jj)>1:
                keep=jj[np.argmin(best[jj]/p-.1*points[jj,3])]
                labels[jj[jj!=keep]]=-1
    return labels,ambiguous

def sample(a,xy,x0=0,y0=0):
    return cv2.remap(a.astype(np.float32,copy=False),(xy[:,0]-x0).astype(np.float32)[None],
        (xy[:,1]-y0).astype(np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)[0]

def supported_path(data,xy):
    inside=sample(data['surface'],xy)>254
    unknown=sample(data['exclusion'],xy)>0
    supported=(sample(data['broad'],xy,int(data['x0']),int(data['y0']))>=.1)&(
        sample(data['clean'],xy,int(data['x0']),int(data['y0']))>=127)&(
        sample(data['body'],xy,int(data['x0']),int(data['y0']))<=.65)&inside&~unknown
    return inside,unknown,supported

def model_envelope(surface,top_curve=None):
    """Display/model domain only: internal holes stay UNKNOWN for measurement.

    Do not expand observed side extents or go above the current top interface.
    The original surface/exclusion arrays are never changed.
    """
    envelope=surface.copy()
    for y,row in enumerate(surface):
        xs=np.flatnonzero(row)
        if len(xs):envelope[y,xs[0]:xs[-1]+1]=255
    if top_curve is not None and len(top_curve):
        q=np.asarray(top_curve)
        for x in range(envelope.shape[1]):
            if x<q[0,0] or x>q[-1,0]:envelope[:,x]=0
            else:envelope[:max(0,math.ceil(float(np.interp(x,q[:,0],q[:,1])))),x]=0
    return envelope

def group_and_fit(data,d):
    points=data['points'];p=float(data['pitch'])
    shape,shape_diag=shared_shape(points,p)
    rows,init_diag=initial_rows(points,shape,p)
    steps=[]
    for iteration in range(PARAMETERS['iterations']):
        labels,ambiguous=assign(points,rows,p);kept=[]
        for k,row in enumerate(rows):
            ids=np.flatnonzero(labels==k)
            if len(ids)<PARAMETERS['minimum_class_points'] or np.ptp(points[ids,1])<PARAMETERS['minimum_class_span_pitch']*p:continue
            row.fit(points,ids);kept.append(row)
        steps.append(dict(iteration=iteration,classes=len(kept),assigned_points=int(np.sum(labels>=0)),
                          ambiguous_points=int(np.sum(ambiguous))))
        rows=kept
    # A final re-assignment makes saved point memberships the actual fit input.
    labels,ambiguous=assign(points,rows,p);candidates=[];rejections=[]
    for k,row in enumerate(rows):
        ids=np.flatnonzero(labels==k)
        if len(ids)<PARAMETERS['minimum_class_points'] or np.ptp(points[ids,1])<PARAMETERS['minimum_class_span_pitch']*p:continue
        row.fit(points,ids)
        residual=np.abs(points[ids,2]-row.predict(points[ids,1]))
        inliers=ids[residual<=.35*p]
        if len(inliers)<PARAMETERS['minimum_class_points'] or np.ptp(points[inliers,1])<PARAMETERS['minimum_class_span_pitch']*p:continue
        row.fit(points,inliers);ids=inliers
        xs=np.arange(math.ceil(points[ids,1].min()),math.floor(points[ids,1].max())+1)
        xy=np.c_[xs,row.predict(xs)]
        inside,unknown,support=supported_path(data,xy)
        why=[]
        model_inside=sample(data.get('model_envelope',data['surface']),xy)>254
        if not np.all(model_inside):why.append('whole_class_model_leaves_geometric_envelope')
        if row.stats['residual_q90_pitch']>PARAMETERS['fit_residual_q90_pitch']:why.append('whole_class_residual_too_large')
        if np.mean(support)<PARAMETERS['minimum_image_support_fraction']:why.append('whole_class_image_support_too_low')
        if why:
            rejections.append(dict(point_ids=points[ids,0].astype(int).tolist(),reasons=why,
                outside_fraction=float(np.mean(~model_inside)),outside_x=xs[~model_inside].tolist(),
                domain=[float(xs[0]),float(xs[-1])],y_at_d=float(row.predict([d])[0])));continue
        candidates.append(dict(xy=xy,point_states=np.where(support,'observed','inferred').tolist(),
            point_ids=points[ids,0].astype(int).tolist(),point_indices=ids,
            fit_statistics=row.stats,model=row.model,support_fraction=float(np.mean(support)),
            unknown_fraction=float(np.mean(unknown)),rank_y=float(row.predict([d])[0]),
            outside_validated_surface_fraction=float(np.mean(~inside)),
            _row=row,
            weight=float(len(ids)*np.mean(points[ids,3]))))
    # Whole-class acceptance: never split a fitted class into disconnected curves.
    kept=[]
    for c in sorted(candidates,key=lambda c:-c['weight']):
        xy=c['xy'];conflicts=[]
        if kept and c['_row'].constrain(points,kept):
            xy=np.c_[xy[:,0],c['_row'].predict(xy[:,0])];c['xy']=xy
            inside,unknown,support=supported_path(data,xy)
            if not np.all(sample(data.get('model_envelope',data['surface']),xy)>254) or np.mean(support)<PARAMETERS['minimum_image_support_fraction']:
                rejections.append(dict(point_ids=c['point_ids'],reasons=['constrained_model_outside_envelope_or_weak_support'],
                    y_at_d=float(c['_row'].predict([d])[0]),domain=[float(xy[0,0]),float(xy[-1,0])]))
                continue
            c.update(point_states=np.where(support,'observed','inferred').tolist(),
                support_fraction=float(np.mean(support)),unknown_fraction=float(np.mean(unknown)),
                outside_validated_surface_fraction=float(np.mean(~inside)))
        for q in kept:
            z=q['xy'];lo=max(xy[0,0],z[0,0]);hi=min(xy[-1,0],z[-1,0])
            if hi<lo:continue
            xs=np.arange(math.ceil(lo),math.floor(hi)+1)
            delta=np.interp(xs,xy[:,0],xy[:,1])-np.interp(xs,z[:,0],z[:,1])
            min_gap=PARAMETERS['minimum_curve_separation_pitch']*p*shape.scale(xs)
            if np.any(np.abs(delta)<min_gap) or np.min(delta)<0<np.max(delta):
                conflicts.append(dict(other_point_id=q['point_ids'][0],min_distance=float(np.min(np.abs(delta))),
                    min_x=float(xs[np.argmin(np.abs(delta))]),domain=[float(lo),float(hi)],crossed=bool(np.min(delta)<0<np.max(delta))))
        if conflicts:rejections.append(dict(point_ids=c['point_ids'],reasons=['whole_class_order_or_spacing_conflict'],conflicts=conflicts,
            y_at_d=c['rank_y'],domain=[float(xy[0,0]),float(xy[-1,0])]))
        else:kept.append(c)
    kept.sort(key=lambda c:c['rank_y'])
    membership=np.full(len(points),-1,int)
    for k,c in enumerate(kept):
        c['boundary_id']=f'BG3_b{k:03d}';membership[c.pop('point_indices')]=k
        c.pop('weight');c.pop('rank_y');c.pop('_row')
    return kept,dict(parameters=PARAMETERS,shared_shape=shape_diag,initialization=init_diag,iterations=steps,
        raw_point_count=len(points),class_count=len(kept),classified_point_count=int(np.sum(membership>=0)),
        unclassified_point_count=int(np.sum(membership<0)),point_membership=membership,
        rejections=rejections,curve_splits=0,curve_id_scope='current_frame_visual_groove',
        geometry_coordinates='processing_grid'),points

def intersections(curves,points,data,d):
    p=float(data['pitch']);out=[];rejected=[]
    by_id={int(pt[0]):pt for pt in points}
    for c in curves:
        xy=c['xy']
        if not xy[0,0]<=d<=xy[-1,0]:continue
        y=float(np.interp(d,xy[:,0],xy[:,1]));obs=np.asarray([by_id[i][1:3] for i in c['point_ids']])
        left=obs[obs[:,0]<=d];right=obs[obs[:,0]>=d];why=[]
        inside,unknown,support=supported_path(data,np.asarray([[d,y]]))
        span=None
        if not len(left) or not len(right):why.append('no_bilateral_point_support')
        else:
            span=float(right[:,0].min()-left[:,0].max())
            limit=PARAMETERS['maximum_count_point_gap_pitch'] if support[0] else PARAMETERS['maximum_inferred_count_point_gap_pitch']
            if span>limit*p:why.append('class_point_gap_at_d_too_long')
        if not inside[0] or unknown[0]:why.append('count_intersection_unknown_or_excluded')
        record=dict(boundary_id=c['boundary_id'],track_id=c['boundary_id'],y=y,
            status='image_supported_fitted_curve' if support[0] else 'point_bracketed_model_interpolation',
            point_bracket_gap_px=span,score=c['support_fraction'])
        if why:rejected.append(dict(**record,reasons=why))
        else:out.append(record)
    out.sort(key=lambda c:c['y'])
    return out,rejected
