"""Read saved pilot candidates only; no prediction changes or image scoring."""
from pathlib import Path
import json,math,hashlib,sys
import numpy as np
import whole_io as io

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def tangent_fit(q,radius):
    q=np.asarray(q,float);q=q[np.argsort(q[:,0])];q=q[np.r_[True,np.diff(q[:,0])>1e-6]]
    angles=[];second=[]
    for x in q[:,0]:
        use=abs(q[:,0]-x)<=radius
        if np.count_nonzero(use)<5:
            use=np.zeros(len(q),bool);use[np.argsort(abs(q[:,0]-x))[:min(5,len(q))]]=True
        p=np.polyfit(q[use,0]-x,q[use,1],min(2,np.count_nonzero(use)-1))
        angles.append(math.degrees(math.atan(p[-2])));second.append(2*p[0] if len(p)==3 else 0.)
    a=np.asarray(angles);k=np.asarray(second)
    return {'x':q[:,0].tolist(),'angle_deg':a.tolist(),'d2y_dx2':k.tolist(),'max_reverse_angle_deg':float(np.max(np.maximum.accumulate(a)-a)),'negative_curvature_fraction':float(np.mean(k<-.0001)),'median_second_derivative':float(np.median(k)),'angle_start_deg':float(a[0]),'angle_end_deg':float(a[-1])}

def candidate(c,whole,free,pitch):
    xy=np.asarray(c['xy']);phase=np.asarray(c['phase']);ts=c['transition_start_index'];ji=c['join_index'];d=np.asarray(free['geometry']['direction_xy']);sign=np.sign(d[0]);fw=float(free['geometry']['width_px'])
    tr=xy[phase=='transition'];angle0=math.degrees(math.atan(d[1]/d[0]));result={'id':c['candidate_id'],'lane_id':c['lane_id'],'selected':c['candidate_id']==whole.get('selected_candidate_id'),'score':c['score'],'join_x':c['join_x'],'free_angle_xplus_deg':angle0,'join_forward_s':c.get('join_forward_s'),'geometry_cost':c['geometry_cost'],'premature_fraction':c['premature_turn_penalty_fraction']}
    if len(tr)<4:return result
    # Hermite transition is a cubic: fit its saved points to recover exact derivatives.
    xx=tr[:,0];mid=float(np.mean(xx));scale=float(np.ptp(xx));poly=np.polyfit((xx-mid)/scale,tr[:,1],3);axis=np.linspace(xx.min(),xx.max(),151);u=(axis-mid)/scale
    deriv=np.polyval(np.polyder(poly),u)/scale;second=np.polyval(np.polyder(poly,2),u)/scale**2;theta=np.degrees(np.arctan(deriv))
    result['transition']={'max_reverse_angle_deg':float(np.max(np.maximum.accumulate(theta)-theta)),'negative_curvature_fraction':float(np.mean(second<-.0001)),'median_second_derivative':float(np.median(second)),'angle_start_deg':float(theta[0]),'angle_end_deg':float(theta[-1]),'min_angle_deg':float(theta.min()),'max_angle_deg':float(theta.max()),'turn_total_deg':float(theta[-1]-theta[0]),'cubic_max_fit_error_px':float(np.max(abs(np.polyval(poly,(xx-mid)/scale)-tr[:,1])))}
    end=ji
    if ji<len(xy):
        length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy[ji:],axis=0),axis=1))]
        end=ji+int(np.searchsorted(length,2*pitch,side='right'))
    combined=xy[max(0,ts-2):max(end,ji+2)]
    if len(combined)>=5:result['transition_earlylane']=tangent_fit(combined,pitch*.75)
    lane=next((v for v in whole['lanes'] if v['lane_id']==c['lane_id']),None)
    if lane is not None:
        lp=np.asarray(lane['xy']);dist=abs(lp[:,0]-c['join_x']);near=dist<=3*pitch
        if np.count_nonzero(near)>=5:
            center=tangent_fit(lp[near],pitch*1.5);bounds=np.asarray(lane['boundary_xy']);rails=[tangent_fit(bounds[near,k],pitch*1.5) for k in range(2)]
            quad=[np.polyfit(v[:,0]-c['join_x'],v[:,1],2) for v in [lp[near],bounds[near,0],bounds[near,1]]]
            result['formal_neighbor_curvature']={'center_median_second_derivative':center['median_second_derivative'],'center_negative_fraction':center['negative_curvature_fraction'],'rail_median_second_derivatives':[v['median_second_derivative'] for v in rails],'rail_negative_fractions':[v['negative_curvature_fraction'] for v in rails],'whole_6pitch_quad_second_derivatives':[float(2*p[0]) for p in quad]}
        near=dist<=3*pitch
        if np.count_nonzero(near)>=5:
            poly2=np.polyfit(lp[near,0]-c['join_x'],lp[near,1],2);laneangle=math.degrees(math.atan(poly2[1]));result['join_lane_angle_fit_deg']=laneangle;result['incoming_to_lane_signed_turn_xplus_deg']=(laneangle-angle0)*sign
        neighbor=[]
        join_y=float(np.interp(c['join_x'],lp[:,0],lp[:,1]))
        for other in whole['lanes']:
            oq=np.asarray(other['xy']);use=abs(oq[:,0]-c['join_x'])<=3*pitch
            if np.count_nonzero(use)<5 or not oq[0,0]<=c['join_x']<=oq[-1,0]:continue
            if abs(np.interp(c['join_x'],oq[:,0],oq[:,1])-join_y)>3*pitch:continue
            pq=np.polyfit(oq[use,0]-c['join_x'],oq[use,1],2)
            neighbor.append({'lane_id':other['lane_id'],'d2y_dx2':float(2*pq[0])})
        result['nearby_formal_lanes_6pitch_quadratic']=neighbor
    return result

def main():
    run=io.HERE/'pilot_002';dest=io.HERE/'transition_curvature_diagnosis_002';dest.mkdir(exist_ok=False)
    protocol=load(run/'source_snapshot/protocol.json');allrows={v['output_file']:v for v in io.selection('all')};out=[]
    for inputid in protocol['pilot_inputs']:
        rel=Path(inputid);path=run/'记录'/rel.parent.name/(rel.stem+'.json');saved=load(path);row=allrows[inputid];c3=load(io.C3/'boundaries/records'/rel.parent.name/(rel.stem+'.json'))
        pitch=float(c3['record']['pitch_px'])
        whole=saved['whole'];out.append({'input':inputid,'record_sha256':io.sha(path),'pitch':pitch,'status':whole['status'],'candidates':[candidate(c,whole,saved['free'],pitch) for c in whole['candidates'] if c['lane_id'] is not None]})
    io.dump(dest/'diagnosis.json',{'scope':'Posthoc saved-candidate geometry diagnostic, not accuracy scoring; fixed pilot002 only','cases':out,'source_sha256':io.sha(__file__)})
    for v in out:
        if any(s in v['input'] for s in ('pair0315__right','ruler_right_05','f000015__left')):
            print(v['input'])
            for c in v['candidates']:
                t=c.get('transition',{});z=c.get('formal_neighbor_curvature',{})
                print(c['id'],'SELECTED' if c['selected'] else '', 'score',round(c['score'],3),'rev',round(t.get('max_reverse_angle_deg',0),2),'net',round(t.get('turn_total_deg',0),2),'quad',z.get('whole_6pitch_quad_second_derivatives'),'neighbor',[(v['lane_id'],round(v['d2y_dx2'],6)) for v in c.get('nearby_formal_lanes_6pitch_quadratic',[])])
    print('saved',str(dest/'diagnosis.json'))
if __name__=='__main__':main()
