"""Read-only algorithm diagnosis on three explicitly requested open cases.

Alternate width gates below are diagnostic counterfactual counts only, never
the production candidate module or saved formal predictions.
"""
from pathlib import Path
import json,time
import cv2,numpy as np
import whole_io as io
import lane_candidates as lc

def main():
    out=io.HERE/'lane_pool_diagnosis_001'
    if out.exists():raise ValueError('No overwrite')
    out.mkdir();cv2.setNumThreads(1);rows=io.selection('all')
    names=['L_run_20260923_01__f000015__left','ruler_left_05','ruler_right_05'];results=[]
    for name in names:
        row=next(r for r in rows if Path(r['output_file']).stem==name);rel=Path(row['output_file']);data=io.inputs(row)
        case=io.read_json(io.HERE/'pilot_001/记录'/rel.parent.name/(name+'.json'));free=case['free'];g=free['geometry'];o=np.asarray(g['origin_xy']);d=np.asarray(g['direction_xy']);w=float(g['width_px']);p=float(data['pitch'])
        tracks,dist,cache,counts=lc._snap_refs(data,o,d,w);rs=lc._merge_rows(tracks,p);lanes,reject=lc._lane_pairs(rs,data,cache,w)
        baseline={'lanes':len(lanes),'join_options':sum(len(lc._join_options(l,o,d,w,p)) for l in lanes),'pairs':[]}
        for l in lanes:
            opts=lc._join_options(l,o,d,w,p)
            baseline['pairs'].append({'id':l['lane_id'],'xspan':[float(l['x'][0]),float(l['x'][-1])],'width_median':float(np.median(l['widths'])),'width_free_ratio':float(np.median(l['widths'])/w),'join_x':[float(l['x'][a['index']]) for a in opts]})
        old=dict(lc.PARAMETERS)
        lc.PARAMETERS['minimum_normal_width_ratio']=0.;lc.PARAMETERS['maximum_normal_width_ratio']=100.
        pitch_only,_=lc._lane_pairs(rs,data,cache,w)
        lc.PARAMETERS.clear();lc.PARAMETERS.update(old)
        # Print candidate lane gaps when the independently measured free optical width
        # is suppressed, while pitch/order/formal-neighbor gates are unchanged.
        counter={'mode':'diagnostic only: remove free-width ratio, retain pitch/order/formal adjacency','lanes':len(pitch_only),'join_options':sum(len(lc._join_options(l,o,d,w,p)) for l in pitch_only),'median_lane_widths':[float(np.median(l['widths'])) for l in pitch_only],'ratio_quantiles':np.percentile([np.median(l['widths'])/w for l in pitch_only],[0,25,50,75,100]).tolist() if pitch_only else None}
        yy,xx=np.nonzero(data['merged']);q=np.c_[xx,yy];n=np.array([-d[1],d[0]]);s=(q-o)@d;delta=abs((q-o)@n)
        near=(s>=0)&(delta<6*p)
        saved=case['whole']['candidates'];straight=next((c for c in saved if c['lane_id'] is None),None)
        end_s=float((np.asarray(straight['xy'][-1])-o)@d) if straight else None
        late=near&(s>=end_s-4*w) if end_s is not None else np.zeros(len(s),bool)
        formal={'near_forward_mask_pixels':int(near.sum()),'near_s_range':[float(s[near].min()),float(s[near].max())] if near.any() else None,'straight_last_s':end_s,'formal_pixels_within6pitch_of_axis_after_last_straight_minus4width':int(late.sum()),'late_x_range':[int(xx[late].min()),int(xx[late].max())] if late.any() else None}
        # A map reveals whether late geometry truly exists in accepted F3 or only
        # in excluded/unfused references. It is not a new prediction or truth mask.
        view=data['original'].copy();view[data['merged']>0]=(0,230,230)
        if near.any():view[yy[near],xx[near]]=(0,0,255)
        for lane in lanes:
            z=np.rint(np.c_[lane['x'],lane['y']]).astype(np.int32);cv2.polylines(view,[z],False,(255,150,0),1)
        if straight:
            z=np.rint(straight['xy']).astype(np.int32);cv2.polylines(view,[z],False,(0,255,0),2)
        io.write_image(out/(name+'_formal.jpg'),cv2.resize(view,(2048,1500)))
        results.append({'input':row['output_file'],'pitch_px':p,'free_width_px':w,'free_width_pitch_ratio':w/p,'formal':formal,'reference_counts':counts,'production_geometry':baseline,'counterfactual_counts':counter,'saved_selected':case['whole']['selected_candidate_id']})
        print(name,'pitch',round(p,2),'width',round(w,2),'lanes',len(lanes),'pitch-only',len(pitch_only),'lateformal',int(late.sum()),flush=True)
    io.dump(out/'diagnosis.json',{'scope':'Three requested cases, saved pilot001 plus pure geometry counterfactual counts; no formal prediction changed.','source_sha256':io.sha(io.HERE/'lane_candidates.py'),'cases':results})
    return 0
if __name__=='__main__':raise SystemExit(main())
