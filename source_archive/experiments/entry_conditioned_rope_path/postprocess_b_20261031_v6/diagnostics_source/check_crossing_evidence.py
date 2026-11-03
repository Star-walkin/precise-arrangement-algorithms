"""Same recorded evidence/mask audit on two failures and two free-segment controls."""
from diagnose_h5_crossings import HERE,H5,F3,intersections,sha
import gzip,json
from pathlib import Path
import cv2
import numpy as np

IDS=['dual_20260923_101517__pair0345__right','dual_20260923_101643__pair0255__right','R_static_20260922_01__f000095__right','R_run_20260923_01__f000055__right']
def gray(p):return cv2.imdecode(np.fromfile(str(p),np.uint8),0)
def vals(im,p):
    q=np.rint(p).astype(int);good=(q[:,0]>=0)&(q[:,0]<im.shape[1])&(q[:,1]>=0)&(q[:,1]<im.shape[0]);out=np.zeros(len(q),bool);out[good]=im[q[good,1],q[good,0]]>0;return out
def main():
    rows=[]
    for ident in IDS:
        rp=next((H5/'all_001').rglob(ident+'.json.gz'));z=json.load(gzip.open(rp,'rt',encoding='utf-8'));f=z['entries'][z['selected_entry']]['free']
        p=np.asarray(f['points_xy'],float);xy=p*2+[0,1050];s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        masks={'F3_exclusion':F3/'记录/masks/exclusion/right_camera'/(ident+'.png')}
        for src in z['input_dependency_sha256']:
            if 'camera_condition_sam2_' in src:masks['SAM2_body']=Path(src)
        masks['C3_incoming_rope_occlusion']=F3.parents[1]/'camera_condition_grouped_conic_20261026_v6/all_001/boundaries/incoming_rope_occlusion/right_camera'/(ident+'.png')
        fields={k:np.asarray(f[k],bool) for k in ['supported','protect_support','raw_a7_supported','observed_straight_support']}
        fields.update({k:vals(gray(v),xy) for k,v in masks.items()})
        hits=intersections(p,['free']*len(p),z['surface_tracks']['boundaries'],'boundary_id')
        def detail(i):
            win=np.abs(s-s[i])<=60
            return {'point_index':i,'xy_original':xy[i].tolist(),'distance_from_entry_original_px':float(s[i]),'status':f['point_status'][i],'values':{k:bool(v[i]) for k,v in fields.items()},'recorded_profile':{k:float(f['evidence'][k][i]) for k in ['center','side_drop','raw_score','raw_contrast']},'fraction_within_60px_each_side':{k:float(v[win].mean()) for k,v in fields.items()}}
        enriched=[]
        for h in hits:
            i=int(np.argmin(np.linalg.norm(xy-h['xy_original'],axis=1)));enriched.append(dict(h,evidence=detail(i)))
        post=(s>=120)&fields['SAM2_body'];tail=s>=max(120,s[-1]-300)
        row={'id':ident,'selected_entry':z['selected_entry'],'free_span_original_px':float(s[-1]),'finite_F3_boundary_crossings':enriched,'last_point':detail(len(p)-1),'last300px_fractions':{k:float(v[tail].mean()) for k,v in fields.items()},'post_entry_inside_SAM2_fractions':{k:float(v[post].mean()) if post.any() else None for k,v in fields.items()},'mask_sources':{k:{'path':str(v),'sha256':sha(v)} for k,v in masks.items()},'limits':'Image-plane masks and recorded supports only; no contact/identity truth; F3 exclusion can also mean unknown texture and is never affirmative target evidence.'}
        rows.append(row)
        print(ident,'span',s[-1],'crossings',len(enriched),'tail',row['last300px_fractions'])
    out={'script_sha256':sha(__file__),'window_original_px':120,'tail_original_px':300,'items':rows,'labels_read':False,'new_inference':False}
    (HERE/'crossing_evidence_failures_controls.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
