"""Read frozen H5 two counterexamples; geometric crossings are not contact truth."""
import gzip,hashlib,json
from pathlib import Path
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
H5=HERE.parent.parent/'postprocess_b_20261031_v5'
IDS=['dual_20260923_101517__pair0345__right','dual_20260923_101643__pair0255__right']
F3=ROOT/'数据资产/05_派生与实验结果/camera_condition_full_surface_joint_20261028_v3/all_001'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_COLOR)
def save(p,img):cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,95])[1].tofile(str(p))
def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]

def intersections(path,stage,lines,idkey):
    """Finite-segment intersections; collinear/tangent overlap deliberately excluded."""
    p=np.asarray(path,float);r=np.diff(p,axis=0);progress=np.r_[0,np.cumsum(np.linalg.norm(r,axis=1))]
    hits=[]
    for line in lines:
        q=np.asarray(line['points_xy'],float)
        if len(q)<2:continue
        s=np.diff(q,axis=0);den=cross(r[:,None,:],s[None,:,:]);delta=q[None,:-1,:]-p[:-1,None,:]
        valid=np.abs(den)>1e-6*(np.linalg.norm(r,axis=1)[:,None]*np.linalg.norm(s,axis=1)[None,:])
        t=np.divide(cross(delta,s[None,:,:]),den,out=np.full_like(den,np.nan),where=valid)
        u=np.divide(cross(delta,r[:,None,:]),den,out=np.full_like(den,np.nan),where=valid)
        ii,jj=np.where(valid&(t>=0)&(t<1)&(u>=0)&(u<1))
        seen=[]
        for i,j in zip(ii,jj):
            pt=p[i]+t[i,j]*r[i]
            if any(np.linalg.norm(pt-v)<.25 for v in seen):continue
            seen.append(pt)
            angle=float(np.degrees(np.arccos(np.clip(abs(r[i]@s[j])/(np.linalg.norm(r[i])*np.linalg.norm(s[j])),-1,1))))
            row={'line_id':line[idkey],'stage':stage[min(i+1,len(stage)-1)],'xy_original':(pt*2+[0,1050]).tolist(),'progress_original_px':float((progress[i]+t[i,j]*np.linalg.norm(r[i]))*2),'crossing_angle_degrees':angle}
            if 'inferred' in line:row['boundary_inferred_at_crossing']=bool(line['inferred'][j] or line['inferred'][j+1])
            hits.append(row)
    return sorted(hits,key=lambda x:x['progress_original_px'])

def main():
    inputs=json.loads((H5/'inputs.json').read_text(encoding='utf-8'));jobs={j['id']:j for j in inputs['jobs']}
    result=[]
    for ident in IDS:
        job=jobs[ident];assert '20260925' not in job['image_path']
        rp=next((H5/'all_001').rglob(ident+'.json.gz'))
        z=json.load(gzip.open(rp,'rt',encoding='utf-8'));entry=z['entries'][z['selected_entry']]
        image=read(job['image_path']);assert sha(job['image_path'])==job['image_sha256']
        f3path=F3/'图像结果/02_F3融合边界/right_camera'/(ident+'.png');f3=read(f3path)
        surf=z['surface_tracks'];rows=[]
        for ci,c in enumerate(entry['connection']['candidates']):
            p=np.asarray(c['points_xy'],float);stage=c['stage'];lens=np.linalg.norm(np.diff(p,axis=0),axis=1)*2
            lengths={s:float(lens[np.asarray(stage[1:])==s].sum()) for s in set(stage)}
            centers=intersections(p,stage,[t for t in surf['tracks'] if t['track_id']!=c['track_id']],'track_id');boundaries=intersections(p,stage,surf['boundaries'],'boundary_id')
            rows.append({'index':ci,'selected':ci==z['selected_candidate'],'kind':c['kind'],'track_id':c['track_id'],'stage_lengths_original_px':lengths,'center_track_crossings':centers,'F3_boundary_polyline_crossings':boundaries,'metrics':c['metrics'],'crossing_counts_by_stage':{s:{'gray_center_track_intersections':sum(h['stage']==s for h in centers),'F3_boundary_intersections':sum(h['stage']==s for h in boundaries)} for s in ['free','transition','surface']},'physical_contact_confirmed':False})
        crop=image[1900:2850,1000:3300].copy();fcrop=f3[1900:2850,1000:3300].copy();over=crop.copy()
        for t in surf['tracks']:
            pts=np.rint(np.asarray(t['points_xy'])*2+[0,1050]-[1000,1900]).astype(np.int32)
            cv2.polylines(over,[pts],False,(130,130,130),1)
        for ci,c in enumerate(entry['connection']['candidates']):
            pts=np.rint(np.asarray(c['points_xy'])*2+[0,1050]-[1000,1900]).astype(np.int32)
            cv2.polylines(over,[pts],False,[(0,255,0),(255,0,255),(255,255,0)][ci%3],2)
        canvas=np.full((1000,2300*3,3),245,np.uint8)
        for i,(im,label) in enumerate([(crop,'Original: fixed crop x1000:3300 y1900:2850'),(fcrop,'Frozen F3 original visualization; excluded free corridor'),(over,'Gray centers; green selected free; other candidates cyan/magenta')]):
            canvas[50:,i*2300:(i+1)*2300]=im
            cv2.putText(canvas,label,(i*2300+12,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(20,20,20),2)
        vp=HERE/(ident+'__original_F3_H5_crop.jpg');save(vp,canvas)
        row={'id':ident,'source_group':job['source_group'],'image_sha256':job['image_sha256'],'H5_record_sha256':sha(rp),'F3_visual_path':str(f3path),'F3_visual_sha256':sha(f3path),'diagnostic_visual':str(vp),'selected_entry':z['selected_entry'],'selected_candidate':z['selected_candidate'],'candidate_details':rows,'rejected':entry['connection']['rejected'],'boundary_count':len(surf['boundaries']),'gray_center_track_count':len(surf['tracks']),'source_dependencies':z['input_dependency_sha256'],'semantic_limit':'Finite image-plane intersections only; free-stage crossings do not prove physical contact, same-rope identity, or an incorrect path. Missing F3 under an excluded active-rope corridor is unknown, not no boundary.'}
        result.append(row)
        print(ident,[(r['kind'],r['crossing_counts_by_stage']) for r in rows])
    out={'source':'Frozen H5 only; no inference, labels, reserved set or devices','coordinate_system':'Original image pixels in output; H5 source half crop xy -> 2*xy+[0,1050]','intersection_rule':'Strict finite polyline intersections, no extrapolation, parallel overlaps excluded; duplicate crossings within 0.5 original px per line merged. IDs are fragments, counts are not rope counts.','items':result,'script_sha256':sha(__file__)}
    (HERE/'H5_two_counterexamples_diagnosis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
