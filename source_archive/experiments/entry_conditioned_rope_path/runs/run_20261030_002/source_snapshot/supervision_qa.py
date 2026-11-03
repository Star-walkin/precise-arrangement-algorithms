"""Post-freeze source-only QA; never edits masks or uses model predictions."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2
import numpy as np

BASE=Path(__file__).resolve().parent;OLD=BASE/'runs/run_20261029_001'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sampled_line(p,spacing=1.):
    p=np.asarray(p,float);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];t=np.linspace(0,s[-1],int(np.ceil(s[-1]/spacing))+1)
    return np.column_stack([np.interp(t,s,p[:,j]) for j in range(2)])
def nearest(p,lines):
    d=np.full(len(p),np.inf)
    for line in lines:
        for a,b in zip(np.asarray(line,float)[:-1],np.asarray(line,float)[1:]):
            v=b-a;t=np.clip((p-a)@v/(v@v),0,1);d=np.minimum(d,np.linalg.norm(p-(a+t[:,None]*v),axis=1))
    return d

def run(run_dir):
    run_dir=Path(run_dir).resolve();out=run_dir/'supervision_qa';out.mkdir(exist_ok=True);index=load(run_dir/'data/index.json')
    train=[i for i in index['items'] if i['split']=='train'];frozen=[run_dir/'data/index.json',run_dir/'supervision/annotation_manifest.json']
    frozen += [Path(i[k]) for i in train for k in ['supervision_cache','supervision_annotation']]
    before={str(p):sha(p) for p in frozen};rows=[]
    for item in train:
        ann=load(item['supervision_annotation']);n=ann['neighbor']
        if not n:continue
        im=cv2.imdecode(np.fromfile(item['image_path'],np.uint8),cv2.IMREAD_COLOR);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
        p=sampled_line(n['points_xy']);source=load(OLD/'data/label_sources'/f"{item['id']}.json");d=nearest(p,source['trusted_center_segments_xy'])
        v=np.array(n['points_xy'][-1])-n['points_xy'][0];v=v/np.linalg.norm(v);normal=np.array([-v[1],v[0]])
        offsets=np.linspace(-45,45,181);q=p[:,None,:]+offsets[None,:,None]*normal[None,None,:]
        samples=cv2.remap(gray,q[:,:,0].astype(np.float32),q[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        profile=samples.mean(0);selected=offsets>=-3;selected&=offsets<=3
        row={'id':item['id'],'annotation_type':'original-image engineering weak negative; not independently verified truth',
          'min_distance_to_reviewed_target_center_original_px':float(d.min()),
          'mean_distance_to_reviewed_target_center_original_px':float(d.mean()),
          'selected_visible_body_gray_mean':float(samples[:,selected].mean()),
          'normal_offsets_original_px':offsets.tolist(),'mean_raw_gray_profile':profile.tolist(),
          'profile_role':'native image texture and separator inspection only; not calibrated rope width or identity metric'}
        canvas=np.full((360,760,3),245,np.uint8);x=60+(offsets+45)/90*650;y=295-profile*230
        for val in [0,.25,.5,.75,1]:
            yy=int(295-val*230);cv2.line(canvas,(60,yy),(710,yy),(210,210,210),1);cv2.putText(canvas,f'{val:g}',(12,yy+5),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,40,40),1)
        for val in [-40,-20,0,20,40]:
            xx=int(60+(val+45)/90*650);cv2.putText(canvas,str(val),(xx-10,320),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,40,40),1)
        xa=int(60+42/90*650);xb=int(60+48/90*650);cv2.rectangle(canvas,(xa,65),(xb,295),(220,220,255),-1)
        cv2.polylines(canvas,[np.column_stack([x,y]).astype(np.int32)],False,(0,100,0),2)
        cv2.putText(canvas,'Raw gray normal profile averaged along weak non-target body fragment',(14,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1,cv2.LINE_AA)
        cv2.putText(canvas,'pink = selected +/-3px support; gray axis fixed0..1; no prediction used',(14,48),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1,cv2.LINE_AA)
        cv2.putText(canvas,'normal offset in original pixels',(245,346),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
        profile_path=out/(item['id']+'__raw_profile.png');cv2.imencode('.png',canvas)[1].tofile(str(profile_path));row['profile_png']=str(profile_path);rows.append(row)
    request_rows=[]
    for sid in ['dual_20260924_111500__left__f00240','dual_20260924_111143__right__f00130']:
        item=next(i for i in train if i['id']==sid);source=load(OLD/'data/label_sources'/f'{sid}.json');lines=source['trusted_center_segments_xy'];end=np.array(lines[-1][-1])
        seed=np.array(item['known_seed_xy']);rightward=seed[-1,0]>seed[0,0]
        b=[int(max(0,end[0]-(650 if rightward else 300))),int(max(1050,end[1]-400)),int(min(4096,end[0]+(300 if rightward else 650))),int(min(3000,end[1]+250))]
        im=cv2.imdecode(np.fromfile(item['image_path'],np.uint8),cv2.IMREAD_COLOR);x0,y0,x1,y1=b;raw=im[y0:y1,x0:x1].copy();marked=raw.copy()
        for line in lines:cv2.polylines(marked,[np.rint(np.array(line)-[x0,y0]).astype(np.int32)],False,(0,255,0),2)
        cv2.circle(marked,tuple(np.rint(end-[x0,y0]).astype(int)),9,(0,255,255),2)
        panel=np.concatenate([raw,marked],1);header=np.zeros((58,panel.shape[1],3),np.uint8)
        cv2.putText(header,'Original context | existing trusted center green ends at yellow; no invented continuation',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.66,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(header,'Request only a short confirmed target continuation + distinct competitor; leave ambiguous dark tail unknown',(12,48),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
        panel=np.concatenate([header,panel],0);path=out/(sid+'__minimal_request_context.png');cv2.imencode('.png',panel)[1].tofile(str(path))
        request_rows.append({'id':sid,'supplemental_context_box_xyxy':b,'path':str(path),'original_minimal_window_contains_tail':True,
          'limitation':'Original small raw crop shows dark near-edge tail but is insufficient alone to convey target identity; supplemental context shows trusted predecessor only.',
          'frozen_template_or_supervision_modified':False,'new_target_label_added':False})
    after={str(p):sha(p) for p in frozen};assert before==after
    result={'exit_code':0,'command':sys.argv,'source_sha256':sha(__file__),'neighbor_source_checks':rows,'minimal_request_checks':request_rows,
      'frozen_artifacts_unchanged':True,'frozen_hashes':before,'evaluation_labels_or_predictions_read':False,
      'reviewer':'Codex original-image verification; not independent human identity validation',
      'scope':'mask conflict verification and visual support QA; no training, parameter changes, or new frozen labels'}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'source_checked_neighbor_fragments':len(rows),'context_requests':len(request_rows),'frozen_unchanged':True,
      'min_target_distances_px':[r['min_distance_to_reviewed_target_center_original_px'] for r in rows],'selected_body_gray_means':[r['selected_visible_body_gray_mean'] for r in rows]}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);run(p.parse_args().run)
