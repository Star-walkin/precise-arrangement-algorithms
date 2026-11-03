"""One bounded training-only P1 diagnostic pass; never opens evaluation labels."""
from pathlib import Path
import argparse,json,hashlib,sys,time
import cv2,numpy as np
import p1_segments as p1

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'experiments/entry_conditioned_rope_path/runs/run_20261030_002'
IDS=['dual_20260924_111500__left__f00240','dual_20260924_111143__right__f00130',
     'C02__dual_20260923_101517__pair0550__right','C07__R_run_stop_run_20260922_01__t012.033__f000368']
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,image):cv2.imencode('.png',image)[1].tofile(str(p))
def text(im,t,pos,color=(30,30,30),scale=.65):cv2.putText(im,t,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
def heat(r):
    m=cv2.applyColorMap(np.round(np.clip(np.nan_to_num(r),0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    m[~np.isfinite(r)]=(110,110,110);return m

def diagnostics(gray,response,bundle,paths,seed,scene,path):
    h,w=gray.shape;gray8=np.round(gray*255).astype(np.uint8);original=cv2.cvtColor(gray8,cv2.COLOR_GRAY2BGR)
    ridge=original.copy();routed=original.copy()
    for s in bundle['segments']:cv2.polylines(ridge,[np.rint(s['points_xy']).astype(np.int32)],False,(0,220,255),2)
    for c in paths['candidates']:
        p=np.rint(c['points_xy']).astype(np.int32)
        for j,state in enumerate(c['segment_status']):
            color={'entry_condition':(255,255,0),'heatmap_supported':(30,235,30),'image_supported':(220,200,0),'bridge':(255,80,220),'ambiguous':(30,30,240)}[state]
            if state!='bridge' or j%2==0:cv2.line(routed,tuple(p[j]),tuple(p[j+1]),color,3)
    seed=np.asarray(seed);cv2.polylines(original,[np.rint(seed).astype(np.int32)],False,(255,255,0),4)
    panels=[original,heat(response),ridge,routed];titles=['Original + initial entry','Raw response: fixed 0..1','Extracted ridges, no bridge','P1 candidates; pink=dashed bridge']
    ph=round(h*900/w);canvas=np.full((ph+80,3600,3),250,np.uint8)
    for j,(im,title) in enumerate(zip(panels,titles)):
        canvas[60:60+ph,j*900:(j+1)*900]=cv2.resize(im,(900,ph));text(canvas,title,(j*900+12,32))
    save(path/'overview.png',canvas)
    direction=p1.unit(seed[-1]-seed[0]);normal=np.array([-direction[1],direction[0]])
    locations={'free_after_entry':seed[-1]+80*direction}
    if scene.get('contact_proxy',{}).get('xy') is not None:locations['visual_interface_proxy']=np.array(scene['contact_proxy']['xy'])
    profiles=[];canvas=np.full((len(locations)*420+65,1840,3),250,np.uint8)
    text(canvas,'Training-only fixed locations; normal follows entry direction. Gray/response display fixed 0..1.',(15,30))
    for row,(name,point) in enumerate(locations.items()):
        cx,cy=np.rint(point).astype(int);x0=max(0,cx-128);x1=min(w,cx+128);y0=max(0,cy-128);y1=min(h,cy+128)
        ybase=65+row*420
        for col,im in enumerate([original,heat(response),ridge]):
            crop=im[y0:y1,x0:x1].copy();a=np.rint(point-normal*64-[x0,y0]).astype(int);b=np.rint(point+normal*64-[x0,y0]).astype(int)
            cv2.line(crop,tuple(a),tuple(b),(255,255,255),1)
            canvas[ybase+35:ybase+395,col*420+10:col*420+370]=cv2.resize(crop,(360,360))
            text(canvas,['Original','Raw response','Ridge overlay'][col]+' '+name,(col*420+10,ybase+20),scale=.48)
        offsets=np.linspace(-64,64,257);pts=point+offsets[:,None]*normal
        gv=p1.sample(gray,pts);rv=p1.sample(np.nan_to_num(response,nan=0.).astype(np.float32),pts)
        xx0=1280;yy0=ybase+55;xx1=1810;yy1=ybase+355
        cv2.rectangle(canvas,(xx0,yy0),(xx1,yy1),(150,150,150),1)
        for v in (0,.5,1):
            yy=round(yy1-v*(yy1-yy0));cv2.line(canvas,(xx0,yy),(xx1,yy),(215,215,215),1);text(canvas,str(v),(xx0-35,yy+5),scale=.45)
        for values,col in [(gv,(75,75,75)),(rv,(40,70,220))]:
            points=np.stack([np.linspace(xx0,xx1,len(values)),yy1-values*(yy1-yy0)],1)
            cv2.polylines(canvas,[np.rint(points).astype(np.int32)],False,col,2)
        text(canvas,'gray (gray); raw center response (red)',(xx0,ybase+25),scale=.5)
        text(canvas,'-64                  0                  +64 original px',(xx0,yy1+30),scale=.45)
        profiles.append({'location':name,'center_xy':point.tolist(),'normal_xy':normal.tolist(),'offset_px':offsets.tolist(),
                         'gray':gv.tolist(),'raw_response':rv.tolist(),'label_or_prediction_used_to_choose_location':False})
    save(path/'fixed_profiles.png',canvas);dump(path/'fixed_profiles.json',profiles)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path)
    p.add_argument('--limit',type=int,default=4);p.add_argument('--models',nargs='+',default=['A','B']);args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);cfg={**p1.DEFAULTS,**(load(args.config) if args.config else {})}
    index=load(OLD/'data/index.json');byid={x['id']:x for x in index['items']};rows=[];dump(out/'config.json',cfg)
    for ident in IDS[:args.limit]:
        item=byid[ident];assert item['split']=='train'
        gray=cv2.imdecode(np.fromfile(item['image_path'],np.uint8),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
        scene=load(item['scene_path']);seed=item['known_seed_xy']
        for model in args.models:
            source=OLD/f'branches/{model}/predictions/CONDITIONAL'/ident;inf=load(source/'inference.json')
            response=np.load(source/'response.npy',mmap_mode='r');dest=out/model/ident;dest.mkdir(parents=True)
            tick=time.perf_counter();bundle,arrays=p1.extract(gray,response,np.isfinite(response),item['bbox'],cfg)
            extract_seconds=time.perf_counter()-tick
            candidates,selected,meta=p1.connect_segments(bundle,gray,response,np.isfinite(response),seed,scene,item['bbox'],cfg)
            paths={'id':ident,'model':model,'mode':'CONDITIONAL','selected_candidate_id':selected,'candidates':candidates,'search':meta,
                   'entry':load(source/'entry.json'),'edge_target':scene,'response_sha256':sha(source/'response.npy'),'parameters':cfg}
            dump(dest/'segments.json',bundle);np.savez_compressed(dest/'derived.npz',**arrays);dump(dest/'paths.json',paths)
            diagnostics(gray,response,bundle,paths,seed,scene,dest)
            row={'id':ident,'source_group':item['source_group'],'split':'train','model':model,'response_path':str(source/'response.npy'),
                 'response_sha256':paths['response_sha256'],'inference_metadata':inf,'segments':len(bundle['segments']),
                 'extract_seconds':extract_seconds,'connect_seconds':meta.get('elapsed_seconds'),
                 'selected':selected,'candidates':len(candidates),'top_length_px':candidates[0]['length_px'] if candidates else 0.,
                 'top_bridge_length_px':candidates[0]['bridge_length_px'] if candidates else 0.,
                 'termination':candidates[0]['termination'] if candidates else meta.get('reason',meta.get('selection_reason')),
                 'extract_budgets':bundle['diagnostics'].get('budget_events',[])}
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='inference_metadata'}),flush=True)
    dump(out/'receipt.json',{'items':rows,'command':sys.argv,'exit_code':0,'source_sha256':sha(p1.__file__),
        'calibration_script_sha256':sha(__file__),'cfg_sha256':p1.config_hash(cfg),'labels_read':False,'evaluation_images_read':False,
        'timing_scope':'development timing may overlap root frozen-forward cache generation; formal time measured separately'})

if __name__=='__main__':main()
