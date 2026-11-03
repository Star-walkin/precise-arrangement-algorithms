"""Render current entry-conditioned model on saved open400 rope observations."""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'数据资产/05_派生与实验结果'
DATA=ASSETS/'camera_condition_images_20261009_v2'
A6=ASSETS/'camera_condition_incoming_template_20261029_v6'
A2=ASSETS/'camera_condition_incoming_rope_20261026_v2/all_001/records'
MODEL=Path(__file__).resolve().parent/'runs/run_20261029_001/model/final.pt'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(work):
    import cv2
    from scene import short_seed
    from prepare import response_from_lines
    cv2.setNumThreads(2);work.mkdir(parents=True,exist_ok=False);cache=work/'cache';cache.mkdir()
    out=work/'原图与热力图';out.mkdir()
    protocol=load(A6/'protocol.json');assert sha(DATA/'manifest.csv')==protocol['input_manifest_sha256']
    rows=list(csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8-sig')));assert len(rows)==400
    jobs=[];skipped=[]
    for row in rows:
        assert '20260925' not in row['source_file']+row['output_file']
        assert not any(v in row['dataset_use_status'].lower() for v in ['reserved','heldout'])
        rel=Path(row['output_file']);p=A6/'all_001/记录'/rel.parent.name/(rel.stem+'.json');result=load(p)['result']
        seeds=[]
        for entry in result['entries']:
            g=entry['free']['geometry'];line=np.array(g['observed_segment_xy'],float)
            direction=np.array(g['direction_xy'])
            if (line[-1]-line[0])@direction<0:line=line[::-1]
            seeds.append({'xy':short_seed(line).tolist(),'source':str(p),'entry_id':entry['entry_id'],'ambiguous':len(result['entries'])>1})
        if not seeds and result['status']=='FREE_ONLY_TEMPLATE_OR_DIRECTION_UNKNOWN':
            fp=A2/rel.parent.name/(rel.stem+'.json');f=load(fp)['free_segment'];assert f['status']=='observed'
            line=np.array(f['observed_segment_xy'],float);direction=f.get('toward_surface_direction_xy')
            if direction is not None and (line[-1]-line[0])@np.array(direction)<0:line=line[::-1]
            seeds.append({'xy':short_seed(line).tolist(),'source':str(fp),'entry_id':'observed_free_only','ambiguous':True})
        if not seeds:skipped.append({'file':row['output_file'],'reason':'no_saved_observed_active_rope_entry'});continue
        image_path=DATA/rel;assert sha(image_path)==row['output_sha256']
        bgr=cv2.imdecode(np.fromfile(image_path,np.uint8),cv2.IMREAD_COLOR);h,w=bgr.shape[:2];assert(h,w)==(3000,4096)
        bbox=[0,1050,w,h];gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
        image=cv2.resize(gray[1050:],(2048,975),interpolation=cv2.INTER_AREA)
        maps=[]
        for seed in seeds:
            a,_=response_from_lines([seed['xy']],image.shape,bbox,.5,4.);assert a.sum()>0;maps.append(a)
        cp=cache/(rel.stem+'.npz');np.savez_compressed(cp,image=image,seeds=np.stack(maps))
        jobs.append({'id':rel.stem,'image_path':str(image_path),'cache':str(cp),'seeds':seeds,'bbox':bbox,'shape':[h,w],
            'output':str(out/(rel.stem+'.jpg')),'image_sha256':row['output_sha256']})
        if len(jobs)%50==0:print('prepared',len(jobs),flush=True)
    lut=cv2.applyColorMap(np.arange(256,dtype=np.uint8)[:,None],cv2.COLORMAP_TURBO)[:,0,::-1].copy();np.save(work/'color_lut.npy',lut)
    dump(work/'jobs.json',{'jobs':jobs,'skipped':skipped,'source_count':400,'model':str(MODEL),'model_sha256':sha(MODEL),
        'output_directory':str(out),'no_training':True,'response_scale':[0,1],'seed_limit_px':120})
    print(json.dumps({'selected':len(jobs),'network_calls':sum(len(j['seeds']) for j in jobs),'skipped':len(skipped)}),flush=True)

def run(work):
    import torch
    import torch.nn.functional as F
    from PIL import Image,ImageDraw,ImageFont
    from train import load_model,tensor_sample
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    config=load(work/'jobs.json');assert sha(MODEL)==config['model_sha256']
    checkpoint=torch.load(MODEL,map_location='cpu',weights_only=False);cfg=checkpoint['config'];net=load_model(MODEL)
    lut=np.load(work/'color_lut.npy');font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',48);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',38)
    tick=time.time();records=[]
    with (work/'progress.jsonl').open('x',encoding='utf-8') as log,torch.inference_mode():
        for n,job in enumerate(config['jobs'],1):
            with np.load(job['cache']) as z:image=z['image'];seeds=z['seeds']
            h,w=job['shape'];canvas=Image.new('RGB',(w*(1+len(seeds)),h+190),(245,247,250));draw=ImageDraw.Draw(canvas)
            with Image.open(job['image_path']) as original:canvas.paste(original.convert('RGB'),(0,125))
            draw.text((25,12),'原图  |  '+job['id'],font=font,fill=(25,35,45))
            ranges=[]
            for k,seed in enumerate(seeds):
                zeros=np.zeros_like(image);a={'image':image,'seed':seed,'target':zeros,'valid':zeros.astype(bool),
                    'stage':zeros.astype(np.uint8),'roi':np.ones_like(image,bool)}
                x,_,_,_,_=tensor_sample(a,cfg);hh,ww=image.shape
                response=net(x)[0,0,:hh,:ww]
                response=F.interpolate(response[None,None],size=(h-1050,w),mode='bilinear',align_corners=False)[0,0].numpy()
                assert np.isfinite(response).all() and 0<=response.min() and response.max()<=1
                rgb=np.full((h,w,3),38,np.uint8);rgb[1050:]=lut[np.rint(response*255).astype(np.uint8)]
                left=w*(k+1);canvas.paste(Image.fromarray(rgb),(left,125))
                label='入口候选'+str(k+1) if len(seeds)>1 else '活动绳入口'
                draw.text((left+25,12),'模型原始热力图  |  '+label+'  |  灰色区域未计算',font=font,fill=(25,35,45))
                # Show the actual short condition only on the heatmap panel.
                points=[(left+p[0],125+p[1]) for p in job['seeds'][k]['xy']]
                draw.line(points,fill=(255,255,255),width=5)
                if job['seeds'][k]['ambiguous']:draw.text((left+25,70),'入口/接续存在歧义；热力图仅为响应强度',font=small,fill=(145,75,0))
                bar=Image.fromarray(lut[np.linspace(0,255,1200).astype(np.uint8)][None].repeat(24,0))
                canvas.paste(bar,(left+25,h+135));draw.text((left+1250,h+124),'0 → 1  响应强度（非概率）',font=small,fill=(25,35,45))
                ranges.append([float(response.min()),float(response.max())])
            dest=Path(job['output']);assert not dest.exists();canvas.save(dest,quality=95,subsampling=0)
            with Image.open(dest) as check:check.verify()
            row={'id':job['id'],'output':str(dest),'response_ranges':ranges,'bytes':dest.stat().st_size}
            records.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            # Remove only this known generated cache file after successful save.
            cp=Path(job['cache']).resolve();assert cp.parent==(work/'cache').resolve();cp.unlink()
            if n==1 or n%20==0:print(json.dumps({'done':n,'total':len(config['jobs']),'seconds':round(time.time()-tick,1)}),flush=True)
    dump(work/'receipt.json',{'exit_code':0,'count':len(records),'seconds':time.time()-tick,'model_sha256':config['model_sha256'],
        'png_jpg_only_delivery':config['output_directory'],'records':records,'command':sys.argv})
    print(json.dumps({'complete':True,'count':len(records),'seconds':time.time()-tick}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--work',type=Path,required=True);args=p.parse_args()
    (prepare if args.mode=='prepare' else run)(args.work.resolve())
