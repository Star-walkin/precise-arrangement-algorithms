"""Image-only delivery of frozen A/B on the prior open400 entry list."""
from pathlib import Path
import argparse,csv,json,hashlib,sys,time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[1]
DATA=ROOT/'数据资产/05_派生与实验结果/camera_condition_images_20261009_v2'
OLD_JOBS=CODE/'batch_20261030_001/jobs.json'
PAIR=CODE/'runs/run_20261030_002/paired_comparison.json'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(work):
    import cv2
    from prepare import response_from_lines
    cv2.setNumThreads(2);work.mkdir(parents=True,exist_ok=False)
    cache=work/'cache';cache.mkdir();out=work/'原图_方案A_方案B';out.mkdir()
    old=load(OLD_JOBS);pair=load(PAIR)
    rows=list(csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8-sig')))
    assert len(rows)==old['source_count']==400
    lookup={Path(r['output_file']).stem:r for r in rows};jobs=[]
    for source in old['jobs']:
        row=lookup[source['id']]
        assert '20260925' not in row['source_file']+row['output_file']
        assert not any(s in row['dataset_use_status'].lower() for s in ['reserved','heldout'])
        original=DATA/row['output_file'];assert sha(original)==source['image_sha256']==row['output_sha256']
        bgr=cv2.imdecode(np.fromfile(original,np.uint8),cv2.IMREAD_COLOR);assert bgr.shape[:2]==(3000,4096)
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
        image=cv2.resize(gray[1050:],(2048,975),interpolation=cv2.INTER_AREA)
        seeds=[]
        for entry in source['seeds']:
            seed,_=response_from_lines([entry['xy']],image.shape,source['bbox'],.5,4.)
            assert seed.sum()>0;seeds.append(seed)
        cp=cache/(source['id']+'.npz');np.savez_compressed(cp,image=image,seeds=np.stack(seeds))
        jobs.append({**source,'image_path':str(original),'cache':str(cp),'output':str(out/(source['id']+'.jpg'))})
    lut=cv2.applyColorMap(np.arange(256,dtype=np.uint8)[:,None],cv2.COLORMAP_TURBO)[:,0,::-1].copy()
    np.save(work/'color_lut.npy',lut)
    dump(work/'jobs.json',{'jobs':jobs,'skipped':old['skipped'],'source_count':400,'source_manifest_sha256':sha(DATA/'manifest.csv'),
      'entry_selection_source':str(OLD_JOBS),'entry_selection_sha256':sha(OLD_JOBS),'pair':pair,
      'output_directory':str(out),'response_scale':[0,1],'multiple_entries':'same three columns repeated in separate rows; never merged',
      'no_training':True,'no_path_decoder':True,'source_sha256':sha(__file__)})
    print(json.dumps({'prepared':len(jobs),'entry_conditions':sum(len(j['seeds']) for j in jobs),'multi_entry_images':sum(len(j['seeds'])>1 for j in jobs)},ensure_ascii=False),flush=True)

def run(work):
    import torch
    import torch.nn.functional as F
    from PIL import Image,ImageDraw,ImageFont
    from train import load_model,tensor_sample
    Image.MAX_IMAGE_PIXELS=200_000_000
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    meta=load(work/'jobs.json');pair=meta['pair'];models={};cfg=None
    for key in ['A','B']:
        cp=pair['main_checkpoints'][key];assert sha(cp['path'])==cp['sha256']
        checkpoint=torch.load(cp['path'],map_location='cpu',weights_only=False)
        assert checkpoint['step']==pair['main_comparison_step']==750
        if cfg is None:cfg=checkpoint['config']
        else:assert cfg==checkpoint['config']
        models[key]=load_model(cp['path'])
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',46)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',36)
    lut=np.load(work/'color_lut.npy');records=[];tick=time.time()
    with (work/'progress.jsonl').open('x',encoding='utf-8') as log,torch.inference_mode():
        for number,job in enumerate(meta['jobs'],1):
            with np.load(job['cache'],allow_pickle=False) as z:im=z['image'];seeds=z['seeds']
            h,w=job['shape'];row_height=h+190
            board=Image.new('RGB',(w*3,row_height*len(seeds)),(245,247,250));draw=ImageDraw.Draw(board)
            with Image.open(job['image_path']) as p:original=p.convert('RGB')
            ranges=[]
            for k,seed in enumerate(seeds):
                top=k*row_height;board.paste(original,(0,top+125))
                entry=f'入口候选{k+1}/{len(seeds)}' if len(seeds)>1 else '相同活动绳入口'
                draw.text((25,top+12),'原图 | '+job['id'],font=font,fill=(25,35,45))
                draw.text((25,top+72),entry+' | 两组均为第750步模型',font=small,fill=(80,90,100))
                zeros=np.zeros_like(im)
                arrays={'image':im,'seed':seed,'target':zeros,'valid':zeros.astype(bool),'stage':zeros.astype(np.uint8),'roi':np.ones_like(im,bool)}
                x,_,_,_,_=tensor_sample(arrays,cfg,augment=False)
                hh,ww=im.shape;row_ranges={}
                for j,key in enumerate(['A','B'],1):
                    response=models[key](x)[0,0,:hh,:ww]
                    response=F.interpolate(response[None,None],size=(h-1050,w),mode='bilinear',align_corners=False)[0,0].numpy()
                    assert np.isfinite(response).all() and 0<=response.min()<=response.max()<=1
                    rgb=np.full((h,w,3),38,np.uint8);rgb[1050:]=lut[np.rint(response*255).astype(np.uint8)]
                    left=j*w;board.paste(Image.fromarray(rgb),(left,top+125))
                    title='方案A：原监督续训' if key=='A' else '方案B：加入邻绳及背景弱负监督'
                    draw.text((left+25,top+12),title+' | '+entry,font=font,fill=(25,35,45))
                    draw.text((left+25,top+72),'原始中心响应；灰色区域未计算；白线为入口提示',font=small,fill=(80,90,100))
                    draw.line([(left+p[0],top+125+p[1]) for p in job['seeds'][k]['xy']],fill='white',width=4)
                    bar=Image.fromarray(lut[np.linspace(0,255,1200).astype(np.uint8)][None].repeat(24,0))
                    board.paste(bar,(left+25,top+h+135))
                    draw.text((left+1250,top+h+124),'0 → 1  响应强度（非概率）',font=small,fill=(25,35,45))
                    row_ranges[key]=[float(response.min()),float(response.max())]
                ranges.append(row_ranges)
            output=Path(job['output']);assert not output.exists()
            board.save(output,quality=95,subsampling=0)
            with Image.open(output) as checked:
                assert checked.size==(w*3,row_height*len(seeds));checked.verify()
            record={'id':job['id'],'output':str(output),'dimensions':[w*3,row_height*len(seeds)],'entry_conditions':len(seeds),'ranges':ranges,'bytes':output.stat().st_size}
            records.append(record);log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush()
            cp=Path(job['cache']).resolve();assert cp.parent==(work/'cache').resolve();cp.unlink()
            if number==1 or number%20==0:print(json.dumps({'done':number,'total':len(meta['jobs']),'seconds':round(time.time()-tick,1)}),flush=True)
    dump(work/'receipt.json',{'exit_code':0,'count':len(records),'network_forward_calls':2*sum(r['entry_conditions'] for r in records),
      'seconds':time.time()-tick,'models':pair['main_checkpoints'],'output_directory':meta['output_directory'],'records':records,
      'source_sha256':sha(__file__),'command':sys.argv})
    print(json.dumps({'complete':True,'count':len(records),'seconds':round(time.time()-tick,1),'output':meta['output_directory']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();(prepare if a.mode=='prepare' else run)(a.work.resolve())
