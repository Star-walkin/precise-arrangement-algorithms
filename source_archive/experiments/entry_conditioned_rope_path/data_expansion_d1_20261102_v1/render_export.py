"""Lossless two-panel export of all292 sources. Raw D1 heat ONLY, no path."""
from __future__ import annotations
import argparse,hashlib,json,time,sys,traceback,concurrent.futures
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
HEADER=160;FOOTER=190;GAP=32;SIDE=140
Image.MAX_IMAGE_PIXELS=150_000_000
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def font(n):return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
def up2(a):
    # align_corners=False bilinear with edge replication, original B/A rule.
    h,w=a.shape;xx=(np.arange(2*w,dtype=np.float32)+.5)*.5-.5;yy=(np.arange(2*h,dtype=np.float32)+.5)*.5-.5
    x0=np.floor(xx).astype(int);y0=np.floor(yy).astype(int);dx=xx-x0;dy=yy-y0
    b=a[:,x0.clip(0,w-1)]*(1-dx)+a[:,(x0+1).clip(0,w-1)]*dx
    return (b[y0.clip(0,h-1)]*(1-dy[:,None])+b[(y0+1).clip(0,h-1)]*dy[:,None]).astype(np.float32)
def one(job):
    tick=time.perf_counter();dest=HERE/'final_292'/(job['id']+'.png');assert not dest.exists()
    assert sha(job['image_path'])==job['image_sha256'];raw=Image.open(job['image_path']).convert('RGB');assert raw.size==(4096,3000)
    selection=load(HERE/'selected_checkpoints.json');k=0;records=[]
    for ei in range(len(job['seeds'])):
        rp=HERE/'inference/gallery/D1'/job['id']/f'entry_{ei:02d}.npz';m=load(rp.with_suffix('.json'))
        assert m['model_sha256']==selection['paths']['D1']['sha256'] and sha(rp)==m['response_sha256']
        assert m['entry_index']==ei and m['image_sha256']==job['image_sha256']
        assert np.array_equal(np.asarray(m['seed_xy']),np.asarray(job['seeds'][ei]['xy']))
        records.append(m)
    rec=records[k];lut=np.load(BASE/'batch_ab_20261030_001/color_lut.npy');right=np.full((3000,4096,3),45,np.uint8)
    if rec['status']=='predicted':
        with np.load(rec['response_path'],allow_pickle=False) as z:a=z['response'].copy()
        assert a.shape==(975,2048) and a.dtype==np.float32 and np.isfinite(a).all() and 0<=a.min()<=a.max()<=1
        right[1050:]=lut[np.rint(up2(a)*255).astype(np.uint8)]
    else:a=None
    # Nothing is drawn over the heat or source image bodies.
    w,h=raw.size;board=Image.new('RGB',(w*2+GAP+SIDE,h+HEADER+FOOTER),(248,249,251));d=ImageDraw.Draw(board)
    board.paste(raw,(0,HEADER));board.paste(Image.fromarray(right),(w+GAP,HEADER))
    d.text((25,15),'原图  '+job['id'],font=font(40),fill='black')
    d.text((25,85),'4096 × 3000 原始方向与像素  无标签和预测覆盖',font=font(31),fill='black')
    d.text((w+GAP+25,15),f'D1 原始网络中心响应  A750续训第{selection["common_step"]}步',font=font(40),fill='black')
    d.text((w+GAP+25,85),f'固定入口 E1/{len(job["seeds"])}  '+('多入口条件分别保存，歧义保留' if len(job['seeds'])>1 else '未叠加路径，不补线，不遮背景'),font=font(31),fill='black')
    bx=2*w+GAP+20;by=HEADER+600;bar=Image.fromarray(lut[np.linspace(255,0,1400).astype(np.uint8)][:,None].repeat(38,1));board.paste(bar,(bx,by))
    for yy,t in [(by,'1.0'),(by+700,'0.5'),(by+1365,'0.0')]:d.text((bx+43,yy),t,font=font(27),fill='black')
    index=load(HERE/'dataset/index.json');train={i['id']:i for i in index['items'] if i['split']=='train'}
    participation='D1训练图' if job['id'] in train else '同训练来源的展示图' if job['source_group'] in load(HERE/'protocol.json')['train_groups'] else '未入训练的展示来源'
    top=h+HEADER+12
    d.text((20,top),f'用途：{participation}  |  推理状态：{rec["status"]}  |  右栏为本轮D1实际前向结果',font=font(32),fill='black')
    d.text((20,top+52),'响应强度固定0～1 TURBO，未经身份概率校准；原图 y<1050 未计算，以中性灰显示。',font=font(31),fill='black')
    d.text((20,top+104),'网络原生网格2048×975回映至原图坐标；无逐图min–max、形态学补线、mask遮挡或多入口热图合并。',font=font(30),fill='black')
    board.save(dest,compress_level=4)
    with Image.open(dest) as checked:
        assert checked.size==(8364,3350) and checked.format=='PNG'
        assert checked.crop((0,HEADER,w,HEADER+h)).tobytes()==raw.tobytes()
        assert np.array_equal(np.asarray(checked.crop((w+GAP,HEADER,w*2+GAP,HEADER+h))),right),'Heat body changed during rendering'
    return {'id':job['id'],'source_group':job['source_group'],'source':job['image_path'],'source_sha256':job['image_sha256'],
        'output':str(dest),'sha256':sha(dest),'bytes':dest.stat().st_size,'size':[8364,3350],
        'left_pixels_identical':True,'heat_body_unmodified':True,'path_overlay':False,'display_entry':k,'entry_count':len(records),
        'entry_source':job['seeds'][k]['source'],'conditions':records,'model':selection['paths']['D1'],'model_step':selection['common_step'],
        'source_is_D1_training_image':job['id'] in train,'source_group_use':participation,'status':rec['status'],
        'response_path':rec['response_path'],'seconds':time.perf_counter()-tick,'full_identity_verified':False}
def main():
    sys.stdout.reconfigure(encoding='utf8');p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    pr=load(HERE/'protocol.json');jobs=pr['gallery_jobs'];assert len(jobs)==len({j['id'] for j in jobs})==292
    receipt=load(HERE/'inference/gallery/receipt.json');assert receipt['exit_code']==0 and len(receipt['rows'])==301
    out=HERE/'final_292';out.mkdir(exist_ok=False);start=time.perf_counter();rows=[];errors=[]
    with (HERE/'render_progress.jsonl').open('x',encoding='utf8') as log,concurrent.futures.ProcessPoolExecutor(a.workers) as ex:
        pending={ex.submit(one,j):j['id'] for j in jobs}
        for f in concurrent.futures.as_completed(pending):
            try:r=f.result();rows.append(r);log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
            except Exception:errors.append({'id':pending[f],'traceback':traceback.format_exc()})
            if (len(rows)+len(errors))%20==0 or len(rows)+len(errors)==292:print({'done':len(rows),'errors':len(errors),'total':292,'seconds':round(time.perf_counter()-start,1)},flush=True)
    dump(HERE/'render_receipt.json',{'exit_code':int(bool(errors)),'command':sys.argv,'rows':rows,'errors':errors,'seconds':time.perf_counter()-start,
        'exactly_two_native_panels':True,'no_path_overlay':True,'native_float_conditions':sum(r['entry_count'] for r in rows),
        'source_sha256':sha(__file__),'output_directory':str(out)})
    assert not errors
if __name__=='__main__':main()
