"""Exactly292 lossless two-panel PNGs. Native originals stay untouched."""
from common import *
import numpy as np,cv2,gzip,os,time,sys,traceback,concurrent.futures
from PIL import Image,ImageDraw,ImageFont
Image.MAX_IMAGE_PIXELS=150_000_000
HEADER=160;FOOTER=190;GAP=32;SIDE=140
def font(n):return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
def readgz(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def up2(a):
    h,w=a.shape;xx=(np.arange(2*w,dtype=np.float32)+.5)*.5-.5;yy=(np.arange(2*h,dtype=np.float32)+.5)*.5-.5
    x0=np.floor(xx).astype(int);y0=np.floor(yy).astype(int);dx=xx-x0;dy=yy-y0
    b=a[:,x0.clip(0,w-1)]*(1-dx)+a[:,(x0+1).clip(0,w-1)]*dx
    return (b[y0.clip(0,h-1)]*(1-dy[:,None])+b[(y0+1).clip(0,h-1)]*dy[:,None]).astype(np.float32)
def segment(d,a,b,color,width=3,dash=False):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if dash:
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/10)))
        for i in range(n):
            if i%2==0:d.line([tuple(a+(b-a)*i/n),tuple(a+(b-a)*(i+1)/n)],fill=color,width=width)
    else:d.line([tuple(a),tuple(b)],fill=color,width=width)
def one(j):
    tick=time.time();k=j['main_display_entry_index'];dest=HERE/'交付_292张原图与热力图'/(j['id']+'.png');assert not dest.exists()
    assert sha(j['image_path'])==j['image_sha256'];raw=Image.open(j['image_path']).convert('RGB');assert raw.size==(4096,3000)
    records=[];floatfiles=[]
    for ei in range(len(j['seeds'])):
        recpath=HERE/'batch_002/R1/gallery'/j['id']/f'entry_{ei:02d}.json.gz';rec=readgz(recpath);records.append(rec)
        src=HERE/'cache/gallery'/j['id']/f'entry_{ei:02d}.npz';target=HERE/'原始浮点响应'/j['id']/f'entry_{ei:02d}.npz';target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():os.link(src,target)
        assert sha(target)==rec['response_cache']['sha256']
        floatfiles.append(dict(entry=ei,path=str(target),sha256=sha(target),response_key='response',shape_hw=[975,2048],dtype='float32',unadorned=True))
    rec=records[k];result=rec['result'];rp=Path(floatfiles[k]['path'])
    with np.load(rp) as z:a=z['response']
    mapped=up2(a);lut=np.load(BASE/'batch_ab_20261030_001/color_lut.npy')
    right=np.full((3000,4096,3),45,np.uint8);right[1050:]=lut[np.rint(mapped*255).astype(np.uint8)]
    heat=Image.fromarray(right);d=ImageDraw.Draw(heat)
    chosen=next((c for c in result['candidates'] if c['id']==result.get('selected_candidate_id')),None)
    available_prefix=False
    if chosen is None and result['candidates']:
        chosen=dict(result['candidates'][0]);points=chosen['points_xy'];statuses=chosen['point_status'];last=0
        for ii,s in enumerate(statuses):
            if s=='inferred':break
            last=ii
        chosen['points_xy']=points[:last+1];chosen['point_status']=statuses[:last+1];available_prefix=True
    if chosen is not None:
        p=np.asarray(chosen['points_xy']);statuses=chosen['point_status']
        competitors=[c for c in result['candidates'] if c['id']!=chosen['id']]
        for c in competitors[:2]:
            q=np.asarray(c['points_xy'])
            for ii in range(1,len(q)):
                if c['point_status'][ii]!='entry_condition':segment(d,q[ii-1],q[ii],(235,80,235),2,True)
        for ii in range(1,len(p)):
            if statuses[ii]=='entry_condition':continue
            ambiguous=False
            if result.get('ambiguous') and competitors:
                q=np.asarray(competitors[0]['points_xy']);ambiguous=np.linalg.norm(q-p[ii],axis=1).min()>max(12.,result.get('width_px',20)*.5)
            weak=statuses[ii]=='inferred';color=(240,80,240) if ambiguous else (255,155,25) if weak else (255,255,255)
            segment(d,p[ii-1],p[ii],color,3,weak or ambiguous)
        if len(p):
            x,y=p[-1];d.ellipse((x-6,y-6,x+6,y+6),outline='white',width=2)
    seed=np.asarray(j['seeds'][k]['xy'])
    for ii in range(1,len(seed)):segment(d,seed[ii-1],seed[ii],(200,200,200),2,False)
    w,h=raw.size;board=Image.new('RGB',(2*w+GAP+SIDE,h+HEADER+FOOTER),(248,249,251));bd=ImageDraw.Draw(board)
    board.paste(raw,(0,HEADER));board.paste(heat,(w+GAP,HEADER))
    bd.text((25,15),'原图  '+j['id'],font=font(40),fill='black')
    bd.text((25,82),'4096 × 3000 原始方向与像素  未叠加标签或预测',font=font(31),fill='black')
    bd.text((w+GAP+25,15),'A750原始中心响应＋R1接续  实验结果',font=font(42),fill='black')
    multi='入口歧义保留' if len(j['seeds'])>1 else '未合并其他入口热图'
    bd.text((w+GAP+25,85),f'展示入口 E{k+1}/{len(j["seeds"])}  固定原编号优先  {multi}',font=font(31),fill='black')
    bar=Image.fromarray(lut[np.linspace(255,0,1400).astype(np.uint8)][:,None].repeat(38,1));bx=2*w+GAP+20;by=HEADER+600
    board.paste(bar,(bx,by))
    for yy,t in [(by,'1.0'),(by+700,'0.5'),(by+1365,'0.0')]:bd.text((bx+43,yy),t,font=font(27),fill='black')
    top=h+HEADER+12
    status='拒识，仅显示可用前缀' if result['selected_candidate_id'] is None else '存在竞争接续，身份未确认' if result['ambiguous'] else '候选路径，完整身份与末端未确认'
    bd.text((20,top),'状态：'+status+'  |  '+result['status'],font=font(33),fill='black')
    bd.text((20,top+52),'白细实线：原图或热图支持；橙虚线：有限桥接；紫虚线：竞争或歧义；灰短线：已给入口，不计识别。',font=font(31),fill='black')
    bd.text((20,top+104),'固定0～1 TURBO色标，响应强度非身份概率；原图y<1050未计算，右栏显示中性灰；网络原生网格2048×975后回映。',font=font(30),fill='black')
    dest.parent.mkdir(parents=True,exist_ok=True);board.save(dest,compress_level=4)
    # Decode final artifact, verify source pixels and exactly2 native viewports.
    with Image.open(dest) as checked:
        assert checked.size==(8364,3350)
        assert checked.crop((0,HEADER,w,HEADER+h)).tobytes()==raw.tobytes()
    return dict(id=j['id'],output=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,dimensions=[8364,3350],source=j['image_path'],source_sha256=j['image_sha256'],
        left_pixels_identical=True,display_entry=k,total_entries=len(j['seeds']),entry_selection='fixed original index0, not by quality or reference',
        conditions=[r['receipt'] for r in records],float_responses=floatfiles,status=result['status'],seconds=time.time()-tick,
        full_target_identity_verified=False)
def main():
    sys.stdout.reconfigure(encoding='utf8');p=load(HERE/'protocol.json');jobs=p['jobs'];assert len(jobs)==292
    assert load(HERE/'batch_002/summary_gallery.json')['exit_code']==0
    target=HERE/'交付_292张原图与热力图';target.mkdir(exist_ok=False);tick=time.time();rows=[];errors=[]
    with (HERE/'render_progress.jsonl').open('x',encoding='utf8') as log,concurrent.futures.ProcessPoolExecutor(4) as ex:
        pending={ex.submit(one,j):j['id'] for j in jobs}
        for f in concurrent.futures.as_completed(pending):
            try:r=f.result();rows.append(r);log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
            except Exception:errors.append(dict(id=pending[f],error=traceback.format_exc()))
            if (len(rows)+len(errors))%15==0 or len(rows)+len(errors)==292:print(dict(done=len(rows),errors=len(errors),total=292,seconds=round(time.time()-tick,1)),flush=True)
    dump(HERE/'render_receipt.json',dict(rows=rows,errors=errors,exit_code=int(bool(errors)),seconds=time.time()-tick,planned=292,
        native_two_panels=True,raw_float_response_count=sum(len(r['float_responses']) for r in rows)))
    if errors:print(errors[:1]);sys.exit(bool(errors))
if __name__=='__main__':main()
