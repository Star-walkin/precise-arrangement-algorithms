"""292 lossless native three-panel views: raw / U1 raw heat / R1 path candidates.

Writes large images ONLY under the explicit C: result directory. User labels
are never opened or painted. Metadata-only training-image identity uses SHA.
"""
from __future__ import annotations
import argparse
from collections import Counter
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
OUTPUT=Path('C:/precise_arrangement_results/manual_r1_20261103_v1/final_292')
WIDTH=4096
HEIGHT=3000
HEADER=160
FOOTER=210
GAP=32
SIDE=140
X_HEAT=WIDTH+GAP
X_PATH=2*(WIDTH+GAP)
BOARD_SIZE=(3*WIDTH+2*GAP+SIDE,HEIGHT+HEADER+FOOTER)
Image.MAX_IMAGE_PIXELS=150_000_000
CONTEXT=None

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def font(size):return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size)

def up2(a):
    """Exactly the old align_corners=False bilinear x2 mapping, edge replicate."""
    h,w=a.shape
    xx=(np.arange(2*w,dtype=np.float32)+.5)*.5-.5
    yy=(np.arange(2*h,dtype=np.float32)+.5)*.5-.5
    x0=np.floor(xx).astype(int);y0=np.floor(yy).astype(int)
    dx=xx-x0;dy=yy-y0
    b=a[:,x0.clip(0,w-1)]*(1-dx)+a[:,(x0+1).clip(0,w-1)]*dx
    return (b[y0.clip(0,h-1)]*(1-dy[:,None])+b[(y0+1).clip(0,h-1)]*dy[:,None]).astype(np.float32)

def dashed(draw,points,color,width=4,dash=18,gap=11):
    """Dashes retain absolute original image coordinates and continuous phase."""
    phase=0.
    for a,b in zip(points[:-1],points[1:]):
        a=np.asarray(a,float);b=np.asarray(b,float);delta=b-a;length=float(np.linalg.norm(delta))
        if length<=1e-9:continue
        start=0.
        while start<length:
            position=phase%(dash+gap)
            enabled=position<dash
            run=min(length-start,(dash if enabled else dash+gap)-position)
            if enabled:
                p=a+delta*(start/length);q=a+delta*((start+run)/length)
                draw.line([tuple(p),tuple(q)],fill=color,width=width)
            start+=run;phase+=run

def candidate(candidate):
    points=np.asarray(candidate.get('points_xy',[]),float)
    states=candidate.get('point_status',[])
    assert points.ndim==2 and points.shape[1]==2 and len(points)==len(states)
    assert np.isfinite(points).all()
    assert (points[:,0]>=0).all() and (points[:,0]<WIDTH).all()
    assert (points[:,1]>=1050).all() and (points[:,1]<HEIGHT).all()
    assert all(s in ('entry_condition','visible_supported','inferred') for s in states)
    if 'supported' in candidate:
        assert candidate['supported']==[s=='visible_supported' for s in states]
    return points,states

def path_overlay(raw,record):
    """Independent faint candidates; selected supported/inferred stages differ."""
    view=raw.copy()
    draw=ImageDraw.Draw(view)
    alternatives=record.get('alternatives',[])
    assert len(alternatives)<=3
    competition_colors=[(150,138,166),(128,146,159),(164,146,154)]
    for item,color in zip(alternatives,competition_colors):
        points,_=candidate(item)
        if len(points)>1:dashed(draw,points,color,width=3,dash=15,gap=8)
    chosen=record.get('selected')
    if chosen is not None:
        points,states=candidate(chosen)
        for j in range(1,len(points)):
            pair=points[j-1:j+1]
            if 'inferred' in states[j-1:j+1]:
                dashed(draw,pair,(255,166,45),width=6,dash=18,gap=11)
            elif states[j]=='entry_condition':
                dashed(draw,pair,(220,223,227),width=5,dash=12,gap=9)
            else:
                draw.line([tuple(pair[0]),tuple(pair[1])],fill=(22,239,192),width=6)
    return view

def source_use(job,index,train_groups,eval_groups):
    match=next((i for i in index['items'] if i['image_sha256']==job['image_sha256']),None)
    if match:
        use='training_manual' if match['origin']=='manual' else 'training_old'
        text='本轮人工训练图' if match['origin']=='manual' else '本轮旧训练图'
    elif job['source_group'] in eval_groups:
        use='historical_open_evaluation_source_display';text='历史开放评价来源展示帧'
    elif job['source_group'] in train_groups:
        use='same_training_source_unused_frame_display';text='同训练来源未用帧'
    else:
        use='other_open_untrained_source_display';text='其他开放未训练来源'
    return use,text,match

def worker_init():
    global CONTEXT
    protocol=load(HERE/'protocol.json')
    selection=load(HERE/'selected_checkpoints.json')
    index=load(HERE/'dataset/index.json')
    infer=load(HERE/'inference/gallery/receipt.json')
    paths=load(HERE/'paths/gallery_receipt.json')
    relation=load(HERE/'relation/model/training_receipt.json')
    assert infer['exit_code']==paths['exit_code']==relation['exit_code']==0
    assert len(infer['rows'])==len(paths['rows'])==301
    CONTEXT=dict(protocol=protocol,selection=selection,index=index,
                 inference={(r['id'],r['entry_index']):r for r in infer['rows']},
                 paths={(r['id'],r['entry_index']):r for r in paths['rows']},
                 train_groups={i['source_group'] for i in index['items'] if i['split']=='train'},
                 eval_groups=set(protocol['eval_groups']),relation=relation,
                 lut=np.load(BASE/'batch_ab_20261030_001/color_lut.npy'))
    assert CONTEXT['lut'].shape==(256,3) and CONTEXT['lut'].dtype==np.uint8

def one(job):
    tick=time.perf_counter();ctx=CONTEXT
    dest=OUTPUT/(job['id']+'.png');assert not dest.exists()
    assert sha(job['image_path'])==job['image_sha256']
    with Image.open(job['image_path']) as f:raw=f.convert('RGB');raw.load()
    assert raw.size==(WIDTH,HEIGHT)
    model=ctx['selection']['paths']['U1'];model_step=ctx['selection']['common_step']
    records=[];path_records=[]
    for entry in range(len(job['seeds'])):
        rec=ctx['inference'][(job['id'],entry)]
        assert rec['entry_index']==entry and rec['image_sha256']==job['image_sha256']
        assert rec['model_sha256']==model['sha256'] and sha(rec['response_path'])==rec['response_sha256']
        assert np.array_equal(np.asarray(rec['seed_xy']),np.asarray(job['seeds'][entry]['xy']))
        p=ctx['paths'][(job['id'],entry)]
        assert sha(p['path'])==p['sha256']
        record=load(p['path']);meta=record['metadata']
        assert record['entry_index']==entry and record['arm']=='R1_new'
        assert meta['image_sha256']==job['image_sha256'] and meta['response_sha256']==rec['response_sha256']
        assert meta['U1_model_sha256']==model['sha256']
        assert meta['relation_model_sha256']==ctx['relation']['portable_sha256']
        assert meta['target_labels_read'] is False and meta['evaluation_references_read'] is False
        assert len(record['result']['candidates'])<=3
        records.append(rec);path_records.append(record)
    # Frozen entry0 regardless of path status or visual heat quality.
    entry=0;rec=records[entry];path=path_records[entry]
    heat=np.full((HEIGHT,WIDTH,3),45,np.uint8)
    if rec['status']=='predicted':
        with np.load(rec['response_path'],allow_pickle=False) as z:response=z['response']
        assert response.shape==(975,2048) and response.dtype==np.float32 and np.isfinite(response).all()
        assert 0<=response.min()<=response.max()<=1
        heat[1050:]=ctx['lut'][np.rint(up2(response)*255).astype(np.uint8)]
    overlay=path_overlay(raw,path)
    board=Image.new('RGB',BOARD_SIZE,(248,249,251))
    board.paste(raw,(0,HEADER));board.paste(Image.fromarray(heat),(X_HEAT,HEADER));board.paste(overlay,(X_PATH,HEADER))
    draw=ImageDraw.Draw(board)
    draw.text((22,15),'原图  '+job['id'],font=font(39),fill='black')
    draw.text((22,88),'4096 × 3000 原始方向、原始像素；无标签或预测覆盖',font=font(30),fill='black')
    draw.text((X_HEAT+22,15),f'U1 原始中心响应｜人工中心+邻绳监督｜第{model_step}步',font=font(39),fill='black')
    draw.text((X_HEAT+22,88),f'固定入口 E1 / {len(records)}；固定0～1 TURBO，无补线、无路径叠加',font=font(30),fill='black')
    chosen=path.get('selected')
    path_state=('歧义：保留竞争' if path['status']=='AMBIGUOUS' else '位置候选' if chosen is not None else '拒识：未选定最终路径')
    draw.text((X_PATH+22,15),'R1_new 活动绳位置候选｜'+path_state,font=font(39),fill='black')
    draw.text((X_PATH+22,88),f'{path["status"]}；端点身份及完整同根尚未验证',font=font(30),fill='black')
    bx=3*WIDTH+2*GAP+20;by=HEADER+600
    colorbar=Image.fromarray(ctx['lut'][np.linspace(255,0,1400).astype(np.uint8)][:,None].repeat(38,1))
    board.paste(colorbar,(bx,by))
    for yy,label in [(by,'1.0'),(by+700,'0.5'),(by+1365,'0.0')]:
        draw.text((bx+43,yy),label,font=font(27),fill='black')
    use,text,matched=source_use(job,ctx['index'],ctx['train_groups'],ctx['eval_groups'])
    top=HEIGHT+HEADER+12
    draw.text((22,top),f'用途：{text}｜来源组：{job["source_group"]}｜固定原编号E1，{len(records)}个入口条件分别保存；不按效果选入口。',font=font(32),fill='black')
    draw.text((22,top+53),'中栏：网络2048×975中心响应回映原图，原图y<1050未计算为灰色；未经身份概率校准，不逐图拉伸、不遮高响应背景。',font=font(31),fill='black')
    draw.text((22,top+106),'右栏：青绿=算法声称图像支持的位置候选；橙虚线=推断连接；灰紫虚线=独立竞争路径（最多3条）；这些均可能错接，不是人工标签。',font=font(31),fill='black')
    board.save(dest,compress_level=3)
    with Image.open(dest) as checked:
        assert checked.format=='PNG' and checked.size==BOARD_SIZE
        assert checked.crop((0,HEADER,WIDTH,HEADER+HEIGHT)).tobytes()==raw.tobytes()
        assert np.array_equal(np.asarray(checked.crop((X_HEAT,HEADER,X_HEAT+WIDTH,HEADER+HEIGHT))),heat)
        assert checked.crop((X_PATH,HEADER,X_PATH+WIDTH,HEADER+HEIGHT)).tobytes()==overlay.tobytes()
    return dict(id=job['id'],source_group=job['source_group'],source=job['image_path'],source_sha256=job['image_sha256'],
        output=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,size=list(BOARD_SIZE),
        left_pixels_identical=True,raw_heat_body_unmodified=True,path_raster_matches_prediction=True,
        image_viewports_xyxy=[[0,HEADER,WIDTH,HEADER+HEIGHT],[X_HEAT,HEADER,X_HEAT+WIDTH,HEADER+HEIGHT],
                              [X_PATH,HEADER,X_PATH+WIDTH,HEADER+HEIGHT]],
        model=model,model_step=model_step,R1_model=dict(path=path['metadata']['relation_model_path'],sha256=path['metadata']['relation_model_sha256']),
        selected_entry=entry,entry_count=len(records),entry_source=job['seeds'][entry].get('source'),
        source_use=use,source_use_label=text,source_is_training_image=matched is not None,
        matched_training_id=matched['id'] if matched else None,matched_training_origin=matched['origin'] if matched else None,
        response_path=rec['response_path'],response_sha256=rec['response_sha256'],
        path_record=ctx['paths'][(job['id'],entry)]['path'],path_record_sha256=ctx['paths'][(job['id'],entry)]['sha256'],
        response_status=rec['status'],path_status=path['status'],selected_candidate_id=chosen['id'] if chosen else None,
        alternative_count=len(path.get('alternatives',[])),all_conditions=[dict(entry_index=i,
            response_path=r['response_path'],response_sha256=r['response_sha256'],status=r['status'],
            path=ctx['paths'][(job['id'],i)]['path'],path_sha256=ctx['paths'][(job['id'],i)]['sha256']) for i,r in enumerate(records)],
        no_manual_labels_rendered=True,full_identity_verified=False,seconds=time.perf_counter()-tick)

def check_synthetic():
    # Test the exact original-coordinate path mapping and absence of accidental
    # joins, using only synthetic images. No user prediction or label opened.
    raw=Image.new('RGB',(WIDTH,HEIGHT),(20,30,40))
    record=dict(selected=dict(points_xy=[[100,1500],[150,1500],[200,1500]],
                              point_status=['entry_condition','visible_supported','inferred'],
                              supported=[False,True,False]),
                alternatives=[dict(points_xy=[[100,1600],[200,1600]],point_status=['entry_condition','visible_supported'])])
    overlay=path_overlay(raw,record)
    a=np.asarray(overlay)
    assert tuple(a[1500,120])==(22,239,192)
    assert tuple(a[1500,160])==(255,166,45)
    assert tuple(a[1550,150])==(20,30,40),'Separate paths must never be filled/joined'
    assert raw.getpixel((120,1500))==(20,30,40),'Overlay mutated original image'
    source=np.array([[0.,1.],[.2,.8]],np.float32);mapped=up2(source)
    assert mapped.shape==(4,4) and mapped[0,0]==0 and mapped[-1,-1]==np.float32(.8)
    assert abs(float(mapped[1,1])-.275)<1e-7
    dest=HERE/'render_synthetic_checks.json'
    dump(dest,dict(exit_code=0,synthetic_only=True,original_coordinate_path_mapping=True,
                  supported_cyan_inferred_orange=True,independent_candidates_not_merged=True,
                  original_image_untouched=True,x2_align_corners_false_mapping=True,
                  board_size=list(BOARD_SIZE),source_sha256=sha(__file__),large_output_directory=str(OUTPUT)))
    print('Synthetic render coordinate/color/mapping checks passed',flush=True)

def main():
    sys.stdout.reconfigure(encoding='utf-8');parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--check-synthetic',action='store_true')
    args=parser.parse_args()
    if args.check_synthetic:return check_synthetic()
    protocol=load(HERE/'protocol.json');jobs=protocol['gallery_jobs']
    assert len(jobs)==len({j['id'] for j in jobs})==292
    infer=load(HERE/'inference/gallery/receipt.json');paths=load(HERE/'paths/gallery_receipt.json')
    assert infer['exit_code']==paths['exit_code']==0 and len(infer['rows'])==len(paths['rows'])==301
    assert {(r['id'],r['entry_index']) for r in infer['rows']}=={(r['id'],r['entry_index']) for r in paths['rows']}
    assert load(HERE/'render_synthetic_checks.json')['source_sha256']==sha(__file__)
    assert not (HERE/'render_receipt.json').exists()
    OUTPUT.mkdir(parents=True,exist_ok=False);started=time.perf_counter();rows=[];errors=[]
    with (HERE/'render_progress.jsonl').open('x',encoding='utf-8') as log,concurrent.futures.ProcessPoolExecutor(args.workers,initializer=worker_init) as pool:
        futures={pool.submit(one,job):job['id'] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                row=future.result();rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            except Exception:
                errors.append(dict(id=futures[future],traceback=traceback.format_exc()))
            if (len(rows)+len(errors))%20==0 or len(rows)+len(errors)==len(jobs):
                print(dict(done=len(rows),errors=len(errors),total=292,seconds=round(time.perf_counter()-started,1)),flush=True)
    receipt=dict(exit_code=int(bool(errors)),command=sys.argv,rows=rows,errors=errors,
                 seconds=time.perf_counter()-started,source_sha256=sha(__file__),
                 output_directory=str(OUTPUT),three_native_viewports=True,viewport_size=[WIDTH,HEIGHT],
                 exactly292_unique_sources=len(rows)==len({r['id'] for r in rows})==292,
                 source_use_counts=dict(Counter(r['source_use'] for r in rows)),
                 entry_conditions=sum(r['entry_count'] for r in rows),
                 U1_inference_receipt_sha256=sha(HERE/'inference/gallery/receipt.json'),
                 R1_path_receipt_sha256=sha(HERE/'paths/gallery_receipt.json'),
                 middle_raw_heat_unmodified=True,no_manual_labels_rendered=True,full_identity_accuracy=None)
    dump(HERE/'render_receipt.json',receipt)
    assert not errors,errors[:1]
    assert len(list(OUTPUT.glob('*.png')))==292
    print('Completed292 native three-panel PNG:',OUTPUT,flush=True)

if __name__=='__main__':main()
