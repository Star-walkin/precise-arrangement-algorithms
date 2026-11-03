"""Lossless original-pixel four panels: source/U1 heat/H6 candidates/H6 result.

Only saved NEW U1 responses and this run's H6 records are consumed. Source,
heat and every displayed path are kept separate; no labels, R1 or B outputs.
"""
from pathlib import Path
from collections import Counter
import argparse,concurrent.futures,gzip,hashlib,json,sys,time,traceback
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
BULK=Path('C:/precise_arrangement_results/manual_u1_p1_20261103_v1')
WIDTH=4096;HEIGHT=3000;HEADER=180;FOOTER=220;GAP=32;SIDE=140
PANEL_X=[i*(WIDTH+GAP) for i in range(4)]
PANEL_BOXES=[[x,HEADER,x+WIDTH,HEADER+HEIGHT] for x in PANEL_X]
BOARD_SIZE=(4*WIDTH+3*GAP+SIDE,HEIGHT+HEADER+FOOTER)
Image.MAX_IMAGE_PIXELS=200_000_000
CONTEXT=None

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load_gz(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def font(n):return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)

def official_lut():
    return cv2.applyColorMap(np.arange(256,dtype=np.uint8)[:,None],cv2.COLORMAP_TURBO)[:,0,::-1].copy()

def up2(a):
    """Original align_corners=False half-grid mapping, edge replicate."""
    h,w=a.shape
    x=(np.arange(2*w,dtype=np.float32)+.5)*.5-.5
    y=(np.arange(2*h,dtype=np.float32)+.5)*.5-.5
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    fx=x-x0;fy=y-y0
    b=a[:,x0.clip(0,w-1)]*(1-fx)+a[:,(x0+1).clip(0,w-1)]*fx
    return (b[y0.clip(0,h-1)]*(1-fy[:,None])+b[(y0+1).clip(0,h-1)]*fy[:,None]).astype(np.float32)

def original_points(xy):
    q=np.asarray(xy,float).reshape(-1,2)
    assert np.isfinite(q).all()
    return np.rint(2*q+[0,1050]).astype(np.int32)

def line(im,points,color,width=6):
    p=original_points(points)
    if len(p)>1:cv2.polylines(im,[p],False,color,width,cv2.LINE_AA)

def candidates_panel(raw,record):
    im=np.rint(raw.astype(np.float32)*.65).astype(np.uint8)
    for track in record['surface_tracks']['tracks']:
        p=np.asarray(track['points_xy'],float);weak=np.asarray(track['f3_pair_inferred'],bool)
        assert len(p)==len(weak)
        for j in range(1,len(p)):
            inferred=bool(weak[j] or weak[j-1])
            if not inferred or j%5<3:line(im,p[j-1:j+1],(110,110,110) if inferred else (170,170,170),2)
    colors=[(55,235,40),(30,220,255),(240,70,220)]  # original H6 BGR reversed to RGB
    for n,item in reversed(list(enumerate(record['displayed_candidates']))):
        assert n<3
        e=record['entries'][item['entry']]
        c=e['connection']['candidates'][item['candidate']]
        line(im,c['points_xy'],colors[n],6)
    for ei,e in enumerate(record['entries']):
        p=(np.asarray(e['seed']['xy'],float)-[0,1050])*.5
        line(im,p,(255,255,255),6)
        xy=tuple(np.rint(np.asarray(e['seed']['xy'][0],float)).astype(int))
        cv2.putText(im,'E'+str(ei+1),xy,cv2.FONT_HERSHEY_SIMPLEX,1.6,(255,255,255),4,cv2.LINE_AA)
    return im

def final_panel(raw,record):
    im=raw.copy();ei=int(record['selected_entry']);ci=record['selected_candidate']
    if ci is None:return im
    c=record['entries'][ei]['connection']['candidates'][ci]
    p=original_points(c['points_xy'])
    assert len(p)==len(c['supported'])==len(c['stage'])==len(c['point_status'])
    for k in range(1,len(p)):
        supported=bool(c['supported'][k] and c['supported'][k-1])
        stage=c['stage'][k]
        color=(45,230,40) if stage=='free' else (35,210,255) if stage=='transition' else (230,70,230)
        heat_only=c['point_status'][k]=='heatmap_supported' and c['point_status'][k-1] in ('heatmap_supported','joint_heat_image_supported','image_supported')
        raw_a7=c['point_status'][k]=='raw_a7_supported' and c['point_status'][k-1] in ('raw_a7_supported','image_supported','joint_heat_image_supported')
        if raw_a7:color=(60,180,255)
        elif heat_only:color=(250,220,20)
        elif not supported:color=(255,155,0)
        if supported or heat_only or raw_a7 or k%5<3:
            cv2.line(im,tuple(p[k-1]),tuple(p[k]),color,6,cv2.LINE_AA)
    if len(p):cv2.circle(im,tuple(p[-1]),14,(255,255,255),4,cv2.LINE_AA)
    return im

def worker_init():
    global CONTEXT
    cv2.setNumThreads(1)
    task=load(HERE/'TASK.json')
    jobs={j['id']:j for j in task['rows']}
    # Metadata only: label geometry and training caches are NEVER opened.
    train=load(BASE/'manual_r1_20261103_v1/dataset/index.json')['items']
    training_sha={i['image_sha256']:i for i in train}
    p=load(BASE/'manual_r1_20261103_v1/protocol.json')
    groups={i['id']:i['source_group'] for i in p['gallery_jobs']}
    CONTEXT=dict(task=task,jobs=jobs,lut=official_lut(),training=training_sha,
                 train_groups=set(p['train_groups']),eval_groups=set(p['eval_groups']),groups=groups)

def purpose(job):
    ctx=CONTEXT;matched=ctx['training'].get(job['image_sha256']);g=ctx['groups'][job['id']]
    if matched:return 'training_'+matched['origin'],'人工训练图' if matched['origin']=='manual' else '旧训练图'
    if g in ctx['eval_groups']:return 'historical_open_evaluation_source_display','历史开放评价来源展示'
    if g in ctx['train_groups']:return 'same_training_source_unused_frame_display','同训练来源未用帧展示'
    return 'other_open_untrained_source_display','其他开放未训练来源展示'

def condition_record(entry,model):
    path=Path(entry['response_file']);metadata=entry['response_metadata']
    assert sha(path)==entry['response_sha256']==metadata['response_sha256']
    assert metadata['model_sha256']==model['sha256'] and metadata['model_step']==800
    assert metadata['branch']=='U1' and metadata['status']=='predicted' and metadata['new_model_forward']
    assert not metadata['target_labels_read']
    assert np.array_equal(np.asarray(metadata['seed_xy']),np.asarray(entry['seed']['xy']))
    return dict(entry_index=entry['entry_index'],seed_xy=entry['seed']['xy'],response_file=str(path),
                response_sha256=entry['response_sha256'],model_path=metadata['model_path'],
                model_sha256=metadata['model_sha256'],model_step=metadata['model_step'],
                source_image_sha256=metadata['image_sha256'],metadata=metadata)

def render_one(record_file,output):
    tick=time.perf_counter();record=load_gz(record_file);ctx=CONTEXT
    job=ctx['jobs'][record['id']];model=ctx['task']['U1_model']
    assert sha(job['image_path'])==job['image_sha256']==record['source_image_sha256']
    with Image.open(job['image_path']) as src:raw=np.asarray(src.convert('RGB')).copy()
    assert raw.shape==(HEIGHT,WIDTH,3)
    entries=record['entries'];ei=int(record['selected_entry'])
    assert 0<=ei<len(entries) and len(entries)==job['entry_count']
    conditions=[condition_record(e,model) for e in entries]
    chosen=entries[ei];selected_condition=conditions[ei]
    with np.load(chosen['response_file'],allow_pickle=False) as z:heat=z['response'].copy()
    assert heat.shape==(975,2048) and heat.dtype==np.float32 and np.isfinite(heat).all() and 0<=heat.min()<=heat.max()<=1
    middle=np.full((HEIGHT,WIDTH,3),45,np.uint8)
    middle[1050:]=ctx['lut'][np.rint(up2(heat)*255).astype(np.uint8)]
    multi=candidates_panel(raw,record);final=final_panel(raw,record)
    ci=record['selected_candidate'];best=None if ci is None else chosen['connection']['candidates'][ci]
    ambiguous=chosen['connection'].get('ambiguous',True) or bool(chosen['seed'].get('direction_ambiguous',False))
    shown=record['displayed_candidates']
    # Match original P1 ambiguity display, without changing chosen result.
    if len(shown)>1 and shown[0]['entry']!=shown[1]['entry'] and shown[0]['score']-shown[1]['score']<.045:ambiguous=True
    quality='未获得可评分接续候选' if best is None else '保留最高评分候选，有歧义' if ambiguous else '候选接续待确认'
    if best and best['kind']=='free_fragment_no_resolved_connection':quality='保留直段，未强制接入弱证据绳圈'
    details='；'.join(f'{n+1}:E{x["entry"]+1} 分={x["score"]:.3f}'+(' 朝外仅诊断' if x.get('task_direction_allowed') is False else '')
                      for n,x in enumerate(shown)) or '没有受支持的自由段接续；不按固定间距补线'
    board=Image.new('RGB',BOARD_SIZE,(246,248,251));draw=ImageDraw.Draw(board)
    panels=[raw,middle,multi,final]
    for array,x in zip(panels,PANEL_X):board.paste(Image.fromarray(array),(x,HEADER))
    titles=['1 原图','2 最新U-Net U1热力图｜条件入口 E'+str(ei+1),
            '3 F3来源中心轨迹与合法接续候选','4 P1/H6冻结后处理｜'+quality]
    subtitles=[job['id'],f'U1第800步；响应强度0～1；入口{ei+1}/{len(entries)}，各条件分别保存',details,
               '沿用P1/H6单轨及非跨轨规则；几何许可不等于同根身份正确']
    third=['4096×3000 原始方向与像素；未叠标注或预测','中性灰区域未计算；无路径叠加、补线或逐图拉伸',
           '灰实线：原F3支持；灰虚线：源内推断；绿/黄/紫：候选','绿：原图+热图；黄：热图；浅蓝：A7；青/紫：过渡/贴面；橙虚：弱证据']
    for x,title,sub,note in zip(PANEL_X,titles,subtitles,third):
        draw.text((x+20,10),title,font=font(38),fill=(20,30,40))
        draw.text((x+20,65),sub,font=font(29),fill=(50,65,80))
        draw.text((x+20,112),note,font=font(29),fill=(50,65,80))
    barx=PANEL_X[-1]+WIDTH+20;bary=HEADER+600
    bar=Image.fromarray(ctx['lut'][np.linspace(255,0,1400).astype(np.uint8)][:,None].repeat(38,1))
    board.paste(bar,(barx,bary))
    for y,t in [(bary,'1.0'),(bary+700,'0.5'),(bary+1365,'0.0')]:draw.text((barx+43,y),t,font=font(27),fill='black')
    use,use_label=purpose(job)
    foot=HEADER+HEIGHT+12
    footer=[f'用途：{use_label}；入口按冻结P1/H6规则评分选择E{ei+1}/{len(entries)}；热图和所选路径入口一致，不合并条件。',
            '本轮为人工监督新U1＋P1后处理；原B750＋H6基准保持不动。没有R1、训练或新增网络前向。',
            '计算域原图y≥1050，网络2048×975回映原图；4栏均原生4096×3000。中心响应未经身份概率校准，候选高分不等于完整识别成功。',
            '图3/图4仅画本轮新热图产生的候选；灰轨来自冻结F3向量。缺线、未知、拒识和真实失败保留。']
    for k,text in enumerate(footer):draw.text((20,foot+47*k),text,font=font(31),fill=(35,45,55))
    dest=Path(output)/(job['id']+'.png');assert not dest.exists(),'Do not overwrite earlier render'
    board.save(dest,compress_level=4)
    del board
    with Image.open(dest) as checked:
        assert checked.format=='PNG' and checked.size==BOARD_SIZE
        for box,array in zip(PANEL_BOXES,panels):
            assert np.array_equal(np.asarray(checked.crop(tuple(box))),array),'Saved native panel pixels changed'
    return dict(id=job['id'],source=job['image_path'],source_sha256=job['image_sha256'],source_group=ctx['groups'][job['id']],
                source_use=use,source_use_label=use_label,output=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,
                size=list(BOARD_SIZE),image_viewports_xyxy=PANEL_BOXES,neutral_rgb=[45,45,45],
                record_gzip=str(record_file),record_path=str(record_file),record_sha256=sha(record_file),selected_entry=ei,selected_candidate=ci,
                displayed_candidates=shown,entry_count=len(entries),conditions=conditions,
                selected_response_file=selected_condition['response_file'],selected_response_sha256=selected_condition['response_sha256'],
                response_path=selected_condition['response_file'],response_sha256=selected_condition['response_sha256'],
                U1_model=model,U1_step=800,heat_native_shape_hw=[975,2048],heat_crop_original_xyxy=[0,1050,4096,3000],
                fixed_heat_scale=[0,1],lut_rgb_sha256=hashlib.sha256(ctx['lut'].tobytes()).hexdigest(),
                left_pixels_identical=True,raw_heat_body_unmodified=True,candidate_raster_matches_record=True,final_raster_matches_record=True,
                selected_kind=best['kind'] if best else None,ambiguity=ambiguous,complete_identity_verified=False,
                no_B_response_or_path_used=True,no_R1_used=True,no_labels_read=True,seconds=time.perf_counter()-tick)

def numerical_check():
    a=np.array([[0.,.1],[.8,1.]],np.float32);u=up2(a)
    assert u.shape==(4,4) and u[0,0]==0 and u[-1,-1]==1
    assert abs(float(u[1,1])-.23125)<1e-6
    p=original_points([[0,0],[1.25,2.5]])
    assert np.array_equal(p,[[0,1050],[2,1055]])
    assert PANEL_BOXES==[[0,180,4096,3180],[4128,180,8224,3180],[8256,180,12352,3180],[12384,180,16480,3180]]
    lut=official_lut();assert lut.shape==(256,3) and lut.dtype==np.uint8
    dump(HERE/'render_numerical_checks.json',dict(passed=True,coordinate_xy_half_to_original=True,
        x2_align_corners_false=True,native_panel_boxes=PANEL_BOXES,board_size=BOARD_SIZE,
        LUT='OpenCV official TURBO RGB, rint to256 bins',no_user_images_or_labels_read=True,source_sha256=sha(__file__)))
    print('Renderer numerical checks passed',flush=True)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='all_001',choices=['pilot_001','all_001'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:numerical_check();return
    run=BULK/args.run;summary=load(run/'summary.json');assert summary['exit_code']==0
    task=load(HERE/'TASK.json');ids=[r['id'] for r in summary['rows']]
    assert len(set(ids))==len(ids)
    if args.run=='all_001':assert len(ids)==292
    output=run/'四栏检测结果';output.mkdir(exist_ok=False)
    records=[run/'记录'/(id+'.json.gz') for id in ids]
    assert all(p.exists() for p in records)
    started=time.perf_counter();rows=[];errors=[]
    receipt_name='render_receipt.json' if args.run=='all_001' else 'render_pilot_receipt.json'
    log=HERE/('render_progress.jsonl' if args.run=='all_001' else 'render_pilot_progress.jsonl')
    with log.open('x',encoding='utf-8') as f,concurrent.futures.ProcessPoolExecutor(args.workers,initializer=worker_init) as ex:
        pending={ex.submit(render_one,p,output):p.stem for p in records}
        for future in concurrent.futures.as_completed(pending):
            try:
                row=future.result();rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            except Exception:errors.append(dict(id=pending[future],error=traceback.format_exc()))
            if (len(rows)+len(errors))%12==0 or len(rows)+len(errors)==len(records):
                print(dict(rendered=len(rows),errors=len(errors),planned=len(records),seconds=round(time.perf_counter()-started,1)),flush=True)
    receipt=dict(exit_code=int(bool(errors)),command=sys.argv,rows=rows,errors=errors,planned=len(records),seconds=time.perf_counter()-started,
                 output_directory=str(output),source_sha256=sha(__file__),task_sha256=sha(HERE/'TASK.json'),
                 calculation_summary_sha256=sha(run/'summary.json'),four_native_viewports=True,viewport_size=[WIDTH,HEIGHT],
                 image_viewports_xyxy=PANEL_BOXES,board_size=list(BOARD_SIZE),entry_conditions=sum(r['entry_count'] for r in rows),
                 source_use_counts=dict(Counter(r['source_use'] for r in rows)),raw_U1_heat_unmodified=True,original_P1_baseline_unchanged=True,
                 full_identity_accuracy=None)
    dump(HERE/receipt_name,receipt)
    if errors:raise RuntimeError('Render errors preserved: '+str(errors[:1]))

if __name__=='__main__':main()
