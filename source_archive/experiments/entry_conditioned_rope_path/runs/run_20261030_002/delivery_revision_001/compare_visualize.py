"""Fixed original/OLD/A/B scientific panels, using Pillow and OpenCV only."""
from pathlib import Path
import argparse,json,sys,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw

BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE))
import visualize as v
from freeze_evaluation_scope import interpolate,box
from compare_evaluate import sample_response

TRAIN_FIXED=['dual_20260924_111500__left__f00240','dual_20260924_111143__right__f00130']
COLORS={'OLD':'#f29b38','A':'#2299ff','B':'#f04a55','gray':'#303540','target':'#38aa69','seed':'#b45add'}
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def source_gray(item):
    assert v.sha(item['image_path'])==item['image_sha256']
    return cv2.imdecode(np.fromfile(item['image_path'],np.uint8),cv2.IMREAD_GRAYSCALE)
def draw_scope(p,scope):
    for s in scope['quality_clipped_center_segments_xy']:v.line(p,s,'yellow',1)
    for r in scope['background_regions']:
        q=r['polygon_xy'];v.line(p,q+[q[0]],'#ff595e',2)
    for pair in scope['target_neighbor_pairs']:v.line(p,pair['neighbor_points_xy'],'cyan',2)
def place(board,panels,start_y,col_width=540,gap=10):
    for j,p in enumerate(panels):board.paste(p['im'],(10+j*(col_width+gap),start_y))
def load_predictions(roots,mode,sid):
    return {key:{'response':np.load(root/'predictions'/mode/sid/'response.npy',mmap_mode='r',allow_pickle=False),
                 'paths':read(root/'predictions'/mode/sid/'paths.json')} for key,root in roots.items()}
def render_eval(item,scope,mode,roots,dest):
    gray=source_gray(item);pred=load_predictions(roots,mode,item['id']);bbox=item['bbox'];size=(540,390)
    board=Image.new('RGB',(2210,920),'#e6edf4');v.text(board,(12,10),item['id']+' | '+mode+' | OLD起点 / A原监督续训 / B监督改进',21)
    v.text(board,(12,45),'主比较 A vs B；灰度0–255，响应0–1；评价参考/工程代理不进入训练',17)
    panels=[v.panel(gray,bbox,'原图：完整搜索ROI',size)]
    for key in ['OLD','A','B']:panels.append(v.panel(gray,bbox,key+' 原始响应：固定0–1',size,pred[key]['response'],True))
    place(board,panels,80)
    p=v.panel(gray,bbox,'仅评价：黄 reviewed目标 / 红背景proxy / 青邻圈proxy',size);draw_scope(p,scope);panels=[p]
    for key in ['OLD','A','B']:
        p=v.panel(gray,bbox,key+' 保存的候选与选择',size);v.pool(p,pred[key]['paths']);v.status(p,pred[key]['paths']);panels.append(p)
    place(board,panels,490);v.text(board,(12,890),'阶段及图像支持不是物理接触或同根身份。已给入口不计新识别，拒识继续保留。',16)
    folder=dest/mode/item['id'];folder.mkdir(parents=True,exist_ok=False);board.save(folder/'full_comparison.png')
    size=(540,630);local=Image.new('RGB',(2210,2730),'#e6edf4')
    v.text(local,(12,10),item['id']+' | '+mode+' | 冻结的自由/接近/后段/背景窗口',20)
    v.text(local,(12,42),'全部模型与模式使用同一原图坐标；每张热图固定0–1；细线仅为保存的候选路径',17)
    for row,z in enumerate(scope['zooms']):
        b=z['bbox'];p=v.panel(gray,b,z['id']+' 原图/评价参考',size);draw_scope(p,scope);panels=[p]
        for key in ['OLD','A','B']:
            p=v.panel(gray,b,key+' 原始响应 + 保存路径',size,pred[key]['response'],True)
            # Keep all candidates separate; selected solid, rejected dashed.
            v.pool(p,pred[key]['paths']);panels.append(p)
        place(local,panels,80+row*650)
    v.text(local,(12,2690),'黄线为旧reviewed中心；青邻绳和红背景为冻结工程代理，不能称为独立身份真值。',16)
    local.save(folder/'fixed_local_comparison.png')
    return [str(folder/'full_comparison.png'),str(folder/'fixed_local_comparison.png')]
def full_map(a,item,linear=False):
    out=np.zeros(item['shape'],np.float32);x0,y0,x1,y1=item['bbox']
    out[y0:y1,x0:x1]=cv2.resize(np.asarray(a,np.float32),(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR if linear else cv2.INTER_NEAREST)
    return out
def chart(board,xy,size,series,title,valid=None):
    x,y=xy;w,h=size;d=ImageDraw.Draw(board);v.text(board,(x,y),title,18)
    left,top,right,bottom=x+40,y+35,x+w-20,y+h-35
    if valid is not None:
        for i,ok in enumerate(valid[:-1]):
            if not ok:d.rectangle((left+i*(right-left)/(len(valid)-1),top,left+(i+1)*(right-left)/(len(valid)-1),bottom),fill='#ded8e4')
    for q in [0,.5,1]:
        yy=bottom-q*(bottom-top);d.line([(left,yy),(right,yy)],fill='#b5c2cf');v.text(board,(x,int(yy)-8),str(q),13)
    for name,a,color in series:
        a=np.asarray(a);run=[]
        for xx,value in zip(np.linspace(left,right,len(a)),a):
            if np.isfinite(value):run.append((xx,bottom-float(np.clip(value,0,1))*(bottom-top)))
            else:
                if len(run)>1:d.line(run,fill=color,width=2)
                run=[]
        if len(run)>1:d.line(run,fill=color,width=2)
    for offset in [-60,0,60]:v.text(board,(int(left+(offset+60)/120*(right-left))-12,bottom+6),str(offset),13)
    lx=left+100
    for name,a,color in series:v.text(board,(lx,y+2),name,15,color);lx+=110
def render_train(item,run,roots,dest):
    gray=source_gray(item);pred=load_predictions(roots,'CONDITIONAL',item['id'])
    with np.load(item['cache'],allow_pickle=False) as z:old={k:z[k] for k in z.files}
    with np.load(item['supervision_cache'],allow_pickle=False) as z:sup={k:z[k] for k in z.files}
    target=full_map(sup['center_target'],item,True);valid=full_map(sup['center_valid'],item)>.5
    neighbor=full_map(sup['neighbor_mask'],item)>.5;background=full_map(sup['background_mask'],item)>.5;seed=full_map(old['seed'],item,True)
    classes=np.full((*gray.shape,3),[50,45,56],np.uint8);classes[valid&(target<.5)]=[55,110,220];classes[valid&(target>=.5)]=[55,210,120]
    classes[neighbor]=[190,70,220];classes[background]=[245,105,45]
    sources=read(Path(roots['OLD'])/'data/label_sources'/(item['id']+'.json'));segments=sources['trusted_center_segments_xy'];line=max(segments,key=lambda p:np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
    board=Image.new('RGB',(2510,1780),'#e6edf4');v.text(board,(12,10),item['id']+' | 训练实际监督、入口及同坐标法向剖面',22)
    v.text(board,(12,45),'绿 中心>=0.5 / 蓝 原低响应 / 紫 新邻圈负例 / 橙 新背景负例 / 灰 ignore；没有新增正例',17)
    profile_records=[]
    for row,(name,fraction) in enumerate([('free_fixed_reference',.25),('continuation_fixed_reference',.65)]):
        short=interpolate(line,fraction,40);center=short[len(short)//2];tangent=short[-1]-short[0];tangent/=np.linalg.norm(tangent);normal=np.array([-tangent[1],tangent[0]])
        points=center+np.linspace(-60,60,241)[:,None]*normal;b=box(center,item['shape'],400);size=(405,510)
        panels=[v.panel(gray,b,name+' 原图',size),v.panel(gray,b,'实际入口通道：fixed0–1',size,seed,True),v.panel(gray,b,'实际B监督类别（原中心未改）',size,classes)]
        for key in ['OLD','A','B']:panels.append(v.panel(gray,b,key+'原始响应0–1',size,pred[key]['response'],True))
        for p in panels:v.line(p,[points[0],points[-1]],'white',1)
        for j,p in enumerate(panels):board.paste(p['im'],(12+j*415,80+row*830))
        center_valid=sample_response(valid.astype(np.float32),points)>.5
        neighbor_valid=sample_response(neighbor.astype(np.float32),points)>.5
        background_valid=sample_response(background.astype(np.float32),points)>.5
        center_values=sample_response(target,points);center_values[~center_valid]=np.nan
        series=[('gray/255',sample_response(gray.astype(np.float32)/255,points),COLORS['gray']),
          ('center',center_values,COLORS['target']),
          ('neighbor0',np.where(neighbor_valid,0.,np.nan),'#be46dc'),
          ('background0',np.where(background_valid,0.,np.nan),'#f5692d'),
          ('seed',sample_response(seed,points),COLORS['seed'])]
        series += [(key,sample_response(pred[key]['response'],points),COLORS[key]) for key in ['OLD','A','B']]
        supervised=center_valid|neighbor_valid|background_valid
        chart(board,(15,600+row*830),(2470,280),series,name+' 法向：原图px / 固定0–1；灰色为直接监督未知',supervised)
        profile_records.append({'name':name,'center_xy':center.tolist(),'points_xy':points.tolist(),'series':{n:[float(x) if np.isfinite(x) else None for x in a] for n,a,c in series},'direct_supervision_valid':supervised.tolist(),
          'supervision_class_at_samples':{'center':center_valid.tolist(),'neighbor':neighbor_valid.tolist(),'background':background_valid.tolist()},
          'unknown_target_policy':'center target NaN outside center valid; neighbor/background zero only inside their own masks; unknown remains line break'})
    folder=dest/'training_checks';folder.mkdir(exist_ok=True);file=folder/(item['id']+'.png');board.save(file)
    save(folder/(item['id']+'.json'),{'scope':'training diagnostic only, not independent evaluation','profiles':profile_records,'new_positive_length_px':0,'source_supervision_cache':item['supervision_cache'],'source_sha256':v.sha(item['supervision_cache'])})
    return str(file)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run).resolve();cv2.setNumThreads(1)
    scope=read(run/'evaluation_scope/scope.json');assert v.sha(run/'evaluation_scope/scope.json')==read(run/'evaluation_scope/freeze_receipt.json')['scope_sha256']
    assert (run/'comparison/evaluation.json').is_file(),'Evaluate only after all prediction pairs are saved'
    index={i['id']:i for i in read(run/'data/index.json')['items']};roots={'OLD':Path(scope['old_run']),'A':run/'branches/A','B':run/'branches/B'}
    dest=run/'comparison/visualizations';dest.mkdir(exist_ok=False);files=[]
    for mode in ['CONDITIONAL','AUTO']:
        for it in scope['items']:
            files+=render_eval(index[it['sample_id']],it,mode,roots,dest);print(json.dumps({'visualized':it['sample_id'],'mode':mode}),flush=True)
    for sid in TRAIN_FIXED:files.append(render_train(index[sid],run,roots,dest))
    for f in files:
        with Image.open(f) as im:im.verify()
    save(run/'comparison/visualization_receipt.json',{'status':'completed','exit_code':0,'files':files,'png_count':len(files),'verified_decode':True,
      'command':sys.argv,'source_sha256':v.sha(__file__),'scale':{'original':[0,255],'response':[0,1]},'fixed_scope_sha256':v.sha(run/'evaluation_scope/scope.json'),
      'training_checks':TRAIN_FIXED,'evaluation_images':8,'evaluation_groups':3,'modes_not_independent_samples':True})
    print(json.dumps({'status':'completed','png_count':len(files)}))
if __name__=='__main__':main()
