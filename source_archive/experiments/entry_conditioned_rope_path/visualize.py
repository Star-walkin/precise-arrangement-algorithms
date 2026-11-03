"""Fixed-scale evidence with the existing Pillow/OpenCV environment.

Decode only the prepared 18-image allowlist. Never load evaluation annotations;
optional reference overlays come from completed independent evaluation JSON.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

STAGE_COLORS={1:'#38a9ff',2:'#ffb627',3:'#44ee85',4:'#c998ff'}
CANDIDATE_COLORS=['#00f5d4','#ff9f1c','#ff4da6']
FONT='C:/Windows/Fonts/msyh.ttc'
POLICY={'gray_display':'original grayscale, fixed0..255; no per-image stretch',
 'heatmap_display':'response/target fixed0..1 with colorbar; NaN darkgray',
 'unknown_supervision_display':'gray, never zero target',
 'phase_colors':'visual scene stages only; not confirmed contact or identity',
 'algorithm_status':'visible_supported is an image-support proxy, not correct identity',
 'zoom_size_original_px':640,
 'zoom_selection':'same frozen data/scenes windows across modes for comparison only, never AUTO detector input; contact proxy, midpoint and far interface; missing interface uses fixed context',
 'evaluation_labels':'not loaded; optional overlay uses post-prediction evaluation artifact only',
 'dependency_note':'reuse project Pillow/OpenCV; missing matplotlib was not installed'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_new(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def text(im,xy,s,size=18,color='#172b3e'):
    ImageDraw.Draw(im).text(xy,s,font=ImageFont.truetype(FONT,size),fill=color)
def heat(a,valid=None):
    ok=np.isfinite(a)
    if valid is not None:ok&=valid
    h=cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(np.nan_to_num(a),0,1)*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    h[~ok]=[50,45,56]
    return h
def colorbar(im,xy,width=300):
    x,y=xy;im.paste(Image.fromarray(heat(np.tile(np.linspace(0,1,width),(14,1)))),(x,y))
    for v in [0,.25,.5,.75,1]:text(im,(x+int(v*(width-18)),y+16),f'{v:g}',13)
    text(im,(x+width+12,y-2),'响应强度 0–1；非真实概率',15)
def panel(gray,box,title,size=(1050,580),array=None,heatmap=False,valid=None):
    x0,y0,x1,y1=map(int,box);w,h=size;im=Image.new('RGB',size,'#f3f6fa');text(im,(8,5),title,18)
    a=gray[y0:y1,x0:x1] if array is None else array[y0:y1,x0:x1]
    if heatmap:a=heat(a,None if valid is None else valid[y0:y1,x0:x1])
    if a.ndim==2:a=np.repeat(a[...,None],3,axis=2)
    a=Image.fromarray(a.astype(np.uint8));scale=min((w-16)/a.width,(h-100)/a.height)
    tw,th=round(a.width*scale),round(a.height*scale);ox,oy=8,39
    im.paste(a.resize((tw,th),Image.Resampling.NEAREST),(ox,oy))
    text(im,(8,h-58),f'原图 xy: [{x0},{y0}] → [{x1},{y1}]',13)
    if heatmap:colorbar(im,(8,h-36),min(300,w-240))
    return {'im':im,'box':box,'scale':scale,'origin':(ox,oy),'content_bottom':oy+th}
def coords(p,points):
    return (np.asarray(points,float)-np.asarray(p['box'][:2]))*p['scale']+np.asarray(p['origin'])
def line(p,points,color,width=2,dashed=False):
    points=np.asarray(points,float)
    if len(points)<2:return
    d=ImageDraw.Draw(p['im'])
    x0,y0,x1,y1=map(int,p['box'])
    for aa,bb in zip(points[:-1],points[1:]):
        ok,a,b=cv2.clipLine((x0,y0,x1-x0,y1-y0),tuple(np.rint(aa).astype(int)),tuple(np.rint(bb).astype(int)))
        if not ok:continue
        a,b=coords(p,[a,b])
        if dashed:
            n=max(1,int(np.ceil(np.linalg.norm(b-a)/4)))
            for i in range(0,n,2):
                aa=a+(b-a)*i/n;bb=a+(b-a)*min(i+1,n)/n
                d.line([tuple(aa),tuple(bb)],fill=color,width=width)
        else:d.line([tuple(a),tuple(b)],fill=color,width=width)
def marker(p,point,color,r=4):
    x0,y0,x1,y1=p['box']
    if not (x0<=point[0]<x1 and y0<=point[1]<y1):return
    x,y=coords(p,[point])[0];ImageDraw.Draw(p['im']).ellipse([x-r,y-r,x+r,y+r],outline=color,width=2)
def scene_lines(p,scene):
    for s in scene.get('edge_segments_xy',[]):line(p,s,'#e8ec76',1)
    for q in scene.get('target_edge_points_xy',[])[::5]:marker(p,q,'white',1)
    c=scene.get('contact_proxy',{}).get('xy')
    if c is not None:marker(p,c,'#ffb627',5)
def pool(p,prediction):
    selected=prediction.get('selected_candidate_id')
    for k,c in enumerate(prediction['candidates']):
        yes=c['id']==selected;line(p,c['points_xy'],CANDIDATE_COLORS[k],3 if yes else 2,not yes)
        text(p['im'],(15,65+k*22),c['id']+(' 已选' if yes else ' 未选'),15,CANDIDATE_COLORS[k])
    if selected is None:text(p['im'],(15,43),'拒识：无已选路径；彩线仅为候选',16,'#ff4d4d')
def status(p,prediction):
    selected=prediction.get('selected_candidate_id');cs=prediction['candidates']
    c=next((c for c in cs if c['id']==selected),None);unselected=c is None
    if c is None and cs:c=cs[0]
    if c is None:text(p['im'],(15,45),'无路径候选',17,'#ff4d4d');return
    xy=c['points_xy'];ss=c['stage'];states=c['point_status']
    for i,(a,b) in enumerate(zip(xy[:-1],xy[1:])):
        color=STAGE_COLORS[int(ss[i])];dash=unselected
        if 'entry_condition' in states[i:i+2]:color='#ffff00';dash=True
        elif 'ambiguous' in states[i:i+2]:color='#ff3030';dash=True
        elif 'inferred' in states[i:i+2]:color='#fff275';dash=True
        line(p,[a,b],color,2,dash)
    marker(p,xy[0],'white');marker(p,xy[-1],'white')
    if unselected:text(p['im'],(15,43),'c0 诊断；仍为拒识，非已选结果',16,'#ff4d4d')
def reference(p,evaluation):
    if evaluation:
        for s in evaluation['source_reference']['trusted_clipped_centerline']:line(p,s,'white',1,True)
def get_seed(item,mode):
    if mode=='CONDITIONAL':return np.asarray(item['known_seed_xy'],float),True
    e=read(item['auto_entry_path']);return np.asarray(e.get('free_seed',[]),float).reshape(-1,2),bool(e['available'])
def get_scene(item,prediction,mode):
    s=prediction.get('edge_target')
    if isinstance(s,dict) and 'contact_proxy' in s:return s
    if mode=='AUTO':
        e=read(item['auto_entry_path'])
        if e.get('scene_path'):return read(e['scene_path'])
    return read(item['scene_path'])
def box_at(c,shape,size=640):
    h,w=shape;x,y=c;x0=max(0,min(w-size,int(round(x-size/2))));y0=max(0,min(h-size,int(round(y-size/2))))
    return [x0,y0,min(w,x0+size),min(h,y0+size)]
def zooms_for(scene,item):
    seed=np.asarray(scene['entry_short_xy']);direction=np.asarray(scene['entry_direction_xy'])
    c=scene.get('contact_proxy',{}).get('xy');known=c is not None;c=np.asarray(c) if known else seed[-1]+300*direction
    edge=np.asarray(scene.get('target_edge_points_xy',[]));distal=np.median(edge,axis=0) if len(edge) else c+900*direction
    return [{'name':'过渡视觉代理' if known else '过渡未知：固定上下文','box':box_at(c,item['shape'])},
      {'name':'贴面侧视觉上下文','box':box_at((c+distal)*.5,item['shape'])},
      {'name':'已观测远侧界面' if len(edge) else '边缘未知：固定上下文','box':box_at(distal,item['shape'])}]
def maps(item):
    h,w=item['shape'];x0,y0,x1,y1=item['bbox'];t=np.zeros((h,w),np.float32);v=np.zeros((h,w),bool)
    if item['split']=='train':
        with np.load(item['cache'],allow_pickle=False) as a:
            t[y0:y1,x0:x1]=cv2.resize(a['target'],(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
            v[y0:y1,x0:x1]=cv2.resize(a['valid'].astype(np.uint8),(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST).astype(bool)
    return t,v
def target_panel(gray,box,t,v,item,size):
    if item['split']=='train':return panel(gray,box,'训练中心目标 / 监督域；灰色 = ignore',size,t,True,v)
    p=panel(gray,box,'评价图：未向训练提供目标或有效标签',size)
    text(p['im'],(15,45),'无评价目标进入训练；预测后独立评分',16,'#ffdf6b');return p
def visualize_case(run,item,mode,output):
    folder=run/'predictions'/mode/item['id'];pf=folder/'paths.json';rf=folder/'response.npy';pred=read(pf)
    response=np.load(rf,mmap_mode='r',allow_pickle=False);assert response.shape==tuple(item['shape'])
    assert sha(item['image_path'])==item['image_sha256']
    gray=cv2.imdecode(np.fromfile(item['image_path'],np.uint8),cv2.IMREAD_GRAYSCALE);assert gray.shape==tuple(item['shape'])
    scene=get_scene(item,pred,mode);seed,available=get_seed(item,mode);t,v=maps(item);box=item['bbox']
    zooms=zooms_for(read(item['scene_path']),item)
    ef=run/'evaluation'/mode/(item['id']+'.json');ev=read(ef) if ef.is_file() else None
    dest=output/mode/item['id'];dest.mkdir(parents=True,exist_ok=False)
    board=Image.new('RGB',(2160,2490),'#e8eef5');text(board,(20,10),item['id']+' | '+mode+' | '+item['split'],23)
    text(board,(20,48),'完整搜索 ROI；原始灰度固定 0–255；视觉阶段/支持代理不等于真实贴面或同根身份',18)
    panels=[];p=panel(gray,box,'原始灰度 + 冻结局部窗口');panels.append(p)
    for z in zooms:
        x0,y0,x1,y1=z['box'];line(p,[[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]],'#9ed9e2',1)
    p=panel(gray,box,'仅初始自由入口提示' if available else 'AUTO 入口失败：无提示');panels.append(p)
    if len(seed):line(p,seed,'yellow',3);marker(p,seed[0],'yellow')
    scene_lines(p,scene);panels.append(target_panel(gray,box,t,v,item,(1050,580)))
    panels.append(panel(gray,box,'原始网络响应：未进行路径连接',(1050,580),response,True))
    p=panel(gray,box,'最多 3 条候选；不改变保存的选择');pool(p,pred);scene_lines(p,scene);panels.append(p)
    p=panel(gray,box,'阶段与歧义；白色虚线 = 预测后局部参考');status(p,pred);reference(p,ev);scene_lines(p,scene);panels.append(p)
    initial_file=folder/'initial_response.npy'
    if initial_file.is_file():
        initial=np.load(initial_file,mmap_mode='r',allow_pickle=False)
        panels.append(panel(gray,box,'未训练基线：同一初始权重 / 固定 0–1',(1050,580),initial,True))
    else:
        panels.append(panel(gray,box,'未训练基线未保存；不能推断其响应'))
    color=heat(response);overlay=(.55*np.repeat(gray[...,None],3,axis=2)+.45*color).astype(np.uint8)
    overlay[~np.isfinite(response)]=np.repeat(gray[...,None],3,axis=2)[~np.isfinite(response)]
    panels.append(panel(gray,box,'原始响应叠加：原灰度 55% + 固定色标 45%',(1050,580),overlay))
    for k,p in enumerate(panels):board.paste(p['im'],(20+(k%2)*1070,90+(k//2)*590))
    text(board,(20,2450),'亮黄 输入入口（不计识别）| 蓝 自由 | 橙 过渡代理 | 绿 贴面侧代理 | 紫 边缘邻域 | 淡黄 推断 | 红虚线 歧义',16)
    board.save(dest/'full_roi.png')
    local=Image.new('RGB',(2180,2110),'#e8eef5');text(local,(20,10),item['id']+' | '+mode+' | 固定 640×640 原图坐标窗口',21)
    text(local,(20,48),'所有热图固定 0–1；未知为未知；白色虚线仅为预测后独立局部参考',18)
    for r,z in enumerate(zooms):
        b=z['box'];size=(525,640)
        ps=[panel(gray,b,z['name']+'：原图',size),target_panel(gray,b,t,v,item,size),panel(gray,b,'原始响应 0–1',size,response,True)]
        p=panel(gray,b,'候选 / 阶段与歧义 / 参考',size);pool(p,pred);status(p,pred);reference(p,ev);scene_lines(p,scene);ps.append(p)
        for col,p in enumerate(ps):local.paste(p['im'],(20+col*540,90+r*660))
    text(local,(20,2070),'远侧界面只是场景候选；碰到边缘不代表末端身份正确。入口失败的 AUTO 不显示条件入口。',17)
    local.save(dest/'local_contexts.png')
    return {'id':item['id'],'mode':mode,'split':item['split'],'entry_available':available,
      'prediction_paths_sha256':sha(pf),'response_sha256':sha(rf),'selected_candidate_id':pred.get('selected_candidate_id'),
      'candidate_count':len(pred['candidates']),'bbox':box,'zooms':zooms,'posthoc_reference_shown':ev is not None,
      'files':[str(dest/'full_roi.png'),str(dest/'local_contexts.png')]}
def main():
    pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('--run',type=Path,required=True);pa.add_argument('--output',type=Path)
    pa.add_argument('--modes',nargs='+',choices=('CONDITIONAL','AUTO'),default=['CONDITIONAL','AUTO']);args=pa.parse_args()
    started=time.time();cv2.setNumThreads(1);run=args.run.resolve();index=read(run/'data/index.json');assert len(index['items'])==18
    assert all('20260925' not in i['image_path'] for i in index['items'])
    output=args.output.resolve() if args.output else run/'visualizations'
    if output.exists():raise FileExistsError('Preserve existing evidence; choose new output directory: '+str(output))
    cases=[(i,m) for m in args.modes for i in index['items'] if m=='CONDITIONAL' or i['split']=='eval']
    for i,m in cases:
        folder=run/'predictions'/m/i['id']
        if not (folder/'paths.json').is_file() or not (folder/'response.npy').is_file():raise FileNotFoundError('Wait for all predictions: '+str(folder))
    rows=[]
    for i,m in cases:
        rows.append(visualize_case(run,i,m,output));print(json.dumps({'visualized':i['id'],'mode':m}),flush=True)
    write_new(output/'manifest.json',{'status':'completed','exit_code':0,'command':[sys.executable,*sys.argv],
      'elapsed_seconds':time.time()-started,'visualizer_sha256':sha(Path(__file__)),'policy':POLICY,'items':rows})
    print(json.dumps({'status':'completed','cases':len(rows),'output':str(output)}),flush=True)
if __name__=='__main__':main()
