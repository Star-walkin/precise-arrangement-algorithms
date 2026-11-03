"""Reuse saved arrays to audit response semantics, ROI edges and normal profiles."""
from pathlib import Path
import argparse,json,sys
import cv2,numpy as np
from PIL import Image,ImageDraw
from visualize import panel,line,text,colorbar,sha

HERE=Path(__file__).resolve().parent
OLD=HERE/'runs/run_20261029_001'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sample(a,xy,method=cv2.INTER_LINEAR):
    p=np.asarray(xy,np.float32)
    return cv2.remap(a,p[:,0][None],p[:,1][None],method,borderMode=cv2.BORDER_REPLICATE).ravel()
def full(a,bbox,shape,nearest=False):
    out=np.zeros(shape,np.float32);x0,y0,x1,y1=bbox
    out[y0:y1,x0:x1]=cv2.resize(a.astype(np.float32),(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    return out
def charts(board,xy,series,valid,origin,size):
    ox,oy=origin;w,h=size;draw=ImageDraw.Draw(board);margin=45
    for v in np.linspace(0,1,5):
        yy=oy+h-margin-v*(h-2*margin);draw.line([(ox+margin,yy),(ox+w-margin,yy)],fill='#ccd4de')
        text(board,(ox+4,yy-8),f'{v:.2f}',14)
    def pts(values):return [(ox+margin+(x+80)/160*(w-2*margin),oy+h-margin-v*(h-2*margin)) for x,v in zip(xy,values)]
    for key,color in [('gray','#404040'),('target','#2eae40'),('prediction','#e73232')]:
        pp=pts(series[key])
        for k in range(len(pp)-1):
            if key!='target' or (valid[k] and valid[k+1]):draw.line(pp[k:k+2],fill=color,width=3)
    draw.rectangle((ox+margin,oy+margin,ox+w-margin,oy+h-margin),outline='#3b4655')
    for x in [-80,-40,0,40,80]:text(board,(ox+margin+(x+80)/160*(w-2*margin)-15,oy+h-margin+8),str(x),14)
    text(board,(ox+60,oy+5),'灰：原灰度/255  绿：有效目标  红：原始响应；横轴原图法向px，纵轴固定0–1',18)

def main(run,extra=False):
    run=Path(run).resolve();run.mkdir(parents=True,exist_ok=True);dest=run/('audit_with_supervision' if extra else 'input_audit');dest.mkdir(exist_ok=False)
    index=load(OLD/'data/index.json');newindex=load(run/'data/index.json') if extra else None
    rows=[]
    for item in index['items']:
        folder=OLD/'predictions/CONDITIONAL'/item['id'];info=load(folder/'inference.json');p=np.load(folder/'response.npy',mmap_mode='r')
        initial=np.load(folder/'initial_response.npy',mmap_mode='r');x0,y0,x1,y1=item['bbox'];region=np.asarray(p[y0:y1,x0:x1])
        rows.append({'id':item['id'],'mode_in_metadata':info['mode'],'response_sha256':sha(folder/'response.npy'),
            'original_bbox':item['bbox'],'model_shape':[975,2048],'input_padded_shape':[992,2048],
            'padding_bottom_model_px':17,'top_original_rows_0_1_mean':float(region[:2,64:-64].mean()),
            'top_inside_rows_16_47_mean':float(region[16:48,64:-64].mean()),
            'bottom_original_rows_last2_mean':float(region[-2:,64:-64].mean()),
            'bottom_inside_rows_16_47_mean':float(region[-48:-16,64:-64].mean()),
            'initial_top2_mean':float(initial[y0:y0+2,64:-64].mean()),
            'initial_bottom2_mean':float(initial[y1-2:y1,64:-64].mean()),
            'nan_outside_search':bool(np.isnan(p[:y0]).all()),'finite_domain':bool(np.isfinite(region).all())})
    item=next(i for i in index['items'] if i['id']=='dual_20260924_111500__left__f00240')
    bgr=cv2.imdecode(np.fromfile(item['image_path'],np.uint8),cv2.IMREAD_COLOR);gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    with np.load(item['cache']) as z:a={k:z[k] for k in z.files}
    oldvalid=a['valid'].astype(bool);target=full(a['target'],item['bbox'],item['shape']);valid=full(oldvalid,item['bbox'],item['shape'],True)>0
    seed=full(a['seed'],item['bbox'],item['shape']);pred=np.load(OLD/'predictions/CONDITIONAL'/item['id']/'response.npy')
    classes=np.zeros(a['image'].shape,np.uint8);classes[oldvalid&(a['target']<.5)]=2;classes[oldvalid&(a['target']>=.5)]=1
    if extra:
        ni=next(i for i in newindex['items'] if i['id']==item['id'])
        with np.load(ni['supervision_cache']) as s:
            classes[s['neighbor_mask'].astype(bool)]=3;classes[s['background_mask'].astype(bool)]=4
    palette=np.array([[75,75,85],[50,230,70],[60,130,250],[250,110,30],[210,60,230]],np.uint8)
    classfull=full(classes,item['bbox'],item['shape'],True).astype(np.uint8);classrgb=palette[classfull]
    label=load(OLD/'data/label_sources'/f"{item['id']}.json");center=np.array(label['trusted_center_segments_xy'][0]);profiles=[]
    for tag,k in [('clear_free',2),('difficult_continuation',10)]:
        c=center[k];t=center[k+1]-center[k-1];t/=np.linalg.norm(t);normal=np.array([-t[1],t[0]])
        box=[int(c[0]-240),int(c[1]-240),int(c[0]+240),int(c[1]+240)]
        d=np.linspace(-80,80,321);points=c+normal*d[:,None]
        vv=sample(valid.astype(np.float32),points,cv2.INTER_NEAREST)>.5
        series={'gray':sample(gray.astype(np.float32)/255.,points),'target':sample(target,points),'prediction':sample(pred,points)}
        board=Image.new('RGB',(2180,1350),'#edf2f7');text(board,(20,10),tag+' | CONDITIONAL已知入口 | 原图同坐标 | '+item['id'],23)
        panels=[panel(gray,box,'原始灰度0–255',(525,610)),panel(gray,box,'监督类别（灰ignore）',(525,610),classrgb),
                panel(gray,box,'实际入口通道0–1；本局部外可为0',(525,610),seed,True),panel(gray,box,'旧模型原始响应0–1',(525,610),pred,True)]
        for j,p in enumerate(panels):line(p,points[[0,-1]],'#ffff00',2);board.paste(p['im'],(20+540*j,65))
        text(board,(20,685),'绿：中心高响应  蓝：原有低响应  橙：新增邻圈负例  紫：新增背景负例  灰：ignore',20)
        charts(board,d,series,vv,(20,720),(2140,390))
        overview=panel(gray,item['bbox'],'入口总览',(800,200));line(overview,item['known_seed_xy'],'yellow',3);board.paste(overview['im'],(20,1100))
        text(board,(850,1140),'中心σ=3原图px；未扩大高斯，未改变色标。标签在ignore处不画成0。',20)
        board.save(dest/(tag+'.png'))
        profiles.append({'name':tag,'center_xy':c.tolist(),'normal_xy':normal.tolist(),'box':box,'offsets_px':d.tolist(),
            'sample_xy':points.tolist(),'valid':vv.tolist(),'supervision_class':sample(classfull.astype(np.float32),points,cv2.INTER_NEAREST).astype(int).tolist(),
            **{k:v.tolist() for k,v in series.items()}})
    # Independent vertical scan of the SAVED raw array, no display colors.
    c=rows[0];q=np.load(OLD/'predictions/CONDITIONAL'/index['items'][0]['id']/'response.npy');vertical=np.nanmean(q[:,64:-64],axis=1)
    edgeplot=Image.new('RGB',(1500,600),'white');dr=ImageDraw.Draw(edgeplot)
    for value in np.arange(0,1.01,.2):
        yy=540-value*450;dr.line([(70,yy),(1450,yy)],fill='#dddddd');text(edgeplot,(15,yy-10),f'{value:.1f}',16)
    pp=[(70+y/3000*1380,540-float(v)*450) for y,v in enumerate(vertical) if np.isfinite(v)];dr.line(pp,fill='#e42c3a',width=2)
    for y in [1050,2999]:
        xx=70+y/3000*1380;dr.line([(xx,80),(xx,540)],fill='#235dba',width=2);text(edgeplot,(xx-60,550),str(y),16)
    text(edgeplot,(30,15),'保存的原始float响应：每行均值（去左右64px），x=原图y；蓝线为计算区上下沿',22)
    edgeplot.save(dest/'raw_boundary_rows.png')
    note={'exit_code':0,'command':sys.argv,'source_sha256':sha(__file__),'rows':rows,'profiles':profiles,
        'source_semantics':{'activation':'EntryUNet.forward sigmoid exactly once','response':'raw single center head, not Grad-CAM or path mask',
            'mapping':'remove padding975x2048 then bilinear align_corners=False to1950x4096; paste y1050..2999',
            'display':'fixed0..1; no per-image minmax; NaN search exterior displayed gray',
            'old_label':'sigma3 original px; <=9px center tube intersect reviewed sides; narrow outside side negative <=4px; quality clipped before raster',
            'seed':'initial120 original px; sigma4; same bbox/scale; original unknown zeros retained',
            'batch_292_mode':'AUTO saved A6 free entries (7 two-entry cases), with3 observed-free A2 fallbacks; different from fixed8 V28 AUTO',
            'fixed_experiment_modes':'read inference.json entry metadata; CONDITIONAL known entry, AUTO frozen V28 entry'},
        'decision':'ROI/padding edge response is present in raw arrays and cannot be fixed by colorbar changes. No coordinate/activation error found. Keep input, padding and mapping unchanged for both A/B; track edge artifact separately.'}
    dump(dest/'audit.json',note)
    print(json.dumps({'exit_code':0,'images':len(rows),'top2_mean':float(np.mean([r['top_original_rows_0_1_mean'] for r in rows])),
        'inside_top_mean':float(np.mean([r['top_inside_rows_16_47_mean'] for r in rows])),'bottom2_mean':float(np.mean([r['bottom_original_rows_last2_mean'] for r in rows])),
        'inside_bottom_mean':float(np.mean([r['bottom_inside_rows_16_47_mean'] for r in rows]))}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--after-data',action='store_true');a=p.parse_args();main(a.run,a.after_data)
