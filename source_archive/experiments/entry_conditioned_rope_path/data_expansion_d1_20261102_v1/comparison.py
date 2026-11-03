"""Fixed A750/D0/D1 diagnostic pictures, never used to choose checkpoints."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
Image.MAX_IMAGE_PIXELS=180_000_000
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def font(n):return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
from render_export import up2
SELECTION=[
 {'id':'dual_20260924_111500__left__f00240','split':'old_train','crop':[900,1850,2500,2450]},
 {'id':'dual_20260924_111143__right__f00130','split':'old_train','crop':[1200,1850,2800,2450]},
 {'id':'dual_20260923_113305__right__f00500','split':'eval','crop':[1400,1800,3000,2400]},
 {'id':'dual_20260923_113745__left__f00450','split':'eval','crop':[900,1850,2500,2450]},
 {'id':'L_run_20260923_01__f000585__left','split':'new_train_if_accepted','crop':[800,2160,2400,2760]},
 {'id':'R_run_20260923_01__f000572__right','split':'new_train_if_accepted','crop':[1000,1850,2600,2450]}]
def freeze():
    p=HERE/'comparison_selection.json';value={'items':SELECTION,'selection_basis':'fixed before formal D0/D1 inference:2oldtrain,2oldopen-eval,2candidate-new difficult appearances; no result-based selection',
        'response_scale':[0,1],'not_used_for_checkpoint_selection':True}
    if p.exists():assert load(p)==value
    else:dump(p,value)
def sample(full,p):
    # Independent bilinear float sample; points strictly inside image.
    p=np.asarray(p);x=p[:,0];y=p[:,1];xi=np.floor(x).astype(int);yi=np.floor(y).astype(int)
    dx=x-xi;dy=y-yi;w=full.shape[1];h=full.shape[0]
    xi=xi.clip(0,w-1);yi=yi.clip(0,h-1);xj=(xi+1).clip(0,w-1);yj=(yi+1).clip(0,h-1)
    return (full[yi,xi]*(1-dx)+full[yi,xj]*dx)*(1-dy)+(full[yj,xi]*(1-dx)+full[yj,xj]*dx)*dy
def mapped_npz(p):
    with np.load(p,allow_pickle=False) as z:a=z['response'].copy()
    full=np.full((3000,4096),np.nan,np.float32);full[1050:]=up2(a);return full
def chart(ident,points,values,path):
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1);s=np.r_[0,np.cumsum(lengths)];assert s[-1]>0
    im=Image.new('RGB',(1800,820),'white');d=ImageDraw.Draw(im);l,t,r,b=115,110,1740,605
    d.text((30,20),'固定中心剖面  '+ident,font=font(34),fill='black')
    def xy(x,y):return l+(r-l)*x/s[-1],b-(b-t)*y
    for v in np.linspace(0,1,6):
        y=xy(0,v)[1];d.line((l,y,r,y),fill='#dddddd',width=2);d.text((28,y-15),f'{v:.1f}',font=font(25),fill='black')
    for v in np.linspace(0,s[-1],6):
        x=xy(v,0)[0];d.text((x-20,b+12),f'{v:.0f}',font=font(25),fill='black')
    d.line((l,t,l,b,r,b),fill='black',width=3)
    for label,color,lx in [('A750','#444444',220),('D0','#315fbb',650),('D1','#c54132',1100)]:
        y=np.asarray(values[label]);mask=np.isfinite(y)
        pp=[xy(float(x),float(v)) for x,v in zip(s[mask],y[mask])]
        if len(pp)>1:d.line(pp,fill=color,width=4)
        d.line((lx,750,lx+55,750),fill=color,width=5);d.text((lx+65,730),label,font=font(29),fill='black')
    d.text((600,655),'沿固定训练中心线距离 原图px',font=font(28),fill='black');im.save(path)
def main():
    freeze();assert (HERE/'evaluation_paths/prediction_save_gate.json').exists(),'All32 eval predictions must be frozen before these diagnostics'
    out=HERE/'diagnostics_comparison';out.mkdir(exist_ok=False);sel=load(HERE/'selected_checkpoints.json')
    old=load(BASE/'runs/run_20261030_002/data/index.json');dataset=load(HERE/'dataset/index.json')
    ix={i['id']:i for i in old['items']};ix.update({i['id']:i for i in dataset['items']});lut=np.load(BASE/'batch_ab_20261030_001/color_lut.npy');rows=[]
    for choice in SELECTION:
        ident=choice['id']
        if ident not in ix:rows.append({'id':ident,'status':'not_accepted_new_label_no_training_item','selection_unchanged':True});continue
        it=ix[ident];files={};maps={}
        if choice['split'].startswith('new_train'):
            rp=BASE/'relation_r1_20261101/cache/gallery'/ident/'entry_00.npz';maps['A750']=mapped_npz(rp);files['A750']={'path':str(rp),'sha256':sha(rp),'model':'frozen A750 cached prior forward, no feature replay as new training'}
        else:
            rp=BASE/'runs/run_20261030_002/branches/A/predictions/CONDITIONAL'/ident/'response.npy';maps['A750']=np.load(rp,allow_pickle=False);files['A750']={'path':str(rp),'sha256':sha(rp),'model':'frozen original A750'}
        for branch in ['D0','D1']:
            rp=HERE/'inference/fixed'/branch/'CONDITIONAL'/ident/'response.npz';meta=load(rp.with_suffix('.json'));assert sha(rp)==meta['response_sha256'] and meta['model_sha256']==sel['paths'][branch]['sha256'];maps[branch]=mapped_npz(rp);files[branch]=meta
        raw=Image.open(it['image_path']).convert('RGB');assert sha(it['image_path'])==it['image_sha256']
        images=[raw]
        for label in ['A750','D0','D1']:
            rgb=np.full((3000,4096,3),45,np.uint8);rgb[1050:]=lut[np.rint(maps[label][1050:]*255).astype(np.uint8)];images.append(Image.fromarray(rgb))
        full=Image.new('RGB',(4*4096+3*25,3300),'white');dr=ImageDraw.Draw(full)
        titles=['原图  '+ident,'A750 原始中心响应','D0 旧10图续训','D1 扩充弱标注续训']
        for k,(im,title) in enumerate(zip(images,titles)):
            x=k*(4096+25);full.paste(im,(x,130));dr.text((x+20,25),title,font=font(43),fill='black')
        dr.text((25,3150),'所有热图固定0～1 TURBO；域外灰色未计算；无路径覆盖。训练中心剖面仅为拟合诊断，不是独立身份准确率。',font=font(41),fill='black')
        grad=Image.fromarray(lut[np.linspace(0,255,1800).astype(np.uint8)][None].repeat(25,0));full.paste(grad,(25,3230));dr.text((1855,3210),'0 → 1 响应强度',font=font(34),fill='black')
        fp=out/(ident+'__full.png');full.save(fp,compress_level=4)
        crop=choice['crop'];w=crop[2]-crop[0];h=crop[3]-crop[1]
        detail=Image.new('RGB',(w*4+75,h+110),'white');dd=ImageDraw.Draw(detail)
        for k,(im,title) in enumerate(zip(images,['原图局部','A750 同位置','D0 同位置','D1 同位置'])):
            x=k*(w+25);detail.paste(im.crop(crop),(x,110));dd.text((x+15,20),title,font=font(38),fill='black')
        dp=out/(ident+'__native_detail.png');detail.save(dp,compress_level=4)
        preview=detail.copy();preview.thumbnail((1800,1000));preview.save(out/(ident+'__qa_preview.png'))
        row={'id':ident,'status':'rendered','source_use':choice['split'],'source_path':it['image_path'],'source_sha256':it['image_sha256'],
            'responses':files,'full_figure':str(fp),'full_dimensions':list(full.size),'native_crop':crop,'detail':str(dp),'detail_dimensions':list(detail.size)}
        lines=[]
        if it.get('origin')=='new':
            annotation=load(it['annotation_path']);lines=[s['points_xy'] for s in annotation['segments'] if s['status']=='accepted_weak']
            row['profile_truth_scope']='accepted automatic weak training center, not independent test truth'
        elif choice['split']=='old_train':
            labels=load(BASE/'runs/run_20261029_001/data/label_sources'/(ident+'.json'));lines=labels['trusted_center_segments_xy'];row['profile_truth_scope']='historical reviewed partial training center'
        if lines:
            # Longest saved segment chosen by fixed length, never by output fit.
            points=np.asarray(max(lines,key=lambda x:np.linalg.norm(np.diff(np.asarray(x),axis=0),axis=1).sum()),float)
            vals={b:sample(maps[b],points) for b in maps};chart(ident,points,vals,out/(ident+'__center_profile.png'))
            np.savez_compressed(out/(ident+'__profile_values.npz'),points_xy=points,**vals)
            row['center_mean']={b:float(np.nanmean(v)) for b,v in vals.items()}
        rows.append(row);print({'compared':ident},flush=True)
    dump(out/'receipt.json',{'rows':rows,'fixed_selection_sha256':sha(HERE/'comparison_selection.json'),'checkpoints_sha256':sha(HERE/'selected_checkpoints.json'),
        'uses_original_float_no_path_overlay':True,'all_crops_native_pixels':True,'checkpoint_choice_not_affected':True})
if __name__=='__main__':
    if '--freeze' in sys.argv:freeze();print('Comparison selection frozen')
    else:main()
