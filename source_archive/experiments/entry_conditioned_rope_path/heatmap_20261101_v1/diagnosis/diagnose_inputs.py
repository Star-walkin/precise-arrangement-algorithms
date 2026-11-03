"""Fixed-coordinate raw/LCN/B inspection; no paths or labels create truth."""
from pathlib import Path
from collections import Counter
import json,hashlib
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
EXP=HERE.parent.parent
H6=EXP/'postprocess_b_20261031_v6'
INDEX=EXP/'runs/run_20261030_002/data/index.json'
AUDIT=EXP/'runs/run_20261030_002/branches/B/model/input_supervision_audit.json'
CONFIG=EXP/'runs/run_20261030_002/branches/B/model/config.json'
IDS=['L_run_20260923_01__f000585__left','dual_20260923_101517__pair0015__right']
PARAMS=dict(sigma_original_px=32.,sigma_model_px=16.,truncate=4,kernel_size=129,std_floor_gray255=2.5,clip_lcn=[-3.,3.],
    calculation='Half-resolution complete search ROI, original gray divided by255; constant-zero weighted Gaussian numerator and denominator; true gray zero retained.',
    display='Raw and B fixed0..1, LCN fixed-3..3. No per-image stretching.',
    normal_profiles='Seed-direction extrapolations only, not target truth; +/-96 original normal px, tangent average +/-20px; no H6 path or label used.')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def gray(p):return cv2.imdecode(np.fromfile(str(p),np.uint8),0)
def stats(v):
    v=np.asarray(v);return dict(mean=float(v.mean()),std=float(v.std()),minimum=float(v.min()),maximum=float(v.max()),quantile_01_50_99=np.quantile(v,[.01,.5,.99]).tolist())
def lcn(im):
    g=cv2.getGaussianKernel(129,16,cv2.CV_32F);blur=lambda a:cv2.sepFilter2D(a,-1,g,g,borderType=cv2.BORDER_CONSTANT)
    m=np.ones_like(im);w=blur(m);mean=blur(im*m)/np.maximum(w,1e-5);variance=np.maximum(blur(im*im*m)/np.maximum(w,1e-5)-mean*mean,0);std=np.sqrt(variance)
    z=(im-mean)/np.maximum(std,2.5/255);return z,mean,std,w
def sample(im,xy):
    return cv2.remap(im,(xy[...,0]/2).astype(np.float32),((xy[...,1]-1050)/2).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))

FONT='C:/Windows/Fonts/arial.ttf'
def draw_maps(path,arrays,titles,limits,bounds,seed=None,centers=None,normal=None,vertical=False):
    x0,y0,x1,y1=bounds;pw=1650 if vertical else 900;ph=int(pw*(y1-y0)/(x1-x0));cw=pw+90;ch=ph+100
    board=Image.new('RGB',(cw if vertical else cw*3,ch*3 if vertical else ch),(250,250,250));d=ImageDraw.Draw(board);font=ImageFont.truetype(FONT,20)
    for k,(a,title,(lo,hi)) in enumerate(zip(arrays,titles,limits)):
        ox=0 if vertical else cw*k;oy=ch*k if vertical else 0
        disp=np.clip((a-lo)/(hi-lo)*255,0,255).astype(np.uint8)
        rgb=cv2.cvtColor(cv2.applyColorMap(disp,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB) if k==2 else np.repeat(disp[...,None],3,2)
        board.paste(Image.fromarray(rgb).resize((pw,ph),Image.Resampling.BILINEAR),(ox,oy+40));d.text((ox+5,oy+6),title,font=font,fill='black')
        ramp=np.linspace(255,0,ph).astype(np.uint8)[:,None].repeat(20,1);bar=cv2.cvtColor(cv2.applyColorMap(ramp,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB) if k==2 else np.repeat(ramp[...,None],3,2)
        board.paste(Image.fromarray(bar),(ox+pw+10,oy+40))
        for f in [0,.5,1]:d.text((ox+pw+33,oy+40+f*(ph-20)),f'{hi-f*(hi-lo):.1f}',font=font,fill='black')
        def coord(pt):return (ox+(pt[0]-x0)/(x1-x0)*pw,oy+40+(pt[1]-y0)/(y1-y0)*ph)
        if seed is not None:d.line([coord(v) for v in seed],fill=(40,255,20),width=3)
        if centers is not None:
            for i,c in enumerate(centers):
                d.line([coord(c-96*normal),coord(c+96*normal)],fill=(255,0,255),width=2);d.text(coord(c+[15,0]),str(i+1),font=font,fill=(255,30,30))
        d.text((ox+5,oy+ph+45),f'Original x={x0}..{x1}, y={y0}..{y1}; fixed scale',font=font,fill='black')
    board.save(path)

def draw_profiles(path,profiles):
    w,h=620,300;board=Image.new('RGB',(w*3,h*4+40),'white');d=ImageDraw.Draw(board);font=ImageFont.truetype(FONT,18)
    d.text((10,8),'Seed-normal slices only; not target labels. Offsets and positions use original pixels.',font=font,fill='black')
    for i,p in enumerate(profiles):
        for k,(name,(lo,hi)) in enumerate([('gray',(0,1)),('lcn',(-3,3)),('B',(0,1))]):
            x0=k*w+65;y0=i*h+85;ww=520;hh=190
            d.text((k*w+8,i*h+45),f'{i+1}: {p["seed_progress_original_px"]}px from seed / {name}',font=font,fill='black')
            for f in [0,.5,1]:
                y=y0+f*hh;d.line([(x0,y),(x0+ww,y)],fill=(220,220,220));d.text((k*w+3,y-8),f'{hi-f*(hi-lo):.1f}',font=font,fill='black')
            for xx in [-96,-48,0,48,96]:
                x=x0+(xx+96)/192*ww;d.line([(x,y0),(x,y0+hh)],fill=(220,220,220));d.text((x-14,y0+hh+5),str(xx),font=font,fill='black')
            pts=[(x0+(xx+96)/192*ww,y0+(hi-np.clip(v,lo,hi))/(hi-lo)*hh) for xx,v in zip(p['normal_offsets_original_px'],p['series'][name])]
            d.line(pts,fill=(10,80,190),width=2)
    board.save(path)

def main():
    HERE.mkdir(exist_ok=True);config=load(H6/'inputs.json');jobs={j['id']:j for j in config['jobs']};index=load(INDEX);train=[j for j in index['items'] if j['split']=='train']
    audit=load(AUDIT);trgroups=Counter(j['source_group'] for j in train);trainseed=np.array([np.mean(j['known_seed_xy'],axis=0) for j in train]);allrows=[]
    sampling={'index_path':str(INDEX),'index_sha256':sha(INDEX),'training_config_path':str(CONFIG),'training_config_sha256':sha(CONFIG),'train_image_count':len(train),'train_source_groups':dict(trgroups),'train_seed_centroid_x_range':trainseed[:,0].min().item().__str__()+'..'+str(trainseed[:,0].max()),'training_seed_centroids_xy':trainseed.tolist(),'neighbor_negative_pixels_total':sum(x['neighbor_pixels'] for x in audit),'background_negative_pixels_total':sum(x['background_pixels'] for x in audit),'old_center_valid_pixels_total':sum(x['old_center_pixels'] for x in audit),'sampling':'Uniform 10 original images/fullsearchROI, no independent-patch sample inflation; 3 left images from one source,7 right from3 sources. Horizontal flip augments view but adds no independent source.'}
    for ident in IDS:
        j=jobs[ident];assert '20260925' not in j['image_path'];assert sha(j['image_path'])==j['image_sha256']
        original=gray(j['image_path']);im=cv2.resize(original[1050:],(2048,975),interpolation=cv2.INTER_AREA).astype(np.float32)/255
        response=np.load(j['response_files'][0])['response'];assert response.shape==im.shape and np.isfinite(response).all()
        z,mean,std,w=lcn(im);np.savez_compressed(HERE/(ident+'__fields.npz'),gray=im,local_mean=mean,local_std=std,weight=w,lcn_unclipped=z,response_B=response)
        seed=np.asarray(j['seeds'][0]['xy'],float);origin=seed[0];tangent=seed[-1]-seed[0];tangent/=np.linalg.norm(tangent);normal=np.array([-tangent[1],tangent[0]])
        distances=[400,800,1200,1600] if ident.endswith('left') else [600,1200,1800,2400]
        centers=np.array([origin+s*tangent for s in distances]);offsets=np.linspace(-96,96,193);along=np.linspace(-20,20,21)
        profiles=[]
        for s,c in zip(distances,centers):
            xy=c+offsets[:,None,None]*normal+along[None,:,None]*tangent
            a={k:np.nanmean(sample(v,xy),axis=1) for k,v in {'gray':im,'lcn':z,'B':response,'std':std}.items()}
            center=abs(offsets)<=8;flank=(abs(offsets)>=16)&(abs(offsets)<=36)
            summary={k:{'axis_center_mean':float(v[center].mean()),'flank_mean':float(v[flank].mean()),'axis_center_minus_flank':float(v[center].mean()-v[flank].mean()),'profile_minimum':float(v.min()),'profile_maximum':float(v.max())} for k,v in a.items()}
            profiles.append(dict(seed_progress_original_px=s,axis_center_original_xy=c.tolist(),normal_xy=normal.tolist(),normal_offsets_original_px=offsets.tolist(),tangent_average_span_original_px=40,series={k:v.tolist() for k,v in a.items()},axis_zone_summary=summary,target_identity_known=False))
        draw_maps(HERE/(ident+'__raw_LCN_B.png'),[im,z,response],['Original gray','LCN sigma32 original px','Frozen B response strength'],[(0,1),(-3,3),(0,1)],[0,1050,4096,3000],seed,centers,normal)
        draw_profiles(HERE/(ident+'__normal_profiles.png'),profiles)
        # Fixed local crop preserves pixel detail without following any prediction.
        bounds=[250,2100,2200,2800] if ident.endswith('left') else [1300,2000,3350,2800]
        x0,y0,x1,y1=bounds;sl=(slice((y0-1050)//2,(y1-1050)//2),slice(x0//2,x1//2))
        draw_maps(HERE/(ident+'__local_same_crop.png'),[v[sl] for v in [im,z,response]],['Raw gray','LCN fixed display','Frozen B response'],[(0,1),(-3,3),(0,1)],bounds,vertical=True)
        tr=[a for a in train if a['source_group']==j['source_group']]
        row=dict(id=ident,source_group=j['source_group'],image_path=j['image_path'],image_sha256=j['image_sha256'],response_path=j['response_files'][0],response_sha256=sha(j['response_files'][0]),seed=j['seeds'][0],same_source_training_ids=[a['id'] for a in tr],same_source_training_seeds=[a['known_seed_xy'] for a in tr],search_roi_original_xyxy=[0,1050,4096,3000],shape_model=[975,2048],gray_stats=stats(im),old_global_normalization=dict(mean=float(im.mean()),std=max(float(im.std()),.08)),lcn_stats=stats(z),local_std_stats=stats(std),std_floor_pixel_fraction=float(np.mean(std<2.5/255)),lcn_display_clip_fraction=float(np.mean(abs(z)>3)),weight_minimum=float(w.min()),response_stats=stats(response),response_fraction_ge_half=float(np.mean(response>=.5)),fixed_local_crop_original_xyxy=bounds,profiles=profiles)
        allrows.append(row)
        print(ident,'source_train_count',len(tr),'gray',row['gray_stats'],'clip',row['lcn_display_clip_fraction'],'B',row['response_stats'])
    out=dict(script_sha256=sha(__file__),parameters=PARAMS,training_scope=sampling,items=allrows,labels_read=False,H6_path_used_as_reference=False,training_or_model_modified=False,notes='LCN changes local contrast only and cannot create target identity. The target/neighbor distinction still requires existing reliable supervision. Fixed profiles use the saved entrance direction solely as reproducible sampling coordinates, not ground truth.')
    (HERE/'input_diagnosis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
