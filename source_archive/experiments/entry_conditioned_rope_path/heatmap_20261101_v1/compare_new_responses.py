"""Post-selection old B/new heatmap diagnostic on frozen sampling coordinates.

No path, label, network call, sample selection or training parameter adjustment.
Peak counts are response morphology, never true localization accuracy.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
DIAG=HERE/'diagnosis'
SETTINGS=dict(response_display=[0,1],raw_display=[0,1],
    smoothing_sigma_original_px=2.,peak_minimum_response=.30,peak_local_prominence=.10,
    peak_prominence_radius_original_px=12,peak_minimum_separation_original_px=12,
    peak_semantics='Diagnostic local response maxima on the same frozen seed-normal slices; extra peaks are not automatically false target detections.',
    interpolation='OpenCV bilinear at exact original coordinates; tangent-average +/-20 original pixels unchanged',
    declared_before_new_results=True)
FONT='C:/Windows/Fonts/arial.ttf'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sample(response,profile):
    c=np.asarray(profile['axis_center_original_xy']);n=np.asarray(profile['normal_xy']);t=np.array([n[1],-n[0]])
    offsets=np.asarray(profile['normal_offsets_original_px']);along=np.linspace(-20,20,21)
    xy=c+offsets[:,None,None]*n+along[None,:,None]*t
    v=cv2.remap(response,(xy[...,0]/2).astype(np.float32),((xy[...,1]-1050)/2).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))
    return np.nanmean(v,axis=1)
def peaks(values,offsets):
    v=cv2.GaussianBlur(np.asarray(values,np.float32)[None,:],(13,1),sigmaX=2,borderType=cv2.BORDER_REFLECT_101)[0]
    candidates=[]
    for i in range(1,len(v)-1):
        if v[i]<.30 or not(v[i]>=v[i-1] and v[i]>v[i+1]):continue
        left=v[max(0,i-12):i];right=v[i+1:min(len(v),i+13)]
        prominence=float(v[i]-max(left.min(),right.min()))
        if prominence>=.10:candidates.append((i,prominence))
    chosen=[]
    for i,p in sorted(candidates,key=lambda q:-float(v[q[0]])):
        if all(abs(offsets[i]-offsets[j])>=12 for j,_ in chosen):chosen.append((i,p))
    return [dict(normal_offset_original_px=float(offsets[i]),smoothed_response=float(v[i]),local_prominence=p) for i,p in sorted(chosen)]
def numbers(values,offsets):
    v=np.asarray(values);x=np.asarray(offsets);c=abs(x)<=8;f=(abs(x)>=16)&(abs(x)<=36)
    return dict(axis_center_mean=float(v[c].mean()),flank_mean=float(v[f].mean()),axis_center_minus_flank=float(v[c].mean()-v[f].mean()),
        maximum_response=float(v.max()),maximum_response_offset_original_px=float(x[v.argmax()]),diagnostic_peaks=peaks(v,x),axis_is_not_ground_truth=True)
def maps(path,arrays,titles,bounds,vertical=False):
    x0,y0,x1,y1=bounds;pw=1650 if vertical else 900;ph=int(pw*(y1-y0)/(x1-x0));cw=pw+90;ch=ph+100
    board=Image.new('RGB',(cw if vertical else cw*3,ch*3 if vertical else ch),(250,250,250));d=ImageDraw.Draw(board);font=ImageFont.truetype(FONT,20)
    for k,(a,title) in enumerate(zip(arrays,titles)):
        ox=0 if vertical else cw*k;oy=ch*k if vertical else 0;disp=np.clip(a*255,0,255).astype(np.uint8)
        rgb=cv2.cvtColor(cv2.applyColorMap(disp,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB) if k else np.repeat(disp[...,None],3,2)
        board.paste(Image.fromarray(rgb).resize((pw,ph),Image.Resampling.BILINEAR),(ox,oy+40));d.text((ox+5,oy+6),title,font=font,fill='black')
        ramp=np.linspace(255,0,ph).astype(np.uint8)[:,None].repeat(20,1);bar=cv2.cvtColor(cv2.applyColorMap(ramp,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB) if k else np.repeat(ramp[...,None],3,2)
        board.paste(Image.fromarray(bar),(ox+pw+10,oy+40))
        for f in [0,.5,1]:d.text((ox+pw+33,oy+40+f*(ph-20)),f'{1-f:.1f}',font=font,fill='black')
        d.text((ox+5,oy+ph+45),f'Original x={x0}..{x1}, y={y0}..{y1}; fixed0..1',font=font,fill='black')
    board.save(path)
def profiles_plot(path,profiles):
    w,h=620,300;board=Image.new('RGB',(w*3,h*4+40),'white');d=ImageDraw.Draw(board);font=ImageFont.truetype(FONT,18)
    d.text((10,8),'Unchanged seed-normal coordinates, not labels. Orange markers: diagnostic response maxima.',font=font,fill='black')
    for i,p in enumerate(profiles):
        for k,name in enumerate(['gray','old_B','new']):
            x0=k*w+65;y0=i*h+85;ww=520;hh=190
            d.text((k*w+8,i*h+45),f'{i+1}: {p["seed_progress_original_px"]}px from seed / {name}',font=font,fill='black')
            for f in [0,.5,1]:
                y=y0+f*hh;d.line([(x0,y),(x0+ww,y)],fill=(220,220,220));d.text((k*w+3,y-8),f'{1-f:.1f}',font=font,fill='black')
            for xx in [-96,-48,0,48,96]:
                x=x0+(xx+96)/192*ww;d.line([(x,y0),(x,y0+hh)],fill=(220,220,220));d.text((x-14,y0+hh+5),str(xx),font=font,fill='black')
            pts=[(x0+(xx+96)/192*ww,y0+(1-np.clip(v,0,1))*hh) for xx,v in zip(p['normal_offsets_original_px'],p['series'][name])];d.line(pts,fill=(10,80,190),width=2)
            if name!='gray':
                for q in p['statistics'][name]['diagnostic_peaks']:
                    x=x0+(q['normal_offset_original_px']+96)/192*ww;y=y0+(1-q['smoothed_response'])*hh;d.ellipse((x-3,y-3,x+3,y+3),fill=(235,120,0))
    board.save(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default=str(HERE/'batch_inference/manifest.json'));ap.add_argument('--out',default='comparison_responses');args=ap.parse_args()
    freeze=load(DIAG/'freeze_diagnosis.json')
    for p,r in freeze['files'].items():assert sha(p)==r['sha256'],f'Frozen diagnostic changed: {p}'
    manifest_path=Path(args.manifest);manifest=load(manifest_path);diagnosis=load(DIAG/'input_diagnosis.json');lookup={r['id']:r for r in manifest['rows']};out=HERE/args.out;out.mkdir(exist_ok=False)
    results=[]
    for row in diagnosis['items']:
        ident=row['id'];m=lookup[ident];entry=next(e for e in m['entries'] if e['entry_index']==0)
        assert np.allclose(entry['seed_xy'],row['seed']['xy'],rtol=0,atol=1e-7)
        path=Path(entry['response_file']);assert sha(path)==entry['response_sha256'];new=np.load(path)['response'];fields=np.load(DIAG/(ident+'__fields.npz'));oldB=fields['response_B'];raw=fields['gray']
        assert new.shape==oldB.shape==(975,2048) and np.isfinite(new).all() and 0<=new.min()<=new.max()<=1
        profiles=[]
        for p in row['profiles']:
            sampled_old=sample(oldB,p);assert np.max(np.abs(sampled_old-np.asarray(p['series']['B'])))<1e-5
            new_v=sample(new,p);profiles.append({k:p[k] for k in ['seed_progress_original_px','axis_center_original_xy','normal_xy','normal_offsets_original_px']}|dict(series={'gray':p['series']['gray'],'old_B':p['series']['B'],'new':new_v.tolist()},statistics={k:numbers(v,p['normal_offsets_original_px']) for k,v in [('old_B',p['series']['B']),('new',new_v)]},target_identity_known=False))
        maps(out/(ident+'__raw_oldB_new.png'),[raw,oldB,new],['Original gray','Frozen B response','New selected response'],[0,1050,4096,3000])
        x0,y0,x1,y1=row['fixed_local_crop_original_xyxy'];sl=(slice((y0-1050)//2,(y1-1050)//2),slice(x0//2,x1//2))
        maps(out/(ident+'__local_same_crop.png'),[v[sl] for v in [raw,oldB,new]],['Original gray','Frozen B response','New selected response'],[x0,y0,x1,y1],True)
        profiles_plot(out/(ident+'__normal_profiles.png'),profiles)
        results.append(dict(id=ident,entry_index=0,seed_xy=entry['seed_xy'],new_response_path=str(path),new_response_sha256=sha(path),old_response_path=row['response_path'],old_response_sha256=row['response_sha256'],profiles=profiles,
            whole_search_summary={k:{'mean':float(v.mean()),'std':float(v.std()),'fraction_ge_half':float(np.mean(v>=.5))} for k,v in [('old_B',oldB),('new',new)]},whole_search_mean_abs_difference=float(np.mean(np.abs(new-oldB))),H6_path_used=False))
    result=dict(command=sys.argv,script_sha256=sha(__file__),manifest_path=str(manifest_path),manifest_sha256=sha(manifest_path),diagnosis_freeze_sha256=sha(DIAG/'freeze_diagnosis.json'),variant=manifest['variant'],model=manifest['model'],settings=SETTINGS,items=results,
        limits='Fewer peaks or more center-axis response is not accuracy; the axis is a fixed seed extrapolation, not a true target center. Review against raw structure. No parameters may be retuned from this diagnostic.',labels_read=False,H6_paths_used=False,new_inference=False)
    save(out/'response_comparison.json',result);print(json.dumps({'output':str(out),'images':len(results),'sha256':sha(out/'response_comparison.json'),'script_sha256':sha(__file__)},ensure_ascii=True))

if __name__=='__main__':main()
