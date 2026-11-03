"""Read-only H4 counterexample diagnosis. No labels, model calls or path edits."""
from pathlib import Path
import gzip, hashlib, importlib.util, json, math, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
H4=BASE/'postprocess_b_20261031_v4'
H2=BASE/'postprocess_b_20261030_v2'
ID='dual_20260923_101517__pair0195__right'
ATTACHMENT=Path('C:/Users/ADMINI~1/AppData/Local/Temp/codex-clipboard-8e01e497-6d75-43d5-9a44-12833210bce5.jpg')

def load(p):
    with (gzip.open(p,'rt',encoding='utf-8') if str(p).endswith('.gz') else open(p,encoding='utf-8-sig')) as f:return json.load(f)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,a):cv2.imencode(p.suffix,a)[1].tofile(str(p))
def panel(a,title,subtitle):
    out=Image.new('RGB',(a.shape[1],a.shape[0]+100),(245,247,250));out.paste(Image.fromarray(a[:,:,::-1]),(0,100));d=ImageDraw.Draw(out)
    d.text((15,9),title,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',28),fill=(20,30,40))
    d.text((15,55),subtitle,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22),fill=(50,65,80))
    return np.asarray(out)[:,:,::-1].copy()
def poly(a,p,color,width=2):cv2.polylines(a,[np.rint(p).astype(np.int32)],False,color,width,cv2.LINE_AA)
def rot(d,degrees):
    t=math.radians(degrees);return np.array([math.cos(t)*d[0]-math.sin(t)*d[1],math.sin(t)*d[0]+math.cos(t)*d[1]])

def main():
    cfg=load(H4/'inputs.json');job=next(j for j in cfg['jobs'] if j['id']==ID)
    assert job['dataset_use_status']=='unlabeled_open_development_no_new_test_split'
    assert '20260925' not in job['image_path']
    rp=H4/'all_001/记录'/f'{ID}.json.gz';record=load(rp)
    entry=record['entries'][record['selected_entry']];free=entry['free'];diag=free['diagnostics']
    original=Path(job['image_path']);assert sha(original)==job['image_sha256']
    response=Path(entry['response_file']);assert sha(response)==entry['response_sha256']
    bgr=cv2.imdecode(np.fromfile(original,np.uint8),cv2.IMREAD_COLOR)
    crop=cv2.resize(bgr,(2048,1500),interpolation=cv2.INTER_AREA)[525:]
    with np.load(response,allow_pickle=False) as z:heat=z['response'].copy()
    gray=cv2.GaussianBlur(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.,(0,0),.7)
    smooth=cv2.GaussianBlur(heat,(0,0),.7)
    helper=H2/'free_candidates.py';spec=importlib.util.spec_from_file_location('h5_diagnostic_h2_profiles',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    seed=(np.asarray(entry['seed']['xy'])-[0,1050])*.5
    initial=np.asarray(diag['initial_direction_xy']);normal=np.array([-initial[1],initial[0]])
    width=free['width_half_px'];progress=np.arange(0.,3000.1,6.)
    summaries=[diag['selected_axis']]+sorted([s for s in diag['best_refined_candidates'] if s['span_original_px']>diag['selected_axis']['span_original_px']],key=lambda s:s['span_original_px'])[-2:]
    axes=[]
    for summary in summaries:
        origin=seed[0]+summary['offset_widths']*width*normal;direction=rot(initial,summary['angle_degrees'])
        points=origin+progress[:,None]*.5*direction
        ev=module._profiles(gray,smooth,np.ones_like(heat),points,direction,width)
        support=ev['valid']&(ev['center']>=.28)&(ev['side_drop']>=.045)&(ev['side_distance_asymmetry_widths']<=.45)
        joint=support&(ev['raw_score']>=.25)&(ev['raw_contrast']>=.008)
        windows=[]
        for lo,hi in [(120,2000),(2000,2022),(2022,2400),(2400,2598),(2598,2800)]:
            use=(progress>=lo)&(progress<hi)
            windows.append(dict(progress_original_px=[lo,hi],center_mean=float(np.mean(ev['center'][use])),side_drop_mean=float(np.mean(ev['side_drop'][use])),raw_score_mean=float(np.mean(ev['raw_score'][use])),heat_support_fraction=float(np.mean(support[use])),joint_support_fraction=float(np.mean(joint[use]))))
        axes.append(dict(saved_axis_summary=summary,origin_half_xy=origin.tolist(),direction_xy=direction.tolist(),points_xy=points.tolist(),evidence={k:np.asarray(v).tolist() for k,v in ev.items()},diagnostic_heat_support=support.tolist(),diagnostic_joint_support=joint.tolist(),windows=windows))
    candidates=[]
    for i,c in enumerate(entry['connection']['candidates']):
        points=np.asarray(c['points_xy'])
        candidates.append(dict(index=i,kind=c['kind'],track_id=c.get('track_id'),rank_score=c['rank_score'],path_length_original_px=float(2*np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),join_start_xy=c.get('join_start_xy'),join_end_xy=c.get('join_end_xy'),last_xy=c['points_xy'][-1],metrics=c['metrics'],identity_verified=c['identity_verified']))
    selected_index=entry['connection']['selected_index'];selected=entry['connection']['candidates'][selected_index]
    color=cv2.applyColorMap(np.rint(np.clip(heat,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    bar=cv2.applyColorMap(np.linspace(255,0,300).astype(np.uint8)[:,None],cv2.COLORMAP_TURBO)
    color[40:340,1930:1960]=np.repeat(bar,30,axis=1)
    for y,text in [(50,'1'),(190,'.5'),(335,'0')]:cv2.putText(color,text,(1970,y),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    old=crop.copy();poly(old,free['points_xy'],(40,230,40));poly(old,selected['points_xy'],(255,60,230))
    existing=crop.copy();colors=[(40,230,40),(255,230,30),(230,40,220)]
    for axis,col in zip(axes,colors):
        pts=np.asarray(axis['points_xy']);keep=progress<=axis['saved_axis_summary']['span_original_px'];poly(existing,pts[keep],col)
        cv2.circle(existing,tuple(np.rint(pts[keep][-1]).astype(int)),6,col,2)
    pictures=[crop,color,old,existing]
    titles=[('原图','原始搜索ROI；不作增强或亮度拉伸'),('冻结方案B响应','固定0–1；响应不等于身份可信度'),('H4已选结果','绿：2022px直段；紫：连接后主路径'),('H4已经存在的直轴候选','绿2022；青2598；紫2658px；长轴不自动是真值')]
    save(HERE/f'{ID}__original_B_H4_existing_axes.jpg',np.concatenate([panel(a,*t) for a,t in zip(pictures,titles)],axis=1))
    box=[1000,2100,2800,2700];x0,y0,x1,y1=box;half=[x0//2,(y0-1050)//2,x1//2,(y1-1050)//2]
    local=[cv2.resize(a[half[1]:half[3],half[0]:half[2]],(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST) for a in pictures]
    save(HERE/f'{ID}__fixed_original_crop.jpg',np.concatenate([panel(a,*t) for a,t in zip(local,titles)],axis=1))
    result=dict(id=ID,attachment=str(ATTACHMENT),attachment_sha256=sha(ATTACHMENT),title_verified_at_original_resolution=True,misread_id_not_used='dual_20260923_101517__pair1195__right',
        command=sys.argv,script_sha256=sha(__file__),source_image=str(original),source_image_sha256=sha(original),H4_record=str(rp),H4_record_sha256=sha(rp),response_file=str(response),response_sha256=sha(response),
        helper_source=str(helper),helper_sha256=sha(helper),coordinate_mapping='original_xy=2*half_crop_xy+[0,1050]',selected_entry=record['selected_entry'],selected_candidate=selected_index,
        seed=entry['seed'],width_original_px=2*width,progress_original_px=progress.tolist(),axes=axes,connection_candidates=candidates,crop_original_xyxy=box,
        diagnosis=['H4 already retained longer refined axes2598/2658px. The selected2022px axis wins the fixed2000px evidence integral; later supported length cannot increase that integral.',
        'The selected connector leaves the already truncated straight endpoint, so its zero abandoned-support fraction does not protect alternative longer straight hypotheses.',
        'B plus grayscale profiles support parts of longer hypotheses; intermittent gaps and terminal identity remain unknown. These diagnostic axes are not new predictions or reference annotations.'],
        labels_read=False,reserved_data_read=False,new_network_inference=False,core_modified=False,independent_identity_reference=False)
    (HERE/f'{ID}__diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    assert sha(original)==job['image_sha256'] and sha(response)==entry['response_sha256']
    print(json.dumps(dict(id=ID,axes=[dict(summary=a['saved_axis_summary'],windows=a['windows']) for a in axes],exit_code=0),ensure_ascii=True))

if __name__=='__main__':main()
