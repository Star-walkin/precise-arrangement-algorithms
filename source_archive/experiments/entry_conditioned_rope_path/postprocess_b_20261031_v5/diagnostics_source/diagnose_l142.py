"""Visual, read-only raw/B normal-strip check after the unresolved L142 stop."""
from pathlib import Path
import gzip,hashlib,json,sys
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent
RUN=HERE.parent
ID='L_run_20260923_01__f000142__left'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with (gzip.open(p,'rt',encoding='utf-8') if str(p).endswith('.gz') else open(p,encoding='utf-8-sig')) as f:return json.load(f)
def sample(a,p):return cv2.remap(a,p[...,0].astype(np.float32),p[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
def write(p,a):cv2.imencode(p.suffix,a)[1].tofile(str(p))
def main():
    config=load(RUN/'inputs.json');job=next(j for j in config['jobs'] if j['id']==ID)
    assert job['dataset_use_status']=='unlabeled_open_development_no_new_test_split'
    record_path=RUN/'pilot_002/记录'/f'{ID}.json.gz';record=load(record_path);entry=record['entries'][record['selected_entry']]
    source=Path(job['image_path']);assert sha(source)==job['image_sha256'];response=Path(entry['response_file']);assert sha(response)==entry['response_sha256']
    original=cv2.imdecode(np.fromfile(source,np.uint8),cv2.IMREAD_COLOR)
    original_gray=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    with np.load(response,allow_pickle=False) as data:heat=data['response'].copy()
    free=entry['free'];origin=np.asarray(free['points_xy'][0]);direction=np.asarray(free['axis_direction_xy']);normal=np.array([-direction[1],direction[0]])
    progress=np.arange(400.,1200.01,2.);offsets=np.arange(-40.,40.01,.5)
    local=origin+progress[:,None,None]*.5*direction+offsets[None,:,None]*.5*normal
    original_points=2*local+[0,1050];raw=sample(original_gray,original_points);values=sample(heat,local)
    raw_im=cv2.cvtColor(np.rint(raw.T*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    heat_im=cv2.applyColorMap(np.rint(np.clip(values.T,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    strips=[cv2.resize(v,(1600,420),interpolation=cv2.INTER_NEAREST) for v in (raw_im,heat_im)]
    for im in strips:
        cv2.line(im,(0,210),(1599,210),(200,200,200),1)
        xx=round((558-400)/800*1599);cv2.line(im,(xx,0),(xx,419),(255,255,255),2)
    board=Image.new('RGB',(1680,1040),(245,247,250));draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    draw.text((12,8),'L142原图/B法向条带：沿已选直轴进度400–1200原px；法向−40到+40原px',font=font,fill=(20,30,40))
    draw.text((12,48),'白竖线558px为保存的自由段末点；横线是轴，不是真值。原灰度固定0–1，B固定0–1。',font=small,fill=(20,30,40))
    board.paste(Image.fromarray(strips[0][:,:,::-1]),(0,100));board.paste(Image.fromarray(strips[1][:,:,::-1]),(0,600))
    draw.text((12,560),'下图B：更强法向峰可能属于邻圈，不能直接当正确中心。',font=font,fill=(20,30,40))
    bar=cv2.applyColorMap(np.linspace(255,0,420).astype(np.uint8)[:,None],cv2.COLORMAP_TURBO)
    bar=np.repeat(bar,22,axis=1);board.paste(Image.fromarray(bar[:,:,::-1]),(1610,600));draw.text((1637,600),'1',font=small,fill=(20,30,40));draw.text((1637,980),'0',font=small,fill=(20,30,40))
    board.save(HERE/'L142_normal_strip_raw_B.png')
    # Original-coordinate crop and strip boundary only; no inferred extension.
    bounds=np.asarray([local[0,0],local[0,-1],local[-1,-1],local[-1,0]])*2+[0,1050]
    lo=np.floor(bounds.min(0)-80).astype(int);hi=np.ceil(bounds.max(0)+80).astype(int);lo=np.maximum(lo,0);hi=np.minimum(hi,[original.shape[1],original.shape[0]])
    crop=original[lo[1]:hi[1],lo[0]:hi[0]].copy();cv2.polylines(crop,[np.rint(bounds-lo).astype(np.int32)],True,(255,220,40),1,cv2.LINE_AA)
    write(HERE/'L142_original_strip_context.png',crop)
    rows=[]
    for t in (600.,800.,1000.):
        idx=int(np.argmin(abs(progress-t)));rows.append(dict(progress_original_px=float(progress[idx]),offsets_original_px=offsets.tolist(),original_gray=raw[idx].tolist(),B_response=values[idx].tolist(),maximum_B_offset_original_px=float(offsets[np.argmax(values[idx])]),maximum_B_not_target_identity=True))
    result=dict(id=ID,command=sys.argv,script_sha256=sha(__file__),source_image=str(source),source_image_sha256=sha(source),response=str(response),response_sha256=sha(response),record=str(record_path),record_sha256=sha(record_path),
        crop_original_xyxy=[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])],free_end_original_xy=(np.asarray(free['points_xy'][-1])*2+[0,1050]).tolist(),normal_profiles=rows,
        method='Display only: raw image at full source resolution; B sampled in frozen half crop. Fixed grayscale/color range. No alternative path generated.',labels_read=False,new_inference=False,core_modified=False,identity_reference=False)
    (HERE/'L142_normal_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(id=ID,exit_code=0,profiles=[{'t':r['progress_original_px'],'max_B_offset':r['maximum_B_offset_original_px']} for r in rows]),ensure_ascii=True))
if __name__=='__main__':main()
