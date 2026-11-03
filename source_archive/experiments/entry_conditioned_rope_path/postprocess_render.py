"""Four-panel image delivery: original, frozen A, frozen B, P1 center overlays."""
from pathlib import Path
import argparse,json,math,sys,time
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont

Image.MAX_IMAGE_PIXELS=250_000_000
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dashed(draw,a,b,color,width,dash=15,gap=11,phase=0.):
    a=np.asarray(a,float);b=np.asarray(b,float);d=b-a;length=float(np.linalg.norm(d))
    if length<1e-6:return
    u=d/length
    for start in np.arange(-(phase%(dash+gap)),length,dash+gap):
        lo=max(0.,start);end=min(length,start+dash)
        if end>lo:draw.line([tuple(a+u*lo),tuple(a+u*end)],fill=color,width=width)
def draw_path(draw,candidate,variant,selected,origin):
    pts=np.asarray(candidate['points_xy'],float)+origin
    states=candidate.get('segment_status',['ambiguous']*(len(pts)-1))
    primary=(0,245,245,245) if variant=='A' else (255,150,0,245)
    uncertain=(235,70,255,235) if variant=='A' else (175,105,255,235)
    secondary=(65,180,230,130) if variant=='A' else (235,175,60,130)
    arc=0.
    for a,b,status in zip(pts[:-1],pts[1:],states):
        if status!='entry_condition':
            if not selected:dashed(draw,a,b,secondary,3,8,12,arc)
            elif status in ['bridge','bridged']:dashed(draw,a,b,primary,7,16,11,arc)
            elif status=='ambiguous':dashed(draw,a,b,uncertain,7,6,10,arc)
            else:draw.line([tuple(a),tuple(b)],fill=primary,width=7)
        arc+=float(np.linalg.norm(b-a))

def run(run_dir):
    run_dir=Path(run_dir).resolve();manifest=load(run_dir/'batch/manifest.json')
    assert (run_dir/'batch/postprocess_receipt.json').exists()
    out=run_dir/'四联图_原图_A_B_中心线示意';out.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',44);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',34)
    lut=cv2.applyColorMap(np.arange(256,dtype=np.uint8)[:,None],cv2.COLORMAP_TURBO)[:,0,::-1].copy()
    bar=Image.fromarray(lut[np.linspace(0,255,1200).astype(np.uint8)][None].repeat(24,0))
    records=[];tick=time.perf_counter()
    for index,job in enumerate(manifest['jobs'],1):
        w,h=job['shape'][1],job['shape'][0];rh=h+210
        with Image.open(job['image_path']) as original:rgb=original.convert('RGB')
        canvas=Image.new('RGB',(4*w,rh*len(job['conditions'])),(245,247,250));draw=ImageDraw.Draw(canvas)
        summaries=[]
        for k,condition in enumerate(job['conditions']):
            top=k*rh;label=f"入口候选 {k+1}/{len(job['conditions'])}";seed=np.asarray(condition['entry_xy'],float)
            canvas.paste(rgb,(0,top+130));draw.text((25,top+10),'原图 | '+job['id'],font=font,fill=(20,30,40))
            draw.text((25,top+74),label+' | 同一入口比较A/B | 未确认目标末端身份',font=small,fill=(90,70,50))
            for col,variant in [(1,'A'),(2,'B')]:
                response=np.load(condition['response_paths'][variant],mmap_mode='r');finite=np.isfinite(response)
                colors=np.full((*response.shape,3),38,np.uint8);colors[finite]=lut[np.rint(np.asarray(response[finite])*255).astype(np.uint8)]
                canvas.paste(Image.fromarray(colors),(col*w,top+130))
                draw.text((col*w+25,top+10),f'方案{variant}原始中心响应 | 冻结750步 | '+label,font=font,fill=(20,30,40))
                draw.text((col*w+25,top+74),'统一0～1；灰色未计算；未滤波、未补线',font=small,fill=(80,90,100))
                canvas.paste(bar,(col*w+25,top+h+145));draw.text((col*w+1250,top+h+135),'0 → 1 响应强度（非概率）',font=small,fill=(20,30,40))
                del response,finite,colors
            canvas.paste(rgb,(3*w,top+130));overlay=Image.new('RGBA',(w,h),(0,0,0,0));od=ImageDraw.Draw(overlay)
            choices={}
            for variant in ['A','B']:
                path=Path(condition['response_paths'][variant]).parent/'P1/paths.json';result=load(path);selected=result['selected_candidate_id']
                for c in result['candidates']:
                    if c['id']!=selected:draw_path(od,c,variant,False,np.array([0,0]))
                if selected:
                    chosen=next(c for c in result['candidates'] if c['id']==selected);draw_path(od,chosen,variant,True,np.array([0,0]))
                choices[variant]={'selected':selected,'candidate_count':len(result['candidates']),
                    'reason':result['search'].get('selection_reason',result['search'].get('reason'))}
            od.line([tuple(p) for p in seed],fill=(255,255,255,255),width=8)
            direction=(seed[-1]-seed[0]);direction/=max(np.linalg.norm(direction),1e-8);normal=np.array([-direction[1],direction[0]])
            end=seed[-1]+direction*35;od.polygon([tuple(end),tuple(end-direction*30+normal*12),tuple(end-direction*30-normal*12)],fill=(255,255,255,255))
            canvas.paste(overlay,(3*w,top+130),overlay)
            draw.text((3*w+25,top+10),'P1中心线示意覆盖（不是绳体宽mask） | A青 / B橙',font=font,fill=(20,30,40))
            draw.text((3*w+25,top+74),'支持实线；补线虚线；歧义紫虚线；细虚线仅为竞争候选；白线入口',font=small,fill=(70,70,85))
            a='暂选' if choices['A']['selected'] else '拒识';b='暂选' if choices['B']['selected'] else '拒识'
            draw.text((3*w+25,top+h+135),f'A/P1：{a}；B/P1：{b}。线宽仅供显示，不代表真实绳宽。',font=small,fill=(30,40,55))
            summaries.append(choices)
        dest=out/(job['id']+'.jpg');canvas.save(dest,quality=95,subsampling=0)
        with Image.open(dest) as checked:assert checked.size==(4*w,rh*len(job['conditions']));checked.verify()
        records.append({'id':job['id'],'output':str(dest),'dimensions':[4*w,rh*len(job['conditions'])],'entry_results':summaries})
        if index==1 or index%30==0:print(json.dumps({'rendered':index,'total':len(manifest['jobs']),'seconds':time.perf_counter()-tick}),flush=True)
    (run_dir/'batch/render_receipt.json').write_text(json.dumps({'exit_code':0,'count':len(records),'output':str(out),'seconds':time.perf_counter()-tick,
        'records':records,'command':sys.argv,'heatmap_scale':[0,1],'fourth_panel':'selected A/P1 and B/P1 center schematic; competitors explicit; no true body mask claimed'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'completed':True,'count':len(records),'output':str(out)},ensure_ascii=False))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--run',required=True);run(p.parse_args().run)
