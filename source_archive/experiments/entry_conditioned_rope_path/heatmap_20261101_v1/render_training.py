"""Render recorded H7 training diagnostics with Pillow; no models/images read."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
FONT=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
SMALL=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
TITLE=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
COLORS={'C':(27,111,177),'D':(212,91,32),'stage-balanced center':(27,111,177),'total objective':(212,91,32)}


def load(p):return json.loads(Path(p).read_text(encoding='utf8'))


def curve(draw,box,series,title,ymax=None):
    x0,y0,x1,y1=box;draw.text((x0,y0-34),title,fill='black',font=FONT)
    xmax=max(max(s['x']) for s in series);ymax=ymax or max(max(s['y']) for s in series)*1.08
    for j in range(5):
        y=y1-(y1-y0)*j/4;draw.line((x0,y,x1,y),fill=(220,224,228));draw.text((x0-60,y-9),f'{ymax*j/4:.3f}',fill=(55,55,55),font=SMALL)
        x=x0+(x1-x0)*j/4;draw.text((x-10,y1+7),f'{xmax*j/4:.0f}',fill=(55,55,55),font=SMALL)
    draw.line((x0,y0,x0,y1,x1,y1),fill='black',width=2)
    for k,s in enumerate(series):
        pts=[(x0+(x1-x0)*x/xmax,y1-(y1-y0)*y/ymax) for x,y in zip(s['x'],s['y'])]
        if len(pts)>1:draw.line(pts,fill=s['color'],width=2)
        draw.text((x0+k*290,y1+32),s['name'],fill=s['color'],font=SMALL)


def main():
    rows=[json.loads(x) for x in (HERE/'smallcheck'/'log.jsonl').read_text().splitlines()]
    result=load(HERE/'smallcheck'/'result.json')
    im=Image.new('RGB',(1250,650),'white');d=ImageDraw.Draw(im)
    d.text((30,20),'H7 independent four-image check (discarded weights)',fill='black',font=TITLE)
    curve(d,(100,110,1190,470),[{'name':k,'x':[r['step'] for r in rows],'y':[r[f] for r in rows],'color':COLORS[k]}
          for k,f in [('stage-balanced center','center_loss'),('total objective','total_loss')]],
          'Recorded per-update loss; four fixed training images cycle')
    d.text((35,565),f"{result['status']} | {result['steps']} steps, {result['training_seconds']:.1f}s | Fixed-four total: "
           f"{result['initial_train']['macro_total_loss']:.5f} -> {result['final_train']['macro_total_loss']:.5f}",fill='black',font=FONT)
    d.text((35,600),'Stage-balanced center is the optimized center term. Overall pixel MSE is a separate diagnostic.',fill=(60,60,60),font=SMALL)
    im.save(HERE/'smallcheck_loss.png')
    if not (HERE/'paired_comparison.json').exists():return
    logs={v:[json.loads(x) for x in (HERE/'branches'/v/'model'/'train_log.jsonl').read_text().splitlines()] for v in ('C','D')}
    im=Image.new('RGB',(1500,1000),'white');d=ImageDraw.Draw(im)
    d.text((30,20),'H7 matched continuations: actual recorded optimization',fill='black',font=TITLE)
    for j,(field,title) in enumerate([('center_loss','Stage-balanced center MSE'),('total_loss','Center + 0.10 neighbor + 0.05 background')]):
        ss=[]
        for v,rows in logs.items():
            y=np.array([r[field] for r in rows]);sm=np.convolve(y,np.ones(25)/25,mode='valid')
            ss.append({'name':f'{v}: 25-update running mean','x':list(range(25,len(y)+1)),'y':sm.tolist(),'color':COLORS[v]})
        curve(d,(100,115+j*400,1430,390+j*400),ss,title)
    cmp=load(HERE/'paired_comparison.json')
    d.text((35,915),f"Common checkpoint: {cmp['main_comparison_step']} updates. Shared inputs/targets/masks verified at {cmp['shared_steps_checked']} updates.",fill='black',font=FONT)
    d.text((35,950),'Curves use changing sampled images. Use common_training_fit.json for fixed-image comparison; no evaluation labels are used.',fill=(60,60,60),font=SMALL)
    im.save(HERE/'training_loss.png')
    fit=load(HERE/'common_training_fit.json');im=Image.new('RGB',(1500,600),'white');d=ImageDraw.Draw(im)
    d.text((30,20),'Fixed training10 response means at the common checkpoint',fill='black',font=TITLE)
    keys=['center_stage_1','center_stage_2','center_stage_3','center_stage_4','neighbor','background']
    labels=['Free','Transition','Attached','Edge-near','Neighbor negative','Background negative']
    colors={'B':(90,90,90),**COLORS}
    for j,(key,label) in enumerate(zip(keys,labels)):
        x=80+j*235;d.text((x-15,500),label,fill='black',font=SMALL)
        for k,v in enumerate(['B','C','D']):
            value=fit[v]['response_summary'][key]['macro_mean'];bh=value*360
            d.rectangle((x+k*52,465-bh,x+k*52+36,465),fill=colors[v])
            d.text((x+k*52-4,465-bh-22),f'{value:.3f}',fill=colors[v],font=SMALL)
            d.text((x+k*52+10,470),v,fill=colors[v],font=SMALL)
    d.text((35,555),'Fixed 0..1 response axis (360 px per unit); macro over images with that labeled group. Response is not identity confidence.',fill=(60,60,60),font=SMALL)
    im.save(HERE/'training_response_means.png')


if __name__=='__main__':main()
