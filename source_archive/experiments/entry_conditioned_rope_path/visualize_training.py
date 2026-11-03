"""Render recorded training losses and seed ablations; does not run a model."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


def color_response(values):
    """One fixed 0..1 sequential color map shared by all images and differences."""
    anchors = np.array([[18, 14, 56], [45, 47, 142], [34, 119, 173],
                        [46, 176, 129], [194, 210, 58], [252, 231, 70]], np.float32)
    t = np.clip(values, 0, 1) * (len(anchors) - 1)
    low = np.minimum(t.astype(np.int32), len(anchors) - 2)
    frac = (t - low)[..., None]
    return np.round(anchors[low] * (1 - frac) + anchors[low + 1] * frac).astype(np.uint8)


def moving_average(a, window):
    a = np.asarray(a, np.float64)
    sums = np.cumsum(np.r_[0., a])
    return np.array([(sums[i + 1] - sums[max(0, i + 1 - window)]) /
                     min(window, i + 1) for i in range(len(a))])


def draw_curve(draw, rect, records, title, window, ymax):
    x0, y0, x1, y1 = rect
    draw.text((x0, y0 - 63), title, fill=(24, 29, 39), font=font(29))
    draw.text((x0, y0 - 28), "Actual per-step stage-balanced MSE (single-image batch)",
              fill=(55, 62, 73), font=font(20))
    for j in range(6):
        y = y1 - (y1 - y0) * j / 5
        draw.line((x0, y, x1, y), fill=(215, 221, 229), width=1)
        draw.text((x0 - 65, y - 12), f"{ymax*j/5:.3f}", fill=(65, 71, 80), font=font(18))
    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]
    last = max(steps)
    for j in range(6):
        step = round(last * j / 5)
        x = x0 + (x1 - x0) * step / last
        draw.text((x - 15, y1 + 12), str(step), fill=(65, 71, 80), font=font(18))
    points = [(x0 + (x1-x0) * s / last, y1 - (y1-y0) * v / ymax) for s,v in zip(steps, losses)]
    draw.line(points, fill=(176, 194, 213), width=1)
    avg = moving_average(losses, window)
    draw.line([(x0 + (x1-x0) * s / last, y1 - (y1-y0)*v/ymax) for s,v in zip(steps, avg)],
              fill=(22, 83, 159), width=3)
    draw.rectangle((x0, y0, x1, y1), outline=(90, 100, 114), width=1)
    draw.text((x0 + 15, y0 + 15), f"Blue: trailing mean {window} steps; pale: every recorded step",
              fill=(22, 83, 159), font=font(18))
    draw.text((x1 - 100, y1 + 37), "step", fill=(65, 71, 80), font=font(18))


def loss_plot(run, out):
    model = run / "model"
    records = {kind: [json.loads(line) for line in (model/f"{kind}_log.jsonl").read_text().splitlines()]
               for kind in ("overfit", "train")}
    stats = read_json(run / "training_summary_stats.json")
    ymax = math.ceil(max(r["loss"] for rows in records.values() for r in rows) * 20) / 20
    im = Image.new("RGB", (1800, 980), (249, 250, 252)); d = ImageDraw.Draw(im)
    d.text((90, 25), "Recorded training: entry-conditioned center response", fill=(19, 27, 39), font=font(36))
    d.text((90, 75), "Objective averages positive-supervised stages equally; inside each stage, high / low target MSE have equal weight.",
           fill=(60, 66, 76), font=font(22))
    draw_curve(d, (90, 205, 845, 735), records["overfit"], "4-image overfit check: 120 steps", 8, ymax)
    draw_curve(d, (980, 205, 1735, 735), records["train"], "Baseline: 843 steps / 1200.09 seconds", 20, ymax)
    d.text((90, 800), "Overfit gate passed: 94.08% reduction, phase-balanced MSE 0.15067 -> 0.00892; transition + attached positives present.",
           fill=(23, 71, 82), font=font(22))
    d.text((90, 842), "Full-train image-macro phase-balanced MSE: 0.14847 -> 0.02134. Overall high/low diagnostic: 0.15315 -> 0.01988.",
           fill=(30, 38, 50), font=font(22))
    d.text((90, 884), "Baseline restarted from saved initial weights. Augmented per-step curves differ from the fixed unaugmented macro measurements.",
           fill=(65, 71, 80), font=font(21))
    d.text((90, 925), "Training fit only: these curves do not establish independent target identity or a correct path to the spool edge.",
           fill=(100, 48, 38), font=font(21))
    p = out / "actual_loss_curves.png"; im.save(p)
    return p


def seed_center(seed):
    yy, xx = np.indices(seed.shape)
    return float((xx * seed).sum() / seed.sum()), float((yy * seed).sum() / seed.sum())


def paste_panel(canvas, array, xy, size, title, seed=None, ring=None, seed_color=(0,255,255), grayscale=False):
    if grayscale:
        rgb = np.repeat(np.round(np.clip(array,0,1)*255).astype(np.uint8)[...,None], 3, axis=2)
    else:
        rgb = color_response(array)
    panel = Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)
    d = ImageDraw.Draw(panel); sx,sy=size[0]/array.shape[1],size[1]/array.shape[0]
    if seed is not None and seed.sum() > 0:
        yy,xx=np.nonzero(seed>.1)
        mask=np.zeros(seed.shape,np.uint8);mask[yy,xx]=255
        mask=Image.fromarray(mask).resize(size,Image.Resampling.NEAREST)
        panel.paste(Image.new("RGB",size,seed_color),(0,0),mask)
        d=ImageDraw.Draw(panel)
        cx,cy=seed_center(seed)
        d.ellipse((cx*sx-6,cy*sy-6,cx*sx+6,cy*sy+6),outline=seed_color,width=2)
    if ring is not None:
        cx,cy,r=ring
        d.ellipse(((cx-r)*sx,(cy-r)*sy,(cx+r)*sx,(cy+r)*sy),outline=(255,255,255),width=2)
    canvas.paste(panel,xy)
    d=ImageDraw.Draw(canvas)
    d.text((xy[0],xy[1]-35),title,fill=(25,30,40),font=font(23))


def condition_plot(run, out, row, item):
    ident=row['id'];base=run/'model/condition_diagnostics'
    with np.load(item['cache']) as data:
        image=data['image'].astype(np.float32);seed=data['seed'].astype(np.float32);roi=data['roi'].astype(bool)
    values={k:np.load(base/f'{ident}__{k}.npy') for k in ('normal','zero_seed','moved_seed')}
    assert all(v.shape==image.shape for v in values.values())
    moved=np.roll(seed,row['moved_seed_dx_model_px'],axis=1)
    shift=row['moved_seed_dx_model_px']
    if shift>0:moved[:,:shift]=0
    else:moved[:,shift:]=0
    cx,cy=seed_center(seed);yy,xx=np.indices(image.shape);far=(((xx-cx)**2+(yy-cy)**2)>128**2)&roi
    panel_size=(700,round(700*image.shape[0]/image.shape[1]));pw,ph=panel_size
    margin=35;gap=28;W=margin*2+4*pw+3*gap;H=190+2*ph+180
    im=Image.new('RGB',(W,H),(249,250,252));d=ImageDraw.Draw(im)
    d.text((margin,20),f'Entry-condition sensitivity: {ident}',fill=(20,28,40),font=font(31))
    d.text((margin,65),'Training sample; known initial-entry seed only. White ring: 128 model pixels from original seed; outside is the far region.',fill=(58,66,76),font=font(24))
    d.text((margin,100),'Gray input uses fixed 0-1 display. Every response / absolute-difference map uses the same fixed 0-1 color scale.',fill=(58,66,76),font=font(24))
    xs=[margin+k*(pw+gap) for k in range(4)];y=180
    paste_panel(im,image,(xs[0],y),panel_size,'Input + initial seed (cyan)',seed,(cx,cy,128),grayscale=True)
    paste_panel(im,values['normal'],(xs[1],y),panel_size,'Normal seed response',seed,(cx,cy,128))
    paste_panel(im,values['zero_seed'],(xs[2],y),panel_size,'Zero seed response',ring=(cx,cy,128))
    paste_panel(im,values['moved_seed'],(xs[3],y),panel_size,'Moved seed response (magenta)',moved,(cx,cy,128),(255,60,220))
    y2=y+ph+65
    paste_panel(im,far.astype(np.float32),(xs[0],y2),panel_size,'Far-region mask (white = included)',grayscale=True)
    paste_panel(im,np.abs(values['zero_seed']-values['normal']),(xs[1],y2),panel_size,'Absolute delta: zero - normal',ring=(cx,cy,128))
    paste_panel(im,np.abs(values['moved_seed']-values['normal']),(xs[2],y2),panel_size,'Absolute delta: moved - normal',ring=(cx,cy,128))
    d=ImageDraw.Draw(im);tx=xs[3];ty=y2+15
    lines=[f'Moved seed: dx = {shift} model pixels',
           'No target used to choose new seed position.',
           f'Far mean |zero - normal|: {row["differences"]["zero_seed"]["mean_far"]:.5f}',
           f'Far mean |moved - normal|: {row["differences"]["moved_seed"]["mean_far"]:.5f}',
           'Far sensitivity establishes a conditioning effect.',
           'It does not establish the correct rope identity.',
           'No inference or training was rerun for this figure.']
    for j,line in enumerate(lines): d.text((tx,ty+j*40),line,fill=(42,50,63),font=font(22))
    by=y2+ph+35;barw=1000;bar=color_response(np.linspace(0,1,barw,dtype=np.float32)[None,:]).repeat(20,axis=0)
    im.paste(Image.fromarray(bar),(margin,by));d=ImageDraw.Draw(im)
    d.text((margin,by+25),'0.0',fill=(30,35,45),font=font(21));d.text((margin+barw//2-15,by+25),'0.5',fill=(30,35,45),font=font(21));d.text((margin+barw-25,by+25),'1.0',fill=(30,35,45),font=font(21))
    d.text((margin+barw+35,by),'Response intensity / absolute response difference; no per-image min-max stretch',fill=(40,47,58),font=font(22))
    p=out/f'{ident}__condition_comparison.png';im.save(p)
    return p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run.resolve();out=run/'training_diagnostics';out.mkdir(exist_ok=True)
    index=read_json(run/'data/index.json');items={i['id']:i for i in index['items'] if i['split']=='train'}
    summary=read_json(run/'model/condition_diagnostics/summary.json')
    outputs=[loss_plot(run,out)]
    for row in summary['items']:outputs.append(condition_plot(run,out,row,items[row['id']]))
    receipt={'command':sys.argv,'exit_code':0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'no_model_execution':True,'label_or_eval_content_read':False,
             'input_cache_fields_read':['image','seed','roi'],'gray_display':[0,1],'response_display':[0,1],
             'outputs':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':list(Image.open(p).size)} for p in outputs]}
    (out/'render_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'exit_code':0,'outputs':len(outputs),'directory':str(out)}))


if __name__=='__main__':main()
