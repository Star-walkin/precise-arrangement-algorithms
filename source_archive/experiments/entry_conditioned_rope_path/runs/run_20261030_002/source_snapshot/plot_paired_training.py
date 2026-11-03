"""Plot saved A/B logs only; no network execution or evaluation-label access."""
from pathlib import Path
import argparse,hashlib,json,math,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def font(n):return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
def nested(row,path):
    for p in path:
        if row is None:return None
        row=row.get(p)
    return row


def smooth(vals,window=20):
    vals=np.asarray([np.nan if v is None else v for v in vals],float)
    return [float(np.nanmean(vals[max(0,i-window+1):i+1])) if np.any(np.isfinite(vals[max(0,i-window+1):i+1])) else None
            for i in range(len(vals))]


def panel(draw,rect,series,title,common,maxstep,fixed=None):
    x0,y0,x1,y1=rect
    draw.text((x0,y0-55),title,fill=(25,34,47),font=font(24))
    finite=[v for _,rows,path in series for r in rows if (v:=nested(r,path)) is not None]
    ymax=fixed or max(.01,math.ceil(max(finite or [.01])*20)/20)
    for j in range(5):
        y=y1-(y1-y0)*j/4
        draw.line((x0,y,x1,y),fill=(215,222,232),width=1)
        draw.text((x0-61,y-10),f'{ymax*j/4:.3f}',fill=(64,72,84),font=font(17))
    for j in range(5):
        step=round(maxstep*j/4);x=x0+(x1-x0)*step/maxstep
        draw.text((x-15,y1+10),str(step),fill=(64,72,84),font=font(17))
    colors={'A':((158,185,216),(25,86,163)),'B':((232,179,143),(197,88,27))}
    for name,rows,path in series:
        values=[nested(r,path) for r in rows];avg=smooth(values)
        for data,col,width in [(values,colors[name][0],1),(avg,colors[name][1],3)]:
            pts=[(x0+(x1-x0)*r['step']/maxstep,y1-(y1-y0)*v/ymax) for r,v in zip(rows,data) if v is not None]
            if len(pts)>1:draw.line(pts,fill=col,width=width)
    cx=x0+(x1-x0)*common/maxstep
    for yy in range(int(y0),int(y1),12):draw.line((cx,yy,cx,min(y1,yy+6)),fill=(80,80,85),width=2)
    draw.rectangle(rect,outline=(90,101,119),width=1)


def summarize_fit(data):
    out={}
    for key in ('initial','final','initial_train','final_train'):
        if key not in data:continue
        rows=data[key]['items'];part={'macro_center_loss':data[key]['macro_A_center_loss'],'macro_B_total_loss':data[key]['macro_B_total_loss']}
        for group in ('center_high','center_stage_1','center_stage_2','center_stage_3','center_stage_4','neighbor','background'):
            means=[r['response'][group]['mean'] for r in rows if r['response'][group]['mean'] is not None]
            part[group+'_macro_response_mean']=float(np.mean(means)) if means else None
        for term in ('neighbor_loss','background_loss'):
            part[term+'_macro']=float(np.mean([r['B_objective'][term] for r in rows]))
        out[key]=part
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    run=args.run.resolve();pair=load(run/'paired_comparison.json');out=run/'training_diagnostics';out.mkdir(exist_ok=True)
    logs={v:[json.loads(line) for line in (run/f'branches/{v}/model/train_log.jsonl').read_text().splitlines()] for v in ('A','B')}
    result={v:load(run/f'branches/{v}/model/result.json') for v in ('A','B')};small=load(run/'smallcheck/result.json')
    commonfit=load(run/'common_training_fit.json')
    common=pair['main_comparison_step'];last=max(len(v) for v in logs.values())
    im=Image.new('RGB',(2360,1330),(250,251,253));d=ImageDraw.Draw(im)
    d.text((80,24),'Paired continuation: actual recorded losses and training-mask responses',fill=(22,30,44),font=font(35))
    d.text((80,76),f'A = original supervision (blue); B = added weak negatives (orange). Main comparison: saved step {common}.',fill=(40,48,61),font=font(25))
    d.text((80,112),'Pale lines: all recorded steps; solid lines: trailing 20-step mean. Dashed vertical line: maximum common saved update.',fill=(65,73,86),font=font(23))
    specs=[('Original center objective: stage-balanced MSE',[('A',logs['A'],['center_loss']),('B',logs['B'],['center_loss'])],None),
           ('B neighbor MSE (unweighted; lambda = 0.5)',[('B',logs['B'],['neighbor_loss'])],None),
           ('B background MSE (unweighted; lambda = 0.1)',[('B',logs['B'],['background_loss'])],None),
           ('Positive-center mean response, sampled image',[(v,logs[v],['response','center_high','mean']) for v in ('A','B')],1.),
           ('Neighbor mean response, same annotated pixels',[(v,logs[v],['response','neighbor','mean']) for v in ('A','B')],1.),
           ('Background mean response, same annotated pixels',[(v,logs[v],['response','background','mean']) for v in ('A','B')],1.)]
    for i,(title,series,fixed) in enumerate(specs):
        col=i%3;row=i//3;x0=80+col*780;y0=240+row*500
        panel(d,(x0,y0,x0+660,y0+360),series,title,common,last,fixed)
    d.text((80,1170),'A does not optimize neighbor/background groups; its responses on those fixed training masks are diagnostics only.',fill=(44,54,68),font=font(23))
    d.text((80,1210),'Empty neighbor groups: zero loss; response gaps are omitted. These training-mask curves are not independent identity accuracy.',fill=(100,48,38),font=font(23))
    d.text((80,1250),'Actual differing final steps are supplemental. Formal A/B prediction and scoring use the checkpoint pair at the dashed line.',fill=(44,54,68),font=font(23))
    figure=out/'paired_actual_loss_and_response_curves.png';im.save(figure)
    summary={'exit_code':0,'command':sys.argv,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'smallcheck':{'steps':small['steps'],'training_seconds':small['training_seconds'],**summarize_fit(small)},
             'branches':{v:{'steps':r['steps'],'training_seconds':r['training_seconds'],'stop_reason':r['stop_reason'],
                             'metrics_scope':'each branch final training-fit diagnostic; not common-step independent evaluation',**summarize_fit(r)} for v,r in result.items()},
             'main_common_step':common,'shared_input_hash_steps_verified':pair['verified_shared_steps'],
             'common_step_fixed_training_fit':{v:summarize_fit({'final':r['fit']})['final'] for v,r in commonfit['models'].items()},
             'common_fit_scope':'old is reference only; formal A/B have equal updates on fixed training masks, not independent identity evaluation',
             'plot':{'path':str(figure),'sha256':hashlib.sha256(figure.read_bytes()).hexdigest()},
             'no_model_execution':True,'no_evaluation_reference_read':True}
    (out/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'exit_code':0,'main_common_step':common,'plot':str(figure)}))


if __name__=='__main__':main()
