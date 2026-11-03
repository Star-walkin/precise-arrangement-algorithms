"""Read-only, post-evaluation V14 native-RGB comparison renderer.

No training, new model inference, threshold choice or score changes. Raw edge
argmax predictions remain independent unordered points, not repaired curves.
Holdout images/labels/caches are inaccessible here until current frozen gates
have passed. Historical artifact hashes are checked without global V11 verify,
which would otherwise open all held-out images before gate authorization.
"""
from pathlib import Path
import argparse
import datetime
from functools import lru_cache
import hashlib
import json

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from .v11_path_contract import strip_to_image
from .v14_boundary_metrics import decode_boundaries


LAB=Path(__file__).resolve().parent
BASE=LAB/'outputs/v14_joint_supervision'
MANIFEST=LAB/'outputs/centerline_v9/dataset/manifest.json'
CACHE_INDEX=LAB/'outputs/v11_path_identity/development_cache/index.json'
ARMS=('center_only','joint')
COLORS=dict(gt_center='#84fa96',gt_edge='#ffffff',center='#21ddff',
            low='#ff4fe1',high='#ffb53e',text='#e6eaf0')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def location(output,phase,arm):
    return output/('single' if phase=='single' else 'development')/arm


def _check_gate(output,phase,protocol_hash):
    """Fail before any requested holdout metadata/image/label/cache is opened."""
    gate_path=output/('gate_'+phase+'.json')
    if not gate_path.is_file():
        raise ValueError('Required frozen '+phase+' mechanism gate is absent')
    gate=read(gate_path)
    if (gate.get('protocol_sha256')!=protocol_hash or gate.get('decision',{}).get('passed') is not True
            or not gate['decision'].get('checks') or not all(v is True for v in gate['decision']['checks'].values())):
        raise ValueError('Required frozen '+phase+' mechanism gate did not pass')
    for arm in ARMS:
        path=location(output,phase,arm)/('evaluation_'+phase+'.json')
        if gate.get('evaluation_sha256',{}).get(arm)!=sha(path):
            raise ValueError('Stale gate evaluation: '+phase+' '+arm)
        evaluated=read(path)
        if evaluated.get('protocol_sha256')!=protocol_hash:
            raise ValueError('Gate evaluation belongs to a different protocol')
        folder=location(output,phase,arm)
        if (evaluated.get('checkpoint_sha256')!=sha(folder/'checkpoint.pt') or
                evaluated.get('exports_sha256')!=sha(folder/('exports_'+phase+'.json'))):
            raise ValueError('Gate inference provenance changed')
    return dict(path=str(gate_path.resolve()),sha256=sha(gate_path),phase=phase)


def verify_provenance(output,phase):
    if phase not in ('single','train','holdout'):
        raise ValueError('Unsupported visualization phase')
    protocol_path=output/'protocol.json'
    if not protocol_path.is_file():
        raise FileNotFoundError('Frozen V14 protocol does not exist: '+str(protocol_path))
    protocol=read(protocol_path)
    digest=sha(protocol_path)
    if protocol.get('version')!=14 or protocol.get('arms')!=list(ARMS):
        raise ValueError('Not the matched V14 trial')
    gates=[]
    if phase in ('train','holdout'):
        gates.append(_check_gate(output,'single',digest))
    if phase=='holdout':
        gates.append(_check_gate(output,'train',digest))
    # These snapshots contain sources and existing aggregate artifacts, not a
    # request to read raw held-out images or individual held-out annotations.
    for group in ('source_snapshot','artifact_snapshot'):
        for path,expected in protocol[group].items():
            if sha(path)!=expected:
                raise ValueError('Frozen V14 source/artifact changed: '+path)
    original=read(MANIFEST)
    expected=[s for s in original['samples'] if s['sample_id']==protocol['single_sample']] if phase=='single' else [
        s for s in original['samples'] if s['split']==phase]
    selected=read(output/(phase+'_manifest.json'))
    if selected!=dict(original,samples=expected):
        raise ValueError('Frozen phase manifest changed')
    if '20260925' in json.dumps(selected):
        raise ValueError('9/25 blind sources are prohibited')
    stages={}
    for arm in ARMS:
        folder=location(output,phase,arm)
        paths={name:folder/(name+'_'+phase+'.json') for name in ('exports','evaluation')}
        paths['report']=folder/'report.json'
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError('Run export and evaluation for BOTH arms before rendering: '+str(path))
        report,exports,evaluation=(read(paths[name]) for name in ('report','exports','evaluation'))
        checkpoint=folder/'checkpoint.pt'
        checkpoint_sha=sha(checkpoint)
        training_phase='single' if phase=='single' else 'train'
        steps=protocol['single_steps' if phase=='single' else 'development_steps']
        if (report['phase']!=training_phase or report['arm']!=arm or report['steps']!=steps or
                report['protocol_sha256']!=digest or report['checkpoint_sha256']!=checkpoint_sha):
            raise ValueError('Wrong fixed-budget checkpoint provenance')
        if (exports['phase']!=phase or exports['arm']!=arm or exports['protocol_sha256']!=digest or
                exports['checkpoint_sha256']!=checkpoint_sha or exports['training_report_sha256']!=sha(paths['report']) or
                exports.get('gt_used_for_inference') is not False):
            raise ValueError('Wrong exported prediction provenance')
        if (evaluation['phase']!=phase or evaluation['arm']!=arm or evaluation['protocol_sha256']!=digest or
                evaluation['checkpoint_sha256']!=checkpoint_sha or evaluation['exports_sha256']!=sha(paths['exports'])):
            raise ValueError('Wrong evaluated prediction provenance')
        wanted={s['sample_id'] for s in expected}
        if (len(exports['records'])!=len(wanted) or {r['sample_id'] for r in exports['records']}!=wanted or
                len(evaluation['rows'])!=len(wanted) or {r['sample_id'] for r in evaluation['rows']}!=wanted):
            raise ValueError('Export/evaluation sample scopes differ from frozen phase')
        stages[arm]=dict(report=report,exports=exports,evaluation=evaluation,
            provenance={str(p.resolve()):sha(p) for p in [*paths.values(),checkpoint]})
    for key in ('initial_state_sha256','sequence_sha256','input_sha256','cache_sha256','steps',
                'optimizer','learning_rate','torch_version','numpy_version'):
        if stages['center_only']['report'][key]!=stages['joint']['report'][key]:
            raise ValueError('Unmatched arms: '+key)
    return protocol,selected,stages,gates


def raw_edge_points(decoded,geometry):
    """Independent points only; never infer a connected/sorted boundary curve."""
    rows=np.asarray([[np.nan if v is None else v for v in side] for side in decoded['edge_rows']],float)
    valid=np.asarray(decoded['edge_valid'],bool)
    if rows.shape!=valid.shape or rows.shape!=(2,geometry['shape_hw'][1]):
        raise ValueError('Malformed evaluated edge coordinates')
    result=[]
    for side in range(2):
        keep=valid[side]&np.isfinite(rows[side])
        points=np.column_stack([np.flatnonzero(keep),rows[side,keep]])
        result.append(strip_to_image(points,geometry) if len(points) else np.empty((0,2)))
    return result


def at_fraction(pieces,fraction):
    lengths=[sum(float(np.linalg.norm(np.asarray(b)-a)) for a,b in zip(p[:-1],p[1:])) for p in pieces]
    total=sum(lengths)
    if total<=0:
        raise ValueError('No trusted reference available for requested display crops')
    target=fraction*total
    for piece,size in zip(pieces,lengths):
        if target<=size:
            p=np.asarray(piece,float)
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
            return np.array([np.interp(target,arc,p[:,k]) for k in (0,1)])
        target-=size
    return np.asarray(pieces[-1][-1],float)


def crop_for(center,width,height,image_size):
    w,h=image_size
    width,height=min(int(np.ceil(width)),w),min(int(np.ceil(height)),h)
    x=int(np.clip(round(center[0]-width/2),0,w-width))
    y=int(np.clip(round(center[1]-height/2),0,h-height))
    return [x,y,x+width,y+height]


@lru_cache(maxsize=12)
def font(size):
    for path in (Path('C:/Windows/Fonts/arial.ttf'),Path('C:/Windows/Fonts/msyh.ttc')):
        if path.exists():
            return ImageFont.truetype(str(path),size)
    return ImageFont.load_default(size=size)


def panel(image,crop,references,prediction=None,size=(600,325),zoom=False):
    """Draw onto the cropped RGB itself so no geometry leaks into letterboxes."""
    x,y,x1,y1=crop
    scale=min(size[0]/(x1-x),size[1]/(y1-y))
    dimensions=(round((x1-x)*scale),round((y1-y)*scale))
    rgb=image.crop(crop).resize(dimensions,Image.Resampling.LANCZOS)
    draw=ImageDraw.Draw(rgb)
    def xy(p):return ((p[0]-x)*scale,(p[1]-y)*scale)
    def lines(pieces,color,width):
        for piece in pieces:
            if len(piece)>1:draw.line([xy(p) for p in piece],fill=color,width=width)
    lines(references['centerline'],COLORS['gt_center'],2)
    lines(references['edge_flange'],COLORS['gt_edge'],1)
    lines(references['edge_surface'],COLORS['gt_edge'],1)
    if prediction is not None:
        lines(prediction['center_segments'],COLORS['center'],2)
        for points,color in zip(prediction['raw_edge_points'],(COLORS['low'],COLORS['high'])):
            for p in points:
                if x<=p[0]<=x1 and y<=p[1]<=y1:
                    u,v=xy(p)
                    radius=2 if zoom else 1
                    draw.ellipse((u-radius,v-radius,u+radius,v+radius),fill=color)
    outer=Image.new('RGB',size,'#141b26')
    origin=((size[0]-dimensions[0])//2,(size[1]-dimensions[1])//2)
    outer.paste(rgb,origin)
    draw=ImageDraw.Draw(outer)
    draw.rectangle((0,0,size[0]-1,size[1]-1),outline='#586476')
    if zoom:
        draw.rectangle((7,7,154,29),fill='#141b26')
        draw.text((11,10),f'{scale:.2f}x | 20 native px',font=font(13),fill='white')
        draw.line([(12,size[1]-16),(12+20*scale,size[1]-16)],fill='white',width=3)
    return outer


def figure(image,sample,phase,references,downstream,predictions,destination):
    all_points=np.vstack([np.asarray(p) for pieces in references.values() for p in pieces])
    low,high=all_points.min(0)-60,all_points.max(0)+60
    context=crop_for((low+high)/2,high[0]-low[0],high[1]-low[1],image.size)
    crops=[context]+[crop_for(at_fraction(downstream,f),256,180,image.size) for f in (.35,.80)]
    canvas=Image.new('RGB',(1840,1270),'#0e141c')
    draw=ImageDraw.Draw(canvas)
    draw.text((15,12),f'V14 {phase.upper()} | '+sample['sample_id'],font=font(22),fill='white')
    warning='TRAIN fitting mechanism test, NOT generalization' if phase in ('single','train') else 'DEVELOPMENT holdout, NOT blind acceptance'
    draw.text((15,43),warning+' | Native RGB, identical crops/scales; GT for display only',font=font(17),fill='#ffd178')
    draw.text((15,72),'Green = trusted human center; white = trusted human edges; cyan = predicted center;',font=font(16),fill='#e1e7ef')
    draw.text((15,95),'pink/orange = raw low/high edge argmax POINTS (not joined, sorted, repaired or identity-confirmed).',font=font(16),fill='#e1e7ef')
    titles=['RGB + trusted GT','center_only (edge head NOT supervised)','joint (+ boundary auxiliary supervision)']
    for k,title in enumerate(titles):draw.text((13+k*610,128),title,font=font(18),fill='white')
    row_titles=['Full trusted annotated context','Fixed native crop: 35% trusted downstream arc (DISPLAY only)',
                'Fixed native crop: 80% trusted downstream arc (DISPLAY only)']
    for j,crop in enumerate(crops):
        y=165+j*352
        draw.text((15,y),row_titles[j],font=font(15),fill='#ccd4df')
        canvas.paste(panel(image,crop,references,zoom=j>0),(10,y+23))
        for k,arm in enumerate(ARMS,1):
            canvas.paste(panel(image,crop,references,predictions[arm],zoom=j>0),(10+610*k,y+23))
    draw.text((15,1245),'Unknown GT regions are omitted, not connected. No event/layer/force timing or physical accuracy is confirmed.',font=font(16),fill='#ffd178')
    canvas.save(destination)
    return dict(path=str(destination.resolve()),sha256=sha(destination),crop_xyxy=crops,
        crop_selection='full trusted annotation bbox; fixed downstream arc fractions .35/.80, display only',
        brightness_adjusted=False,same_crop_and_scale=True,raw_edges_joined=False)


def render(output,phase='single',ids=None,run_name=None):
    protocol,manifest,stages,gates=verify_provenance(output,phase)
    wanted=ids or ([protocol['single_sample']] if phase=='single' else [])
    if not wanted or len(wanted)!=len(set(wanted)):
        raise ValueError('Explicit unique --ids required for train/holdout; no score-based automatic selection')
    samples={s['sample_id']:s for s in manifest['samples']}
    if not set(wanted)<=set(samples):
        raise ValueError('Requested IDs are outside the authorized frozen phase')
    if run_name is None:
        run_name=phase+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    if not run_name or Path(run_name).name!=run_name or run_name in ('.','..') or '/' in run_name or '\\' in run_name:
        raise ValueError('run-name must be one directory name')
    destination=output/'visuals'/run_name
    if destination.exists():raise FileExistsError('Never overwrite prior visual diagnostics: '+str(destination))
    cache={e['sample_id']:e for e in read(CACHE_INDEX)['entries']}
    from .centerline_review_v5 import validate_annotation,read_clean_image
    from .centerline_quality_v5 import load_label_quality
    from .centerline_path_metrics_v6 import clip_scoring_segments
    records=[]
    # No requested field image/label/cache is opened before all gates/provenance
    # and requested scope have been checked above.
    for sid in wanted:
        sample=samples[sid];entry=cache[sid]
        for key in ('image_sha256','annotation_sha256','quality_sha256','split','group_id'):
            if entry[key]!=sample[key]:raise ValueError('Changed source/cache provenance: '+key)
        if sha(entry['cache'])!=entry['cache_sha256'] or sha(sample['annotation_path'])!=sample['annotation_sha256']:
            raise ValueError('Requested cache/annotation changed')
        label=validate_annotation(read(sample['annotation_path']),sample)
        quality=load_label_quality(sample)
        if quality['quality_sha256']!=sample.get('quality_sha256'):raise ValueError('Trusted quality changed')
        trusted=quality.get('supervision_polylines',label['polylines'])
        ignore=label['ignore_polygons']+quality['precision_exclude_polygons']
        references={k:clip_scoring_segments([trusted.get(k,[])],ignore_polygons=ignore)
                    for k in ('centerline','edge_flange','edge_surface')}
        downstream=clip_scoring_segments([trusted['centerline']],ignore_polygons=ignore,
            free_seed=label['polylines']['free_seed'],downstream=True)
        if not downstream:raise ValueError('Requested diagnostic has no trusted downstream display reference')
        predictions={};artifacts={str(Path(sample['annotation_path']).resolve()):sample['annotation_sha256'],
            str(Path(entry['cache']).resolve()):entry['cache_sha256']}
        with np.load(entry['cache'],allow_pickle=False) as archive:valid=archive['valid_mask'].copy()
        for arm in ARMS:
            stage=stages[arm]
            pred=next(p for p in stage['exports']['records'] if p['sample_id']==sid)
            evaluated=next(r for r in stage['evaluation']['rows'] if r['sample_id']==sid)
            if sha(pred['path'])!=pred['sha256'] or pred['input_sha256']!=entry['inference_input_sha256']:
                raise ValueError('Evaluated exported prediction changed')
            with np.load(pred['path'],allow_pickle=False) as archive:
                independently=decode_boundaries(archive['edge_probability'],valid)
            raw=evaluated['boundary']['decoded']
            reference_rows=np.array([[0. if v is None else v for v in side] for side in raw['edge_rows']])
            if (not np.array_equal(independently['edge_valid'],raw['edge_valid']) or
                    not np.array_equal(independently['edge_rows'],reference_rows)):
                raise ValueError('Evaluated raw edge coordinates disagree with GT-independent decode')
            # Center gaps are the frozen decoder's gaps; each segment is drawn
            # separately. GT exclusion clips trusted references, not prediction.
            predictions[arm]=dict(center_segments=evaluated['decoded']['segments'],
                raw_edge_points=raw_edge_points(raw,entry['geometry']))
            artifacts[str(Path(pred['path']).resolve())]=pred['sha256']
        image=read_clean_image(sample)
        destination.mkdir(parents=True,exist_ok=True)
        path=destination/(sid+'.png')
        metadata=figure(image,sample,phase,references,downstream,predictions,path)
        records.append(dict(sample_id=sid,figure=metadata,image_sha256=sample['image_sha256'],
            annotation_sha256=sample['annotation_sha256'],quality_sha256=quality['quality_sha256'],
            input_sha256=entry['inference_input_sha256'],read_artifact_sha256=artifacts))
    report=dict(phase=phase,protocol_sha256=sha(output/'protocol.json'),renderer_sha256=sha(__file__),
        verification='current frozen V14 source/artifact snapshots and checkpoint/report/export/evaluation chains',
        phase_gates=gates,arm_artifact_sha256={arm:stage['provenance'] for arm,stage in stages.items()},
        records=records,training=False,new_inference=False,
        posthoc_edge_decode='recomputed from existing probabilities and IMAGE valid only; verified equal to evaluated coordinates',
        gt_used_for_display_only=True,scores_modified=False,hardware_access=False,force_command=None,
        holdout_requested=phase=='holdout',blind_0925_read=False)
    with (destination/'diagnostics.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    (destination/'README.md').write_text('# V14 只读事后对照图\n\n'
        '三列使用完全相同原图裁剪与缩放：人工参考、center_only、joint。白线为可信人工双边，绿色为人工中心；'
        '青线为已评估中心候选，粉/橙散点为原始两侧 argmax，未经排序、补线或位置修正。center_only 的双边头未受监督。\n\n'
        '未知人工区不连接；模型中心仅连接原 decoder 同一连续片段。仅作显示，不更改评分、训练或施力判断。'
        '放大框为可信下游弧长35%和80%的固定256×180原图像素范围，不根据模型错误挑选。'
        'single/train图不证明泛化；holdout也是开发留出，不是工业盲测。\n',encoding='utf-8')
    print(json.dumps(dict(output=str(destination.resolve()),images=[r['figure']['path'] for r in records]),ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE)
    parser.add_argument('--phase',choices=('single','train','holdout'),default='single')
    parser.add_argument('--ids',nargs='+')
    parser.add_argument('--run-name')
    args=parser.parse_args()
    render(args.output.resolve(),args.phase,args.ids,args.run_name)
