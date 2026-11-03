"""Fixed V9 paired-result illustrations, never an input to decoding/scoring.

Representatives and display rules are fixed before reading predictions.  Each
sample gets a full-frame overview plus a seed/trusted-reference detail view.
Partial annotations are not exhaustive identity labels or force validation.
"""
from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .evaluate_centerline_multiscale_v9 import (
    ROOT, _dataset, canonical_hash, load_prediction, V9_PROTOCOL,
)
from .prepare_centerline_v5 import file_hash
from .centerline_review_v5 import read_clean_image


REPRESENTATIVES=(
    ('train_old_f246','dual_20260924_111500__left__f00246','train'),
    ('train_C07','C07__R_run_stop_run_20260922_01__t012.033__f000368','train'),
    ('holdout_old_left_f550','dual_20260923_113745__left__f00550','holdout'),
    ('holdout_C04','C04__dual_20260923_113029__t05__left','holdout'),
    ('holdout_C05','C05__dual_20260923_113029__t25__left','holdout'),
)
DISPLAY_PROTOCOL=dict(version=1,representatives=[list(item) for item in REPRESENTATIVES],
    views=['full_image','manual_seed_and_trusted_reference_bbox_plus_96px'],
    detail_padding_native_px=96,panel_content_max_wh=[1000,720],
    probability_range=[0.,1.],probability_colormap='OpenCV INFERNO, RGB',
    probability_overlay_alpha=.35,pixel_center_mapping='map_xy=(full_xy+0.5)*[0.5,0.5]-0.5',
    trusted_reference='saved evaluation full_reference_segments after quality/ignore clipping',
    predictions='saved decoded segments; never joined across gaps or decoded again',
    manual_seed_used=True,partial_reference=True,identity_confirmed=False,force_validation=False,
    selection='fixed sample IDs, never selected by prediction quality')
GREEN=(50,255,110);MAGENTA=(255,60,210);YELLOW=(255,225,20);CYAN=(20,210,255)


def _font(size=16):
    path=Path('C:/Windows/Fonts/arial.ttf')
    return ImageFont.truetype(str(path),size) if path.exists() else ImageFont.load_default()


def fixed_rows(summary):
    """Exact IDs; never replace an absent representative with a prettier one."""
    rows={row['sample_id']:row for row in summary['rows']}
    if len(rows)!=len(summary['rows']):
        raise ValueError('Duplicate sample rows')
    selected={}
    for alias,identifier,split in REPRESENTATIVES:
        if identifier not in rows or rows[identifier]['split']!=split:
            raise ValueError('Missing/wrong-split fixed representative: '+identifier)
        selected[identifier]=rows[identifier]
    return selected


def detail_roi(image_shape_hw,seed,reference_segments):
    """GT-guided display crop ONLY; no prediction-dependent crop selection."""
    height,width=map(int,image_shape_hw)
    arrays=[np.asarray(seed,float).reshape(-1,2)]
    arrays.extend(np.asarray(segment,float).reshape(-1,2) for segment in reference_segments if len(segment))
    points=np.concatenate(arrays)
    if len(points)<2 or not np.isfinite(points).all():
        raise ValueError('Finite manual seed required for the fixed display crop')
    pad=DISPLAY_PROTOCOL['detail_padding_native_px']
    lower=np.maximum(0,np.floor(points.min(0)-pad)).astype(int)
    upper=np.minimum([width,height],np.ceil(points.max(0)+pad+1)).astype(int)
    return [int(lower[0]),int(lower[1]),int(upper[0]),int(upper[1])]


def probability_crop(probability,roi):
    """Interpolate native/2 maps at native pixel centres, with zero outside."""
    x0,y0,x1,y1=map(int,roi)
    xs=(np.arange(x0,x1,dtype=np.float32)+.5)*.5-.5
    ys=(np.arange(y0,y1,dtype=np.float32)+.5)*.5-.5
    mx,my=np.meshgrid(xs,ys)
    return cv2.remap(np.asarray(probability,np.float32),mx,my,cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def probability_rgb(probability):
    """Fixed 0..1 conversion, deliberately not normalized per image/arm."""
    values=np.asarray(probability,float)
    if not np.isfinite(values).all() or np.any(values<0) or np.any(values>1):
        raise ValueError('Probability values outside the fixed display scale')
    gray=np.rint(values*255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(gray,cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)


def _segments(draw,segments,roi,scale,offset,color,width):
    for segment in segments:
        points=[((float(p[0])-roi[0])*scale+offset[0],(float(p[1])-roi[1])*scale+offset[1]) for p in segment]
        if len(points)>=2:
            draw.line(points,fill=(0,0,0),width=width+2)
            draw.line(points,fill=color,width=width)


def _pct(value):
    return 'n/a' if value is None else f'{100*value:.1f}%'


def _arm_subtitle(row):
    score=row['metrics']['downstream']
    reason=row['decoded'].get('reason','unspecified')
    return f"downstream coverage {_pct(score['covered_fraction'])}; precision@5px {_pct(score['prediction_precision_at_5px'])}; {reason}"


def render_figure(image_rgb,seed,reference_segments,control,native,roi,*,sample_id,split,view_name):
    """Pure display: arms contain saved row + optional arrays, not models."""
    x0,y0,x1,y1=map(int,roi)
    crop=np.asarray(image_rgb,np.uint8)[y0:y1,x0:x1]
    if not crop.size:
        raise ValueError('Empty display ROI')
    max_width,max_height=DISPLAY_PROTOCOL['panel_content_max_wh']
    scale=min(max_width/crop.shape[1],max_height/crop.shape[0])
    display_size=(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale)))
    panel_w=1000;header_h=58;content_h=display_size[1];panel_h=header_h+content_h
    padding=12;top_h=115;bottom_h=88
    canvas=Image.new('RGB',(3*panel_w+4*padding,2*panel_h+top_h+bottom_h+3*padding),(25,28,34))
    draw=ImageDraw.Draw(canvas)
    title=f'{sample_id} | {split} | {view_name} | native ROI {list(roi)}'
    draw.text((padding,10),title,fill='white',font=_font(21))
    draw.text((padding,40),'MANUAL SEED required | PARTIAL trusted GT | NOT identity or force validation',fill=YELLOW,font=_font(21))
    draw.text((padding,72),'Green: trusted scored reference   Yellow: manual seed   Magenta: saved decoded segments (gaps retained)',fill='white',font=_font(18))
    panels=[('Original + trusted partial GT',None,None),('Control: body probability',control,'body'),
            ('Native: body probability',native,'body'),('Coverage: cyan=both; red=control only; blue=native only',None,'coverage'),
            ('Control: center probability',control,'center'),('Native: center probability',native,'center')]
    for index,(title,arm,kind) in enumerate(panels):
        col=index%3;row_index=index//3
        px=padding+col*(panel_w+padding);py=top_h+padding+row_index*(panel_h+padding)
        base=crop.copy();missing=False
        if kind=='coverage':
            masks=[]
            for source in (control,native):
                arrays=source.get('arrays')
                masks.append(np.zeros(crop.shape[:2],bool) if arrays is None else probability_crop(arrays['coverage'],roi)>=.999999)
            tint=np.zeros_like(base);both=masks[0]&masks[1]
            tint[both]=CYAN;tint[masks[0]&~masks[1]]=(255,40,40);tint[masks[1]&~masks[0]]=(40,80,255)
            base=np.rint(.5*base+.5*tint).astype(np.uint8)
        elif arm is not None:
            arrays=arm.get('arrays')
            if arrays is None:
                base=np.full_like(base,40);missing=True
            else:
                alpha=DISPLAY_PROTOCOL['probability_overlay_alpha']
                color=probability_rgb(probability_crop(arrays[kind],roi))
                base=np.rint((1-alpha)*base+alpha*color).astype(np.uint8)
        image=Image.fromarray(base).resize(display_size,Image.Resampling.LANCZOS)
        ox=px+(panel_w-display_size[0])//2;oy=py+header_h
        canvas.paste(image,(ox,oy))
        draw.rectangle((px,py,px+panel_w-1,py+panel_h-1),outline=(100,105,110),width=1)
        draw.text((px+8,py+5),title,fill='white',font=_font(18))
        if arm is not None:
            subtitle=_arm_subtitle(arm['row'])
            draw.text((px+8,py+31),subtitle,fill='white',font=_font(12))
        elif kind=='coverage':
            draw.text((px+8,py+31),'Coverage is diagnostic only. Outside windows remains in the scored reference denominator.',fill='white',font=_font(13))
        else:
            draw.text((px+8,py+31),'GT/quality used for display and score only. No algorithm points are human labels.',fill='white',font=_font(13))
        # Draw onto a clipped panel, so distant predictions cannot escape ROI.
        overlay=canvas.crop((ox,oy,ox+display_size[0],oy+display_size[1]))
        overlay_draw=ImageDraw.Draw(overlay)
        _segments(overlay_draw,reference_segments,roi,scale,(0,0),GREEN,2)
        _segments(overlay_draw,[seed],roi,scale,(0,0),YELLOW,3)
        if arm is not None:
            _segments(overlay_draw,arm['row']['decoded']['segments'],roi,scale,(0,0),MAGENTA,3)
        if missing:
            overlay_draw.text((20,20),'NO VALID EXPORTED MAP - retained, not replaced',fill='white',font=_font(20))
        canvas.paste(overlay,(ox,oy))
    footer_y=canvas.height-bottom_h+10
    draw.text((padding,footer_y),'Fixed representatives / crop and color rules. Detail ROI is manual-seed + trusted-reference bbox; overview shows the full frame.',fill='white',font=_font(17))
    bar=np.tile(np.linspace(0,1,500,dtype=np.float32),(18,1))
    canvas.paste(Image.fromarray(probability_rgb(bar)),(padding,footer_y+35))
    for value in (0.,.25,.5,.75,1.):
        label=f'{value:.2f}';font=_font(15)
        half_width=draw.textbbox((0,0),label,font=font)[2]/2
        draw.text((max(padding,padding+499*value-half_width),footer_y+55),label,fill='white',font=font)
    draw.text((padding+520,footer_y+35),'Fixed probability scale, both arms; heatmap blended at 35% over the original image.',fill='white',font=_font(16))
    return canvas


def _read_summary(path,mode,manifest_sha):
    value=json.loads(Path(path).read_text('utf-8-sig'))
    if (value.get('experiment_version')!=9 or value.get('mode')!=mode
            or value.get('manifest_sha256')!=manifest_sha
            or value.get('protocol')!=V9_PROTOCOL
            or value.get('protocol_sha256')!=canonical_hash(V9_PROTOCOL)):
        raise ValueError('Evaluation summary is not the fixed V9 protocol/data/arm')
    return value


def _load_arm(folder,record,row,rgb,snapshot,manifest_sha,mode):
    for key,expected in (('image_sha256',record['sample']['image_sha256']),
            ('annotation_sha256',record['annotation_sha256']),('quality_sha256',record['quality']['quality_sha256'])):
        if row.get(key)!=expected:
            raise ValueError('Evaluated source changed: '+key)
    if row['prediction_status']!='loaded':
        return dict(row=row,arrays=None)
    loaded=load_prediction(folder,record,rgb,snapshot,manifest_sha,mode)
    if loaded is None:
        raise ValueError('A scored export disappeared')
    meta,body,center,coverage=loaded
    for key,value in row['prediction_source'].items():
        if meta.get(key)!=value:
            raise ValueError('Export differs from saved evaluation: '+key)
    return dict(row=row,arrays=dict(body=body,center=center,coverage=coverage))


def render_results(manifest_path,control_summary,native_summary,control_exports,native_exports,output):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Use a new visualization directory; prior illustrations are immutable')
    manifest,records,snapshot=_dataset(manifest_path)
    manifest_sha=file_hash(manifest_path)
    summaries={mode:_read_summary(path,mode,manifest_sha) for mode,path in
               (('lowpass_control',control_summary),('native_detail',native_summary))}
    selected={mode:fixed_rows(value) for mode,value in summaries.items()}
    record_by_id={record['sample']['sample_id']:record for record in records}
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema_version=1,display_protocol=DISPLAY_PROTOCOL,
        display_protocol_sha256=canonical_hash(DISPLAY_PROTOCOL),visualizer_source_sha256=file_hash(__file__),
        manifest_sha256=manifest_sha,control_summary_sha256=file_hash(control_summary),
        native_summary_sha256=file_hash(native_summary),images=[],model_or_decoder_executed=False,
        identity_confirmed=False,force_validation=False)
    for alias,identifier,split in REPRESENTATIVES:
        record=record_by_id[identifier];rgb=np.asarray(read_clean_image(record['sample']))
        seed=record['label']['polylines']['free_seed']
        c_row=selected['lowpass_control'][identifier];n_row=selected['native_detail'][identifier]
        reference=c_row['metrics']['score_support']['full_reference_segments']
        if reference!=n_row['metrics']['score_support']['full_reference_segments']:
            raise ValueError('The paired evaluations do not share the same scored reference')
        control=_load_arm(control_exports,record,c_row,rgb,snapshot,manifest_sha,'lowpass_control')
        native=_load_arm(native_exports,record,n_row,rgb,snapshot,manifest_sha,'native_detail')
        rois=dict(overview=[0,0,rgb.shape[1],rgb.shape[0]],detail=detail_roi(rgb.shape[:2],seed,reference))
        for view_name,roi in rois.items():
            figure=render_figure(rgb,seed,reference,control,native,roi,sample_id=identifier,split=split,view_name=view_name)
            path=output/f'{alias}_{view_name}.png';figure.save(path)
            report['images'].append(dict(sample_id=identifier,split=split,view=view_name,roi_xyxy=roi,
                image=str(path.resolve()),sha256=file_hash(path),control_status=c_row['prediction_status'],
                native_status=n_row['prediction_status'],reference_is_partial=True))
    (output/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
    lines=['# Fixed V9 paired illustrations','',
        'Manual seed required. Green is the trusted, scored **partial** reference; magenta is the saved decoded path. '
        'These pictures do not confirm rope identity or force timing. No models or decoders were executed.',
        '', 'Representatives were fixed before predictions: old training f246, training C07, old held-out left f550, held-out C04/C05. '
        'Both arms use the same original-image ROI and 0–1 probability scale. The detail crop depends only on manual seed/reference, '
        'and every sample also has a full-frame view so erroneous extra paths are not silently cropped away.', '',
        'GT/quality influence only this illustration and the saved scores. The coverage panel does not remove unobserved reference from the score denominator.', '']
    for entry in report['images']:
        lines.append(f"- {entry['sample_id']} — {entry['view']}: [{Path(entry['image']).name}]({Path(entry['image']).name})")
    (output/'README.md').write_text('\n'.join(lines)+'\n','utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    base=ROOT/'vision_geometry_lab/outputs/centerline_v9'
    parser.add_argument('--manifest',type=Path,default=base/'dataset/manifest.json')
    parser.add_argument('--control-summary',type=Path,default=base/'lowpass_control_evaluation/summary.json')
    parser.add_argument('--native-summary',type=Path,default=base/'native_detail_evaluation/summary.json')
    parser.add_argument('--control-exports',type=Path,default=base/'lowpass_control_exports/predictions')
    parser.add_argument('--native-exports',type=Path,default=base/'native_detail_exports/predictions')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cv2.setNumThreads(2)
    result=render_results(args.manifest,args.control_summary,args.native_summary,args.control_exports,args.native_exports,args.output)
    print(json.dumps(dict(output=str(args.output),images=len(result['images']))))


if __name__=='__main__':
    main()
