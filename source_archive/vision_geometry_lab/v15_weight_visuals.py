"""Separate posthoc V15 visual comparisons; no model execution or scoring.

Single/train show fixed V14 center_only(0), V14 joint(.5), V15 joint_low(.1)
beside trusted references, at identical crops and native-pixel scales. A later
authorized holdout comparison contains only center_only(0) and joint_low(.1):
the failed V14 .5 arm is NOT newly evaluated there.
"""
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from .v14_joint_visuals import panel,at_fraction,crop_for,font,sha


LAB=Path(__file__).resolve().parent
BASE=LAB/'outputs/v15_aux_weight'
PHASE_ARMS=dict(single=('center_only','joint','joint_low'),
                train=('center_only','joint','joint_low'),holdout=('center_only','joint_low'))
ARM_TITLES=dict(center_only='center_only | weight 0 (edges unsupervised)',
    joint='joint | weight 0.5 (frozen V14 reference)',
    joint_low='joint_low | weight 0.1 (V15)')


def phase_arms(phase):
    if phase not in PHASE_ARMS:
        raise ValueError('Unsupported visual phase')
    return PHASE_ARMS[phase]


def figure(image,sample,phase,references,downstream,predictions,destination):
    """Pure drawing: GT may choose display crops, never model inputs or scores.

    `predictions[arm]` is the evaluated center_segments/raw_edge_points bundle
    accepted by the V14 pure panel helper. Raw independent edges remain points;
    decoded center segment gaps and trusted-reference gaps are never joined.
    """
    arms=phase_arms(phase)
    if set(predictions)!=set(arms):
        raise ValueError('Exactly the authorized phase arms must be supplied; no implicit missing arm')
    pieces=[p for segments in references.values() for p in segments if len(p)>0]
    if not pieces:
        raise ValueError('No trusted reference for display')
    points=np.vstack([np.asarray(p) for p in pieces])
    low,high=points.min(0)-60,points.max(0)+60
    context=crop_for((low+high)/2,high[0]-low[0],high[1]-low[1],image.size)
    crops=[context]+[crop_for(at_fraction(downstream,f),256,180,image.size) for f in (.35,.80)]
    columns=len(arms)+1
    canvas=Image.new('RGB',(columns*610+10,1270),'#0e141c')
    draw=ImageDraw.Draw(canvas)
    draw.text((15,12),f'V15 {phase.upper()} | '+sample['sample_id'],font=font(22),fill='white')
    scope='TRAIN mechanism test, NOT generalization' if phase in ('single','train') else 'DEVELOPMENT holdout, NOT blind acceptance'
    draw.text((15,43),scope+' | Native RGB; identical crop / scale / reference; GT for display only',font=font(17),fill='#ffd178')
    draw.text((15,72),'Green = trusted human center; white = trusted human edges; cyan = predicted center;',font=font(16),fill='#e1e7ef')
    draw.text((15,95),'pink/orange = raw low/high argmax POINTS (not joined, sorted, repaired or rope-identity-confirmed).',font=font(16),fill='#e1e7ef')
    for k,title in enumerate(['RGB + trusted GT']+[ARM_TITLES[a] for a in arms]):
        draw.text((13+k*610,128),title,font=font(18),fill='white')
    row_titles=['Full trusted annotated context','Fixed native crop: 35% trusted downstream arc (DISPLAY only)',
                'Fixed native crop: 80% trusted downstream arc (DISPLAY only)']
    for j,crop in enumerate(crops):
        y=165+j*352
        draw.text((15,y),row_titles[j],font=font(15),fill='#ccd4df')
        canvas.paste(panel(image,crop,references,zoom=j>0),(10,y+23))
        for k,arm in enumerate(arms,1):
            canvas.paste(panel(image,crop,references,predictions[arm],zoom=j>0),(10+610*k,y+23))
    footer=('Unknown GT regions are omitted, not connected. No layer/event/force timing or physical accuracy is confirmed.'
            if phase!='holdout' else 'Only weight 0 / 0.1 authorized here. Weight 0.5 NOT evaluated on holdout. No deployment or physical confirmation.')
    draw.text((15,1245),footer,font=font(16),fill='#ffd178')
    canvas.save(destination)
    return dict(path=str(Path(destination).resolve()),sha256=sha(destination),crop_xyxy=crops,
        arm_order=list(arms),columns=columns,
        crop_selection='full trusted annotation bbox; fixed downstream arc fractions .35/.80, DISPLAY only',
        zoom_native_size_xy=[256,180],brightness_adjusted=False,same_crop_and_scale=True,
        raw_edges_joined=False,center_segments_bridged=False,gt_used_for_inference=False,
        old_weight_05_holdout_evaluated=False)
