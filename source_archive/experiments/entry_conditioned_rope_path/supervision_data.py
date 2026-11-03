"""Frozen training-side weak-negative additions for the 2026-10-30 A/B study.

No model predictions or evaluation annotations are opened. Coordinates below
were selected on the ten allowed original training images and native crops.
They are engineering weak annotations, never independent ground truth.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import cv2
import numpy as np

BASE=Path(__file__).resolve().parent
OLD=BASE/'runs/run_20261029_001'
DEFAULT_RUN=BASE/'runs/run_20261030_002'

# Image-visible fixed apparatus only. Not an outside-spool foreground rule.
LEFT_BG=[
 ('control_box',[[1040,1470],[1950,1470],[1950,1730],[1040,1730]]),
 ('left_diagonal_support',[[80,1250],[850,1580],[850,1650],[80,1330]]),
 ('right_diagonal_support',[[2190,1740],[2740,1520],[2765,1580],[2210,1810]])]
RIGHT_BG=[
 ('control_box_buttons',[[1710,1500],[2630,1500],[2630,1595],[1710,1595]]),
 ('left_diagonal_support',[[880,1500],[1400,1700],[1370,1760],[860,1570]])]

# Independently placed short visible rope-body fragments, not translated target
# copies, not automatic ridge peaks, and not the dark separators. Each is below
# the reviewed target in this image. The width is annotation support tolerance.
NEIGHBORS={
 'dual_20260924_111500__left__f00240':{
   'points_xy':[[1750,2290],[1800,2298],[1850,2305]],
   'review_box_xyxy':[1700,2225,1950,2360],
   'reason':'Separate bright rope-body band below reviewed target, bounded by dark separators and containing internal diagonal strand texture; not the target adjacent dark groove.',
   'uncertainty':'Short second discernible underlying band; immediate-neighbor numbering and full winding identity are not asserted.'},
 'dual_20260924_111143__right__f00130':{
   'points_xy':[[1730,2072],[1770,2062],[1810,2053],[1850,2043]],
   'review_box_xyxy':[1690,2010,1910,2185],
   'reason':'Visible bright center portion of a separate wound-rope band with two separators, below and spatially distinct from the reviewed incoming target.',
   'uncertainty':'Only this short visible band is a negative; no hidden continuation or calibrated rope width is supplied.'},
 'C02__dual_20260923_101517__pair0550__right':{
   'points_xy':[[1760,2163],[1800,2156],[1840,2148]],
   'review_box_xyxy':[1690,2080,1930,2260],
   'reason':'Clear underlying rope-body portion below the reviewed current target arc; white body and internal strand marks are visible instead of a separator-only dark line.',
   'uncertainty':'Crossing/occluded locations outside this fragment stay unknown; local body-center placement is an engineering approximation.'},
 'C01__dual_20260923_101517__pair0050__right':{
   'points_xy':[[1860,2530],[1900,2525],[1940,2520],[1980,2515]],
   'review_box_xyxy':[1790,2430,2050,2600],
   'reason':'Visible wound-rope body separated below the dark reviewed incoming target; its shallower local direction and two enclosing separators distinguish the local arcs.',
   'uncertainty':'Current target may still be suspended in projection; this is a local non-target arc proxy, not physical-contact or material-identity truth.'}}

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def length(lines):return float(sum(np.linalg.norm(np.diff(np.asarray(l,float),axis=0),axis=1).sum() for l in lines if len(l)>1))
def xy(points,item):return np.rint((np.asarray(points,float)-np.asarray(item['bbox'][:2]))*item['scale']).astype(np.int32)
def polygon_mask(polys,shape,item):
    m=np.zeros(shape,np.uint8)
    for p in polys:cv2.fillPoly(m,[xy(p,item)],1)
    return m.astype(bool)
def line_mask(points,shape,item,halfwidth_original=3.):
    m=np.zeros(shape,np.uint8);cv2.polylines(m,[xy(points,item)],False,1,1)
    return cv2.distanceTransform(1-m,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)<=halfwidth_original*item['scale']

def paired_positive_segments(lines,xlo,xhi):
    """Return only existing reviewed points clipped to the negative x-range."""
    out=[]
    for line in lines:
        run=[]
        for a,b in zip(np.asarray(line,float)[:-1],np.asarray(line,float)[1:]):
            if abs(b[0]-a[0])<1e-9:continue
            t0,t1=sorted([(xlo-a[0])/(b[0]-a[0]),(xhi-a[0])/(b[0]-a[0])]);lo=max(0,t0);hi=min(1,t1)
            if hi<=lo:continue
            aa=a+(b-a)*lo;bb=a+(b-a)*hi
            if run and np.linalg.norm(np.array(run[-1])-aa)>1e-5:out.append(run);run=[]
            if not run:run.append(aa.tolist())
            run.append(bb.tolist())
        if len(run)>1:out.append(run)
    return out

def render(item,im,old,neighbor,background,annotations,out):
    x0,y0,x1,y1=item['bbox'];raw=cv2.resize(im[y0:y1,x0:x1],(old['image'].shape[1],old['image'].shape[0]),interpolation=cv2.INTER_AREA)
    overlay=raw.copy();overlay[background]=(.55*overlay[background]+.45*np.array([255,0,255])).astype(np.uint8)
    overlay[neighbor]=(0,0,255);overlay[old['valid']&(old['target']<.5)]=(255,180,0);overlay[old['valid']&(old['target']>=.5)]=(0,255,0)
    cv2.polylines(overlay,[xy(item['known_seed_xy'],item)],False,(0,255,255),2)
    cv2.putText(overlay,'green: old center | cyan: old low | red: weak neighbor | magenta: weak background | others: ignore',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),1,cv2.LINE_AA)
    cv2.imencode('.png',overlay)[1].tofile(str(out/(item['id']+'__overlay.png')))
    n=annotations.get('neighbor')
    if n:
        b=n['review_box_xyxy'];xx0,yy0,xx1,yy1=b;crop=im[yy0:yy1,xx0:xx1].copy();ann=crop.copy()
        for l in annotations['reused_positive_pair_segments_xy']:
            cv2.polylines(ann,[np.rint(np.array(l)-[xx0,yy0]).astype(np.int32)],False,(0,255,0),1)
        cv2.polylines(ann,[np.rint(np.array(n['points_xy'])-[xx0,yy0]).astype(np.int32)],False,(0,0,255),2)
        # Raw beside marked image retains evidence hidden by annotation strokes.
        panel=np.concatenate([crop,ann],1);panel=cv2.resize(panel,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
        cv2.imencode('.png',panel)[1].tofile(str(out/(item['id']+'__neighbor_raw_and_marks.png')))

def build(run):
    run=Path(run).resolve();data=run/'data';dest=run/'supervision'
    for p in [data,dest,dest/'cache',dest/'annotations',dest/'overlays',dest/'minimal_annotation_requests']:p.mkdir(parents=True,exist_ok=True)
    assert not (data/'index.json').exists(),'Preserve frozen run; use a new run directory'
    old_index=load(OLD/'data/index.json');index=copy.deepcopy(old_index);train=[i for i in index['items'] if i['split']=='train']
    assert len(train)==10 and len({i['source_group'] for i in train})==4
    rows=[];annotation_rows=[];tick=time.time()
    for item in train:
        assert '20260925' not in item['image_path'];assert sha(item['image_path'])==item['image_sha256']
        with np.load(item['cache'],allow_pickle=False) as z:old={k:z[k] for k in z.files}
        assert old['image'].shape==(975,2048)
        source_path=OLD/'data/label_sources'/f"{item['id']}.json";source=load(source_path)
        center=source['trusted_center_segments_xy'];bg=LEFT_BG if item['id'].startswith('dual_20260924_111500') else RIGHT_BG
        background=polygon_mask([p for _,p in bg],old['image'].shape,item);neighbor=np.zeros_like(background)
        n=copy.deepcopy(NEIGHBORS.get(item['id']))
        if n:neighbor=line_mask(n['points_xy'],old['image'].shape,item)
        # Reviewed supervision outranks engineering weak additions. Candidate
        # additions overlapping the old support +4px guard are ignored only in
        # their new group, leaving A and B original center supervision identical.
        guard=cv2.dilate(old['valid'].astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        conflict=(neighbor|background)&guard
        extra_ignore=conflict.copy();neighbor &= ~guard&old['roi'];background &= ~guard&old['roi']&~neighbor
        assert not np.any(neighbor&background) and not np.any((neighbor|background)&old['valid'])
        assert not np.any((neighbor|background)&~old['roi'])
        cache=dest/'cache'/f"{item['id']}.npz"
        np.savez_compressed(cache,center_target=old['target'],center_valid=old['valid'],center_stage=old['stage'],
                            neighbor_mask=neighbor,background_mask=background,extra_ignore=extra_ignore)
        item['supervision_cache']=str(cache);item['supervision_cache_sha256']=sha(cache)
        pair=paired_positive_segments(center,min(p[0] for p in n['points_xy']),max(p[0] for p in n['points_xy'])) if n else []
        annotation={'id':item['id'],'split':'train','source_group':item['source_group'],'source_metadata':item['source_metadata'],
          'image_path':item['image_path'],'image_sha256':item['image_sha256'],'coordinate_system':'original_full_image_xy',
          'annotation_origin':'Codex original-image visual engineering weak annotation; not human/independent ground truth',
          'predictions_used_for_annotation_selection':False,'evaluation_images_labels_predictions_read':False,
          'reused_positive_source':str(source_path),'reused_positive_source_sha256':sha(source_path),
          'reused_positive_length_original_px':length(center),'new_positive_length_original_px':0.,
          'new_positive_segments_xy':[],'reused_positive_pair_segments_xy':pair,
          'background':[{'annotation_id':name,'type':'background_zero_response','polygon_xy':poly,
            'source':'original RGB visual engineering weak annotation',
            'accepted_reason':'Visible fixed control apparatus or support interior, away from active rope, winding surface and unknown continuation.',
            'uncertainty':'Only polygon interior is supervised; no extrapolation to all outside-spool or all dark pixels.'} for name,poly in bg],
          'neighbor':None if not n else {'annotation_id':'visible_competing_rope_body_short_fragment','type':'neighbor_zero_response',
            'source':'original RGB native-resolution visual weak annotation, checked against reviewed target location',
            'support_halfwidth_original_px':3.,'support_width_meaning':'annotation tolerance around visible non-target body portion, not true rope width',**n},
          'reviewed_center_preserved_bitwise':True,'conflict_policy':'New weak group ignores any overlap with old reviewed support +4px guard; center arrays unchanged',
          'conflict_new_group_ignored_model_pixels':int(conflict.sum()),'physical_contact_confirmed':False,
          'uncertain_continuation_policy':'no extra positive target; C07/C08 remain clipped at C7; unobserved/ambiguous tail stays ignore'}
        ann_path=dest/'annotations'/f"{item['id']}.json";dump(ann_path,annotation);item['supervision_annotation']=str(ann_path)
        im=cv2.imdecode(np.fromfile(item['image_path'],np.uint8),cv2.IMREAD_COLOR);render(item,im,old,neighbor,background,annotation,dest/'overlays')
        row={'id':item['id'],'source_group':item['source_group'],'reused_positive_length_original_px':length(center),
          'new_positive_length_original_px':0.,'neighbor_line_length_original_px':length([n['points_xy']]) if n else 0.,
          'neighbor_model_pixels':int(neighbor.sum()),'background_model_pixels':int(background.sum()),
          'original_center_valid_pixels':int(old['valid'].sum()),'new_conflict_ignored_pixels':int(conflict.sum()),
          'center_exactly_original':True,'stages':{str(s):{'center_high':int((old['valid']&(old['stage']==s)&(old['target']>=.5)).sum()),
            'neighbor':int((neighbor&(old['stage']==s)).sum()),'background':int((background&(old['stage']==s)).sum())} for s in range(1,5)}}
        rows.append(row);annotation_rows.append(annotation)
        # Minimal template requests only short target/competitor segments; no
        # whole-spool or whole-strand labeling is requested or fabricated.
        endpoint=np.array(center[-1][-1]);request_box=[int(max(0,endpoint[0]-220)),int(max(1050,endpoint[1]-180)),int(min(4096,endpoint[0]+220)),int(min(3000,endpoint[1]+250))]
        request={'id':item['id'],'image_path':item['image_path'],'image_sha256':item['image_sha256'],'purpose':'training-side short continuation identity and endpoint confirmation',
          'crop_xyxy':request_box,'coordinate_system':'original_full_image_xy','last_trusted_center_xy':endpoint.tolist(),
          'correct_target_after_competition_xy':[],'competing_neighbor_center_xy':[],'target_near_true_edge_xy':[],
          'unknown_regions_xy':[],'human_confirmed':False,'requested_minimum':'one correct center short segment beyond competition + one distinct competing short segment; edge anchor only if visible',
          'not_for_evaluation_input_hint':True,'use_until_confirmed':'ignore; do not train or score these empty template fields'}
        dump(dest/'minimal_annotation_requests'/f"{item['id']}.json",request)
        x0,y0,x1,y1=request_box;cv2.imencode('.png',im[y0:y1,x0:x1])[1].tofile(str(dest/'minimal_annotation_requests'/f"{item['id']}__raw.png"))
        print(json.dumps({'prepared_supervision':item['id'],'background':row['background_model_pixels'],'neighbor':row['neighbor_model_pixels'],'conflict':row['new_conflict_ignored_pixels']}),flush=True)
    index['previous_index']=str(OLD/'data/index.json');index['previous_index_sha256']=sha(OLD/'data/index.json')
    index['supervision_revision']='20261030 weak negatives; all old center target/valid/stage unchanged'
    index['supervision_scope']='10 training images only; existing 8 evaluation rows unchanged; no new target positives'
    dump(data/'index.json',index)
    totals={k:sum(r[k] for r in rows) for k in ['reused_positive_length_original_px','new_positive_length_original_px','neighbor_line_length_original_px','neighbor_model_pixels','background_model_pixels','original_center_valid_pixels','new_conflict_ignored_pixels']}
    summary={'items':rows,'totals':totals,'train_images':10,'train_source_groups':4,'neighbor_images':sum(r['neighbor_model_pixels']>0 for r in rows),
      'background_images':sum(r['background_model_pixels']>0 for r in rows),'new_positive_images':0,
      'actual_B_scope':'original reviewed center supervision + ten-image background weak negatives + four-image short neighboring-rope-body weak negatives; no new positive continuation',
      'limitations':['weak negative annotations are not independent evaluation truth','short neighbors do not establish whole-spool instance identity',
                    'confirmed post-competition target extensions and true endpoint references remain missing'],
      'command':sys.argv,'elapsed_seconds':time.time()-tick,'exit_code':0,'source_sha256':sha(__file__),
      'index_sha256':sha(data/'index.json'),'original_inputs_unchanged':True,'evaluation_annotation_or_prediction_content_read':False}
    dump(dest/'statistics.json',summary)
    dump(dest/'annotation_manifest.json',{'items':[{'id':a['id'],'path':str(dest/'annotations'/f"{a['id']}.json"),'sha256':sha(dest/'annotations'/f"{a['id']}.json")} for a in annotation_rows],
      'selection_basis':'original training images and existing reviewed local target geometry only; no current/old response-based relabeling'})
    (dest/'MINIMAL_ANNOTATION_REQUESTS.md').write_text('''# 最小补标缺口\n\n本轮新增目标正线长度为0；已复用可信旧中心，但不能将其算作新接续证据。新增背景和4个短邻绳可见主体负例均为工程弱标注，不是独立真值。\n\n优先检查 `minimal_annotation_requests/` 中两张代表图：`dual_20260924_111500__left__f00240` 和 `dual_20260924_111143__right__f00130`。每张只需确认竞争之后的正确目标中心短段及一段非目标邻绳；确实可见时再给目标近真实边缘锚点。模板提供原图坐标和最后可信点，空字段仍为ignore。C07/C08的C7后段不得自动补入。\n\n这些训练侧补标不得自动成为评价时额外入口提示；未来评价侧标注需独立保存并仅在冻结预测后评分。无需标完整卷面或每股钢丝。\n''',encoding='utf-8')
    print(json.dumps({'complete':True,'totals':totals}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(DEFAULT_RUN));build(p.parse_args().run)
