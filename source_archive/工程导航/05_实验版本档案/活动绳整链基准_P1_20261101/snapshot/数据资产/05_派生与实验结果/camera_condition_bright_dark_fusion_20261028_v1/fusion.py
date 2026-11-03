"""F1结果融合：坐标不移动、曲线不重拟合；亮暗分区只是选择来源。"""
import cv2
import numpy as np

def combine(data):
    s=data['surface']>0;b=s&(data['bright']>0);d=s&(data['dark']>0)&~b
    c3=(data['c3_smooth']>0)&b;d4=(data['d4_smooth']>0)&d
    merged=c3|d4
    inferred=((data['c3_inferred']>0)&b)|((data['d4_inferred']>0)&d)
    if np.any(inferred&~merged):raise ValueError('inferred must belong to boundary')
    owners={'C3':b,'D4':d};points=[];yellow=np.zeros(s.shape,np.uint8)
    # 只使用已接受曲线的黄点，不把拒绝组/拒绝子段重新显示出来。
    candidates=[]
    for g in data['c3_record']['fit']['groups']:
        if g.get('curve_id'):
            candidates.extend(('C3',float(x),float(y)) for x,y in g['observed_points_xy'])
    tracks={t['track_id']:t for t in data['d4_record']['new_tracks']}
    used=set()
    for c in data['d4_record']['fit']['curves']:
        for j in c['source_point_indices']:
            p=tracks[c['track_id']]['points'][j]
            if p['point_id'] not in used:
                used.add(p['point_id']);candidates.append(('D4',float(p['x']),float(p['y'])))
    for source in ('C3','D4'):
        layer=np.zeros(s.shape,np.uint8);seen=set()
        for origin,x,y in candidates:
            if origin!=source:continue
            xx,yy=int(round(x)),int(round(y))
            if (xx,yy) in seen or not(0<=yy<s.shape[0] and 0<=xx<s.shape[1]) or not owners[source][yy,xx]:continue
            seen.add((xx,yy));points.append({'source':source,'x':xx,'y':yy})
            cv2.circle(layer,(xx,yy),1,255,-1,cv2.LINE_8)
        layer[~owners[source]]=0;yellow|=layer
    masks={k:v.astype(np.uint8)*255 for k,v in {'surface':s,'bright_owner':b,'dark_owner':d,'uncovered':s&~(b|d),
      'c3_boundaries':c3,'d4_boundaries':d4,'merged':merged,'inferred':inferred}.items()}
    masks['accepted_yellow']=yellow
    metadata={'pixels':{k:int(np.count_nonzero(v)) for k,v in masks.items()},
      'input_roi_overlap_pixels':int(np.count_nonzero(s&(data['bright']>0)&(data['dark']>0))),
      'c3_boundary_clipped_pixels':int(np.count_nonzero((data['c3_smooth']>0)&~b)),
      'd4_boundary_clipped_pixels':int(np.count_nonzero((data['d4_smooth']>0)&~d)),
      'accepted_point_centers':len(points),
      'source_curve_ids':{'C3':['C3:'+str(c['curve_id']) for c in data['c3_record']['fit']['curves']],
                          'D4':['D4:'+str(c['curve_id']) for c in data['d4_record']['fit']['curves']]},
      'same_rope_identity_between_sources':None,'new_boundary_pixels_generated':0}
    return masks,points,metadata

def render(data,masks,points):
    """整幅卷面显示：青色正式线、紫色短虚线推断、黄色接受点。"""
    surface=masks['surface']>0;dark=masks['dark_owner']>0
    image=data['original'].copy();image[dark]=data['enhanced'][dark];image[~surface]=0
    supported=(masks['merged']>0)&(masks['inferred']==0)
    image[supported]=(230,190,35)
    # 只隐藏部分推断像素形成虚线，不生成额外连接。正式mask保留完整推断像素。
    dash=((np.arange(image.shape[1])//6)%2==0)[None,:]
    image[(masks['inferred']>0)&dash]=(210,65,210)
    for p in points[::2]:
        cv2.drawMarker(image,(p['x'],p['y']),(0,230,255),cv2.MARKER_CROSS,5,1,cv2.LINE_8)
    # 不画分区框，避免把亮暗交界画成新的绳边；全卷面外统一黑色。
    image[~surface]=0
    return image
