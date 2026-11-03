"""Post-hoc fair-domain finite-line G0. Does not rerun or edit frozen detectors.

Shared: final saved free geometry, search start, x step/budget, image margins,
bounded convex visibility envelope, and the short allowed outside-envelope run.
Different by design: G0 keeps its line/width and has no image-evidence scoring,
curved candidates, strand/occlusion cues or confidence-based stopping.
"""
from pathlib import Path
import hashlib
import json
import math
import cv2
import numpy as np

import whole_io as io


HERE = Path(__file__).resolve().parent
OUT = HERE / 'evaluation_g0_001'
FINAL = HERE / 'all_001'


def visibility_envelope(original_surface):
    # Exact final whole_path.py rule, independently reproduced for evaluation.
    surface = original_surface.copy()
    contours, _ = cv2.findContours(surface, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= max(50., .0001 * np.count_nonzero(surface))]
    if contours:
        hull = cv2.convexHull(np.vstack(contours)); envelope = np.zeros_like(surface)
        cv2.fillConvexPoly(envelope, hull, 255)
        ratio = np.count_nonzero(envelope) / max(1, np.count_nonzero(surface))
        if ratio <= 1.8: surface |= envelope
        else: ratio = 1.
    else: ratio = 1.
    return surface, float(ratio)


def ribbon(shape, xy, width):
    result = np.zeros(shape, np.uint8); xy = np.asarray(xy, float)
    for a, b in zip(xy[:-1], xy[1:]):
        delta = b-a; length = float(np.linalg.norm(delta))
        if length < .01: continue
        normal = np.array([-delta[1], delta[0]]) / length
        points = np.rint([a+normal*width/2, b+normal*width/2,
                          b-normal*width/2, a-normal*width/2]).astype(np.int32)
        cv2.fillConvexPoly(result, points, 255)
    return result


def finite_line(surface, free, parameters):
    empty = np.zeros(surface.shape, np.uint8); geo = free.get('geometry', {})
    if free.get('status') != 'observed' or geo.get('status') != 'observed':
        return {'status': 'NO_UNAMBIGUOUS_SHARED_SEED', 'mask': empty, 'xy': [], 'search_xy': [],
                'reason': 'same_final_seed_eligibility_as_A3'}
    d = np.asarray(geo['direction_xy'], float); width = float(geo['width_px'])
    if abs(d[0]) < .25:
        return {'status': 'NO_ELIGIBLE_X_MONOTONIC_SEED', 'mask': empty, 'xy': [], 'search_xy': [],
                'reason': 'same_final_near_vertical_guard'}
    origin = np.asarray(geo['origin_xy']); end = np.asarray(geo['visible_end_xy'])
    sign = float(np.sign(d[0])); slope = float(d[1]/d[0]); step = float(parameters['step_x'])
    start = end-d*min(width*1.5, max(0., float(np.dot(end-origin, d))))
    h,w = surface.shape
    entered = bool(surface[int(np.clip(start[1],0,h-1)),int(np.clip(start[0],0,w-1))])
    maxsteps = int(min(2500.,width*parameters['maximum_search_widths'])/step)
    mask_float = (surface>0).astype(np.float32)
    outside = 0.; path = [start.tolist()]; reason = 'bounded_search'
    for k in range(1, maxsteps+1):
        x = float(start[0]+sign*step*k)
        if x < width or x >= w-width: reason = 'image_side'; break
        y = float(start[1]+sign*step*k*slope)
        if y < width or y >= h-width: reason = 'image_vertical_side'; break
        inside = bool(cv2.remap(mask_float,np.array([[x]],np.float32),np.array([[y]],np.float32),
                               cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)[0,0] > .3)
        entered |= inside
        outside = 0. if inside else outside+math.hypot(step,step*slope)
        if entered and not inside and outside >= width*.8:
            reason = 'surface_visibility_end'; break
        path.append([x,y])
    ends = np.asarray(free['free_segment']['observed_segment_xy']); outer = ends[np.argmin(ends@d)]
    prefix = np.linspace(outer,start,max(2,int(np.linalg.norm(outer-start)/step)))
    full = np.vstack([prefix,np.asarray(path)])
    return {'status': 'FINITE_LINE_CANDIDATE', 'mask': ribbon(surface.shape,full,width),
            'xy': full.tolist(), 'search_xy': path, 'start_xy': start.tolist(), 'end_xy': path[-1],
            'origin_xy': origin.tolist(), 'direction_xy': d.tolist(), 'width_px': width,
            'reason': reason, 'max_steps': maxsteps, 'step_x_px': step,
            'maximum_x_search_px': maxsteps*step, 'after_entering_allowed_outside_distance_px': width*.8,
            'image_margin_px': width, 'image_support_tested': False}


def overlay(image, mask):
    out = image.copy(); take=mask!=0
    out[take] = np.clip(out[take]*.6+np.asarray([220,65,220])*.4,0,255).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out,contours,-1,(230,60,230),1)
    return out


def crop(center, shape, size=(680,500)):
    w,h=size;x=int(np.clip(round(center[0]-w/2),0,shape[1]-w));y=int(np.clip(round(center[1]-h/2),0,shape[0]-h))
    return [x,y,w,h]


def comparison(images, labels, rectangle, output, max_panel_width=None):
    x,y,w,h=rectangle;panels=[]
    for image,label in zip(images,labels):
        a=image[y:y+h,x:x+w]
        if max_panel_width and w>max_panel_width:
            a=cv2.resize(a,(max_panel_width,max(1,round(h*max_panel_width/w))),interpolation=cv2.INTER_AREA)
        head=np.full((42,a.shape[1],3),18,np.uint8)
        cv2.putText(head,label,(6,17),cv2.FONT_HERSHEY_SIMPLEX,.42,(240,240,240),1,cv2.LINE_AA)
        cv2.putText(head,f'x={x},y={y}; crop={w}x{h}',(6,34),cv2.FONT_HERSHEY_SIMPLEX,.32,(180,180,180),1,cv2.LINE_AA)
        panels.append(np.vstack([head,a]))
    io.write_image(output,np.hstack(panels))


def main():
    if OUT.exists(): raise ValueError('Never overwrite an evaluation run')
    rows=io.selection('pilot');assert len(rows)==16
    final_context=io.read_json(FINAL/'run_start.json')
    for path,expected in final_context['sources_sha256'].items(): assert io.sha(path)==expected
    original_manifest=io.read_json(FINAL/'output_manifest.json')
    saved_hash={str(Path(f['path'])):f['sha256'] for f in original_manifest['files']}
    out_source={str(Path(__file__).resolve()):io.sha(__file__),str(HERE/'whole_io.py'):io.sha(HERE/'whole_io.py')}
    io.dump(OUT/'run_context.json',{'evaluation':'finite straight G0 with final A3 search visibility domain',
        'version':'IR-A3.0.1_posthoc_G0','selection':[r['output_file'] for r in rows],
        'final_run_context_sha256':io.sha(FINAL/'run_start.json'),'frozen_core_sha256':final_context['sources_sha256'],
        'evaluation_source_sha256':out_source,'prediction_rerun':False,
        'shared_rules':['final saved automatic free geometry','same seed eligibility','same search start',
          'same bounded convexhull rule','same x step and x search budget','same image margins',
          'same bilinear envelope membership threshold','same short allowed outside-envelope distance'],
        'differences_by_design':['G0 fixed straight centerline and measured constant width',
          'G0 has no image-evidence stopping, curve search or identity scoring'],
        'old_G0_warning':'Earlier runner G0 used original S rather than the final bounded convex envelope; its earlier termination could be caused by a narrower support domain, so it is not a fair domain comparison. Old files are retained untouched.'})
    cases=[]
    for row in rows:
        rel=Path(row['output_file']);role=rel.parent.name;stem=rel.stem
        record_path=FINAL/'记录'/role/(stem+'.json'); overlay_path=FINAL/'02_原图叠加'/role/rel.name
        for p in [record_path,overlay_path]: assert io.sha(p)==saved_hash[str(p.relative_to(FINAL))]
        record=io.read_json(record_path);surface_path=io.C4/'roi/surface_masks'/role/rel.name;original_path=io.DATA/rel
        assert io.sha(original_path)==row['output_sha256']
        assert io.sha(surface_path)==record['record']['input_dependency_sha256'][str(surface_path)]
        surface=io.read_image(surface_path,0);original=io.read_image(original_path)
        envelope,ratio=visibility_envelope(surface)
        expected_ratio=record['whole']['diagnostics'].get('visibility_envelope_area_ratio')
        if expected_ratio is not None: assert abs(ratio-expected_ratio)<1e-12
        result=finite_line(envelope,record['free'],record['whole']['diagnostics']['parameters'])
        g0=overlay(original,result['mask']);final=io.read_image(overlay_path)
        mask_path=OUT/'01_G0_mask'/role/rel.name;g0_path=OUT/'02_G0_overlay'/role/rel.name
        io.write_image(mask_path,result['mask']);io.write_image(g0_path,g0)
        io.write_image(OUT/'visibility_masks'/role/rel.name,envelope)
        chosen=next((c for c in record['whole']['candidates'] if c['candidate_id']==record['whole'].get('selected_candidate_id')),None)
        bound=np.argwhere((surface!=0)|(result['mask']!=0))
        if len(bound):
            y0,x0=np.maximum(bound.min(axis=0)-60,0);y1,x1=np.minimum(bound.max(axis=0)+61,surface.shape)
            overview=[int(x0),int(y0),int(x1-x0),int(y1-y0)]
        else:overview=[500,1400,3000,1600]
        previews={};labels=['Original RGB','G0 same domain: geometry only','Final A3 saved prediction']
        p=OUT/'03_comparisons'/role/(stem+'_overview.jpg');comparison([original,g0,final],labels,overview,p,640)
        previews['overview']={'path':str(p),'rectangle':overview,'resized_for_display':True}
        for label,end in [('g0_tail',result.get('end_xy')),('final_tail',chosen['xy'][-1] if chosen else None)]:
            if end is None:continue
            rect=crop(end,surface.shape);p=OUT/'03_comparisons'/role/(stem+'_'+label+'.jpg')
            comparison([original,g0,final],labels,rect,p)
            previews[label]={'path':str(p),'rectangle':rect,'resized_for_display':False}
        metrics={'final_status':record['whole']['status'],'g0_status':result['status'],'g0_stop_reason':result['reason'],
            'g0_end_xy':result.get('end_xy'),'final_end_xy':chosen['xy'][-1] if chosen else None,
            'final_stop_reason':chosen['end_reason'] if chosen else None,'visibility_area_ratio':ratio,
            'final_visibility_ratio_matches':expected_ratio is None or abs(ratio-expected_ratio)<1e-12}
        if chosen and result.get('direction_xy'):
            d=np.asarray(result['direction_xy']);n=np.array([-d[1],d[0]]);origin=np.asarray(result['origin_xy'])
            deviations=abs((np.asarray(chosen['xy'])-origin)@n)
            metrics['final_maximum_normal_deviation_from_G0_px']=float(deviations.max())
            metrics['endpoint_separation_px']=float(np.linalg.norm(np.asarray(chosen['xy'][-1])-np.asarray(result['end_xy'])))
        case={'output_file':row['output_file'],'metrics':metrics,'previews':previews,
            'input_sha256':{str(p):io.sha(p) for p in [original_path,surface_path,record_path,overlay_path]},
            'g0_mask_sha256':io.sha(mask_path),'g0_overlay_sha256':io.sha(g0_path),
            'free_geometry_exactly_from_final_record':record['free']['geometry'],
            'g0':{k:v for k,v in result.items() if k!='mask'},'actually_viewed':False}
        io.dump(OUT/'records'/role/(stem+'.json'),case);cases.append(case)
    for path,expected in final_context['sources_sha256'].items(): assert io.sha(path)==expected
    io.dump(OUT/'index.json',{'cases':cases,'planned':16,'completed':16,'frozen_sources_unchanged':True,
        'same_input_and_geometry_domain':True,'accuracy':None,'equipment_access':False,'reserved_inputs':False})
    print('Saved fixed16 fair-domain G0 comparisons; no frozen source changed.')


if __name__=='__main__':main()
