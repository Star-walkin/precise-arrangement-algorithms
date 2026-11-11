"""Optional translated-crop backend for the unchanged LC1/F3 observation gates.

This is not the default backend. With c3_record=None, F3 is run on a lossless
integer crop with the normalization halo retained. All measured coordinates
are translated back before the original SurfaceObserver gates execute.
No observations are cached, invented, fitted into missing regions or resized.
"""
from __future__ import annotations
import copy
import math
import time
import types
import numpy as np

import lc1_surface as _base
from f3_core_adapter import load_core


class SurfaceObserver(_base.SurfaceObserver):
    """Same constructor/step contract as lc1_surface.SurfaceObserver.

    Instances process frames serially. c3_record != None uses the full-image
    original backend because its absolute shape reference is not transformed.
    """
    def __init__(self,surface_mask_original,pitch_px,c3_record=None,*,band_halfwidth_pitches=4.0):
        super().__init__(surface_mask_original,pitch_px,c3_record,
                         band_halfwidth_pitches=band_halfwidth_pitches)
        self._crop_receipt={}
        # Reuse the exact original step body and every observation/count gate.
        # Only this private function's global core-call binding is replaced;
        # neither the original module globals nor vendor functions are patched.
        original_step=_base.SurfaceObserver.step
        namespace=original_step.__globals__.copy()
        namespace['detect_surface_core']=self._cropped_core
        self._same_step=types.FunctionType(original_step.__code__,namespace,
                                           '_surface_step_with_private_crop_call',
                                           original_step.__defaults__,original_step.__closure__)
        self._same_step.__kwdefaults__=original_step.__kwdefaults__

    def step(self,bgr,d=None,active_y=None,exclusion_mask=None):
        self._crop_receipt={}
        if self.c3_record is not None:
            result=super().step(bgr,d=d,active_y=active_y,exclusion_mask=exclusion_mask)
            result['diagnostics']['fast_crop']={'enabled':False,'fallback_reason':'c3_record_absolute_shape_reference_requires_original_backend'}
            return result
        result=self._same_step(self,bgr,d=d,active_y=active_y,exclusion_mask=exclusion_mask)
        result['diagnostics']['fast_crop']=self._crop_receipt
        result['diagnostics']['backend']='optional_lossless_crop_F3_core_original_count_gates'
        return result

    def _cropped_core(self,image,band_mask,pitch,*,exclusion_mask=None,c3_record=None):
        if c3_record is not None:
            raise AssertionError('Absolute C3 reference must use the original backend')
        yy,xx=np.nonzero(band_mask)
        if not len(xx):
            self._crop_receipt={'enabled':False,'fallback_reason':'empty_narrow_surface_mask'}
            return _base.detect_surface_core(image,band_mask,pitch,
                                             exclusion_mask=exclusion_mask,c3_record=None)
        core=load_core()
        normalization=core._module('normalization')
        shadow=core._module('shadow')
        norm_margin=int(math.ceil(float(normalization.PARAMETERS['normalization_margin_pitch'])*pitch))
        shadow_margin=int(math.ceil(float(shadow.PARAMETERS['margin_pitch'])*pitch))
        # F3 itself truncates normalization to bbox + norm_margin. Retaining that
        # whole rectangle reproduces its original reflect-border domain exactly.
        halo=max(norm_margin,shadow_margin)+2
        h,w=band_mask.shape
        x0=max(0,int(xx.min())-halo);x1=min(w,int(xx.max())+halo+1)
        y0=max(0,int(yy.min())-halo);y1=min(h,int(yy.max())+halo+1)
        old=np.zeros_like(band_mask) if exclusion_mask is None else exclusion_mask
        local=_base.detect_surface_core(np.ascontiguousarray(image[y0:y1,x0:x1]),
                                        np.ascontiguousarray(band_mask[y0:y1,x0:x1]),pitch,
                                        exclusion_mask=np.ascontiguousarray(old[y0:y1,x0:x1]),
                                        c3_record=None)
        output=dict(local)
        translated=[]
        for point in local.get('points',[]):
            q=dict(point);q['x']=int(point['x'])+x0;q['y']=float(point['y'])+y0
            translated.append(q)
        by_id={p['point_id']:p for p in translated}
        output['points']=translated
        output['tracks']=[{**track,'points':[by_id[p['point_id']] for p in track['points']]}
                          for track in local.get('tracks',[])]
        fit=dict(local.get('fit',{}))
        curves=[]
        for curve in fit.get('curves',[]):
            c=dict(curve)
            xy=np.asarray(curve.get('sampled_points_xy',[]),float).reshape(-1,2)
            c['sampled_points_xy']=(xy+np.array([x0,y0])).tolist()
            curves.append(c)
        fit['curves']=curves
        output['fit']=fit
        # Preserve exclusions outside the crop, including exclusions far from d.
        combined=old.copy()
        combined[y0:y1,x0:x1]|=local['exclusion_mask']
        output['exclusion_mask']=combined
        diag=copy.deepcopy(local.get('diagnostics',{}))
        diag['old_exclusion_pixels']=int(np.count_nonzero(old))
        diag['combined_exclusion_pixels']=int(np.count_nonzero(combined))
        diag['old_exclusion_preserved']=bool(np.all(combined[old>0]==255))
        if 'bbox_xyxy' in diag.get('normalization',{}):
            q=diag['normalization']['bbox_xyxy'];diag['normalization']['bbox_xyxy']=[q[0]+x0,q[1]+y0,q[2]+x0,q[3]+y0]
        if 'crop_xyxy' in diag.get('shadow',{}):
            q=diag['shadow']['crop_xyxy'];diag['shadow']['crop_xyxy']=[q[0]+x0,q[1]+y0,q[2]+x0,q[3]+y0]
        for node in diag.get('period',{}).get('period_nodes',[]):
            if 'original_x' in node:node['original_x']+=x0
        diag['crop_translation_only']=True
        output['diagnostics']=diag
        # Internal dense field arrays and fit rasters stay crop-local and are
        # not returned by SurfaceObserver.step. The public measured/fitted XY
        # and the exclusion mask above are full-image coordinates.
        self._crop_receipt={'enabled':True,'crop_xyxy':[x0,y0,x1,y1],
                            'integer_translation_xy':[x0,y0],
                            'halo_px':halo,'vendor_normalization_margin_px':norm_margin,
                            'vendor_shadow_margin_px':shadow_margin,
                            'input_pixels':h*w,'cropped_input_pixels':(y1-y0)*(x1-x0),
                            'cropped_input_fraction':float((y1-y0)*(x1-x0)/(h*w)),
                            'physical_image_borders_preserved':True,
                            'normalization_original_crop_domain_retained':True,
                            'original_short_bracket_dedup_unknown_count_gates_reused':True,
                            'new_boundary_pixels_invented':False,'resize':False,
                            'c3_record':None}
        return output


FastSurfaceObserver=SurfaceObserver


def _compare_permitted_first_frames():
    """Standalone optional two-open-frame equivalence/performance check."""
    import json
    from pathlib import Path
    import cv2
    root=Path(__file__).resolve().parent
    outdir=root/'surface_fast_probe_results';outdir.mkdir(exist_ok=True)
    rows=[]
    cv2.setNumThreads(1)
    for session,role in [('dual_20260923_113305','right'),('dual_20260924_111500','left')]:
        smoke=json.loads((root/'surface_smoke_results'/f'{session}_{role}.json').read_text(encoding='utf-8'))
        cap=cv2.VideoCapture(smoke['video']);ok,image=cap.read();cap.release()
        if not ok:raise ValueError('Cannot decode permitted development first frame')
        mask=cv2.imdecode(np.fromfile(smoke['surface_mask_source'],np.uint8),0)
        if role=='left':image=cv2.flip(image,1);mask=cv2.flip(mask,1)
        observers=[_base.SurfaceObserver(mask,smoke['pitch_px']),SurfaceObserver(mask,smoke['pitch_px'])]
        # Repeat once after lazy imports/kernel warmup; compare the same inputs.
        results=[];timings=[]
        for observer in observers:
            values=[]
            for _ in range(2):
                tick=time.perf_counter()
                result=observer.step(image,d=smoke['d'],active_y=smoke['active_y'])
                values.append(time.perf_counter()-tick)
            results.append(result);timings.append(values)
        a,b=results
        xa=np.asarray(a['crossings_y']);xb=np.asarray(b['crossings_y'])
        crossing_delta=float(np.max(abs(xa-xb))) if xa.shape==xb.shape and xa.size else (0. if xa.shape==xb.shape else None)
        raw_a=[(p[0],p[1]) for segment in a['boundary_polylines_xy'] for p in segment]
        raw_b=[(p[0],p[1]) for segment in b['boundary_polylines_xy'] for p in segment]
        gates=['count_valid','count_status','count_observed_grooves','observed_groove_count_lower_bound',
               'reasons','edge_y','edge_source','edge_current_frame_confirmed','local_pitch_px','local_pitch_source',
               'continuous_pitch_distance','continuous_pitch_distance_candidate']
        matches={key:a[key]==b[key] for key in gates}
        row={'session':session,'role':role,'frame':0,'d':smoke['d'],'active_y':smoke['active_y'],
             'input_scope':'same permitted first-frame API probe; active_y is the existing Hough smoke condition, not S1 validation',
             'timings_seconds':{'original':timings[0],'fast':timings[1]},
             'warm_speedup':timings[0][-1]/timings[1][-1],
             'all_count_and_reason_fields_exactly_match':all(matches.values()),'field_matches':matches,
             'raw_observed_points_exactly_match':raw_a==raw_b,
             'original_crossings':a['crossings_y'],'fast_crossings':b['crossings_y'],
             'maximum_crossing_delta_px':crossing_delta,
             'original_count_valid':a['count_valid'],'fast_count_valid':b['count_valid'],
             'original_reasons':a['reasons'],'fast_reasons':b['reasons'],
             'crop':b['diagnostics'].get('fast_crop'),
             'original_raw_point_count':a['diagnostics']['raw_point_count'],
             'fast_raw_point_count':b['diagnostics']['raw_point_count']}
        (outdir/f'{session}_{role}_comparison.json').write_text(json.dumps({'comparison':row,'original':a,'fast':b},ensure_ascii=False,indent=2),encoding='utf-8')
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    receipt={'scope':'Two permitted first frames only; optional backend, not promoted to default.',
             'reserved_media_read':False,'vendor_modified':False,'original_adapter_modified':False,'cases':rows}
    (outdir/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    _compare_permitted_first_frames()
