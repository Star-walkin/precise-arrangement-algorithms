"""D4纯合成自检：不读实图、标签或设备；不改变算法源文件。

运行示例（工作目录为本文件所在目录）：
  D:/precise_arrangement/vision_prototype/.venv/Scripts/python.exe -B D4selfcheck.py
默认把带时间戳的JSON证据写到selfcheck_records；已有结果绝不覆盖。
这些检查验证机制、接口和拒绝规则，不是现场绳边准确率。
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent


def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def shadow_width_mechanism(sw,rc):
    im=np.full((120,240,3),200,np.uint8)
    im[29:31,20:220]=50;im[66:74,20:220]=50
    roi=np.zeros((120,240),np.uint8);roi[10:110,20:220]=255
    field=sw.build_shadow_field(im,im,roi,20,None)
    x0,y0=field['x0'],field['y0']
    def count(key,lo,hi):
        return int(np.count_nonzero(field[key][lo-y0:hi-y0,60-x0:180-x0]))
    values={'narrow_2px_binary_pixels':count('binary_shadow',25,35),
            'narrow_2px_cleaned_pixels':count('cleaned_shadow',25,35),
            'broad_8px_binary_pixels':count('binary_shadow',63,77),
            'broad_8px_cleaned_pixels':count('cleaned_shadow',63,77)}
    assert values['narrow_2px_cleaned_pixels']==0
    assert values['broad_8px_cleaned_pixels']>0
    assert np.all(field['cleaned_shadow']<=field['binary_shadow'])
    hh,ww=field['cleaned_shadow'].shape
    outside=roi[y0:y0+hh,x0:x0+ww]==0
    for key in ('binary_shadow','cleaned_shadow','valid_support'):
        assert field[key].dtype==np.uint8 and np.all(np.isin(field[key],[0,255]))
        assert not np.any(field[key][outside])
    for key in ('broad_score','body_score','width_px','tangent_slope','coarse_tangent_slope','prefiltered_gray'):
        assert field[key].dtype==np.float32 and field[key].shape==(hh,ww)
        assert np.all(np.isfinite(field[key])) and not np.any(field[key][outside])
    for key in ('broad_score','body_score'):
        assert float(field[key].min())>=0 and float(field[key].max())<=1
    return {**values,'binary_subset_and_array_contracts_passed':True}


def shadow_direction_and_empty(sw,rc):
    roi=np.zeros((120,240),np.uint8);roi[10:110,20:220]=255
    vertical=np.full((120,240,3),200,np.uint8);vertical[10:110,80:88]=50
    field=sw.build_shadow_field(vertical,vertical,roi,20,None)
    support=int(np.count_nonzero(field['valid_support'][30:90,75:95]))
    assert support==0
    flat=np.full_like(vertical,100)
    empty=sw.build_shadow_field(flat,flat,np.zeros_like(roi),20,None)
    assert empty['binary_shadow'].shape==(0,0) and empty['diagnostics']['status']=='empty_roi'
    f=sw.build_shadow_field(flat,flat,roi,20,None)
    assert not np.any(f['valid_support'])
    return {'vertical_band_supported_pixels':support,'flat_supported_pixels':0,
            'empty_roi_and_none_reference_passed':True}


def convex_variable_curvature(sw,rc):
    pts=[{'x':x,'y':50+.001*(x-40)**2+2e-6*(x-40)**3+2.5*np.sin(x/13),
          'score':.9,'width':6.,'slope':.002*(x-40)+6e-6*(x-40)**2,'component':1}
         for x in range(40,321,5)]
    xy,stats=rc._smooth_xy(pts,20)
    slope=np.diff(xy[:,1].astype(float));minimum=float(np.min(np.diff(slope)))
    rise=float(np.mean(slope[-40:])-np.mean(slope[:40]))
    # 输出为float32；极小负二阶差分可来自量化，不等于解析导数下降。
    assert minimum>=-5e-5 and rise>.4
    assert stats['derivative_decrease_after_projection']==0
    assert stats['convexity_prior_applied'] is True
    pava=rc._isotonic_increasing(np.asarray([1.,0.,3.]),np.asarray([1.,3.,2.]))
    assert np.allclose(pava,[.25,.25,3.])
    return {'minimum_float32_second_difference':minimum,
            'analytic_derivative_decrease':stats['derivative_decrease_after_projection'],
            'edge_minus_center_mean_slope':rise,'weighted_pava_known_answer':pava.tolist(),
            'fit_statistics':stats}


def raster_staircase_rejection(sw,rc):
    roi=np.full((100,100),255,np.uint8);unknown=np.zeros_like(roi)
    # 两端都不在(10,12)，但实际1px LINE_8阶梯经过该像素。
    staircase=np.asarray([[10.,10.],[11.,14.]],np.float32)
    unknown[12,10]=255
    bad_unknown=rc._raster_invalid_x(staircase,roi,unknown,0,0)
    assert 10 in bad_unknown
    unknown[:]=0;roi[12,10]=0
    bad_roi=rc._raster_invalid_x(staircase,roi,unknown,0,0)
    assert 10 in bad_roi
    return {'bad_unknown_columns':bad_unknown.tolist(),'bad_roi_columns':bad_roi.tolist(),
            'fixture_has_clear_endpoints_and_invalid_intermediate_raster_pixel':True}


def full_curve_rules(sw,rc):
    roi=np.full((220,360),255,np.uint8)
    points=[{'x':x,'y':75+.0006*(x-170)**2,'score':.9,'width':6.,
             'slope':.0012*(x-170),'component':1} for x in range(40,321,5)]
    shadow={'x0':0,'y0':0,'broad_score':np.full(roi.shape,.9,np.float32),
            'cleaned_shadow':roi.copy(),'body_score':np.zeros(roi.shape,np.float32)}
    guard={'body_barrier':np.zeros(roi.shape,np.float32),'unknown_mask':np.zeros(roi.shape,np.uint8)}
    results={}
    for mode in ('plain','body','unknown','duplicate','empty','short_gap','long_gap'):
        guard['body_barrier'][:]=0;guard['unknown_mask'][:]=0
        shadow['broad_score'][:]=.9;shadow['cleaned_shadow'][:]=255
        if mode=='body':guard['body_barrier'][:,170:190]=1
        if mode=='unknown':guard['unknown_mask'][:,170:190]=255
        if mode=='short_gap':shadow['broad_score'][:,170:182]=0;shadow['cleaned_shadow'][:,170:182]=0
        if mode=='long_gap':shadow['broad_score'][:,160:205]=0;shadow['cleaned_shadow'][:,160:205]=0
        tracks=[{'track_id':'a','points':points}]
        if mode=='duplicate':tracks.append({'track_id':'b','points':[{**p,'y':p['y']+3,'score':.4} for p in points]})
        if mode=='empty':tracks=[]
        output=rc.fit_validated_curves(tracks,roi,20,shadow,guard)
        count=len(output['curves'])
        if mode in ('body','unknown'):
            assert not np.any(output['smooth_mask'][:,170:190]) and count==2
        if mode in ('plain','duplicate','short_gap'):assert count==1
        if mode=='duplicate':assert output['curves'][0]['track_id']=='a'
        if mode=='empty':assert not np.any(output['smooth_mask']) and count==0
        if mode=='short_gap':assert np.any(output['inferred_mask'][:,170:182])
        if mode=='long_gap':
            assert not np.any(output['smooth_mask'][:,160:205]) and count==2
            assert any('long_image_evidence_gap' in r['reason'] for r in output['rejections'])
            assert output['diagnostics']['refit_attempts']>=1
        assert output['smooth_mask'].shape==roi.shape and output['smooth_mask'].dtype==np.uint8
        assert np.all(output['inferred_mask']<=output['smooth_mask'])
        results[mode]={'accepted_curves':count,'rejection_records':len(output['rejections']),
                       'refit_attempts':output['diagnostics']['refit_attempts'],
                       'inferred_pixels':int(np.count_nonzero(output['inferred_mask']))}
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    now=datetime.now(timezone.utc)
    target=args.output if args.output else ROOT/'selfcheck_records'/f"selfcheck_{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    if target.exists():raise FileExistsError(f'Will not overwrite an existing receipt: {target}')
    paths={'selfcheck':Path(__file__).resolve(),'shadow_width':ROOT/'shadow_width.py','ribbon_curves':ROOT/'ribbon_curves.py'}
    before={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    sw=_load('d4_selfcheck_shadow',paths['shadow_width']);rc=_load('d4_selfcheck_ribbon',paths['ribbon_curves'])
    cases=[]
    for test in (shadow_width_mechanism,shadow_direction_and_empty,convex_variable_curvature,
                 raster_staircase_rejection,full_curve_rules):
        start=time.perf_counter()
        try:
            values=test(sw,rc);case={'name':test.__name__,'status':'passed','values':values}
        except Exception as exc:
            case={'name':test.__name__,'status':'failed','error':str(exc),'traceback':traceback.format_exc()}
        case['elapsed_seconds']=time.perf_counter()-start;cases.append(case)
    after={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    unchanged=before==after;code=0 if unchanged and all(t['status']=='passed' for t in cases) else 1
    report={'created_utc':now.isoformat(),'scope':'pure synthetic mechanism and rejection-rule checks; no real images decoded',
            'accuracy_claim':'none: these checks do not measure real-image boundary accuracy, material identity, contact, or device safety',
            'source_files':{name:{'path':str(paths[name]),'sha256_before':before[name],'sha256_after':after[name]} for name in paths},
            'sources_unchanged':unchanged,'cases':cases,'passed_cases':sum(t['status']=='passed' for t in cases),
            'total_cases':len(cases),'exit_code':code,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__,
                           'platform':platform.platform(),'executable':sys.executable},
            'command_argv':sys.argv,'reproduce_command':f'"{sys.executable}" -B "{paths["selfcheck"]}"'}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({'receipt':str(target.resolve()),'passed':report['passed_cases'],'total':len(cases),'exit_code':code,'source_sha256':before},ensure_ascii=True))
    return code


if __name__=='__main__':
    raise SystemExit(main())
