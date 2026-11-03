"""One explicit post-audit derivative-unit correction, not threshold search.

Reuse V18 source frames, immutable background poses and semantic seeds. Preserve
V18 and its failed startup measurements. No ground truth, event rules or device.
"""
from __future__ import annotations
import argparse
import datetime
from pathlib import Path
import time

import cv2
import numpy as np

from vision_geometry_lab import v18_front_trial as prior
from vision_geometry_lab.v18_front_observability import build_fixed_baseline
from vision_geometry_lab.v18_unit_profile import semantic_from_seeds, sample_unit_normal

BASE=prior.LAB/'outputs/v18_gradient_units'
RULES={**prior.RULES,'minimum_abs_gradient':1.,
    'revision':'post_audit_single_derivative_unit_correction_not_blind_validation',
    'threshold_derivation':'unnormalized_3x3_Sobel_8_divided_by_1D_gain_8',
    'semantic_gradient_units':'gray_levels_per_current_pixel',
    'normal_gradient_units':'gray_levels_per_reference_pixel',
    'semantic_seeds_and_background_poses':'unchanged_frozen_V18_outputs',
    'threshold_scan':False}


def freeze(base=BASE):
    base=Path(base)
    if base.exists():raise FileExistsError('fresh output required')
    original=prior.verify(prior.BASE)
    dependencies={str(prior.BASE/'protocol.json'):prior.sha(prior.BASE/'protocol.json')}
    for side in ('left','right'):
        for name in ('index.json','measurement.npz'):
            path=prior.BASE/side/name;dependencies[str(path)]=prior.sha(path)
        index=prior.read(prior.BASE/side/'index.json')
        if index['output_status']!='complete_observations_only_no_events':raise ValueError('source incomplete')
        if index['measurement_sha256']!=dependencies[str(prior.BASE/side/'measurement.npz')]:raise ValueError('source changed')
    sources=[Path(__file__),prior.LAB/'v18_unit_profile.py']
    prior.write(base/'protocol.json',dict(schema='v18_unit_correction_1',
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),videos=original['videos'],rules=RULES,
        source_hashes={str(p):prior.sha(p) for p in sources},dependencies=dependencies,
        previous_trial=str(prior.BASE),
        study_limitation='After seeing V18 startup failure; unit rationale checked on <=5s only. No blind-validation claim.',
        stop_rule='Only this derivative-unit correction; no further threshold tuning to expose event'))


def verify(base=BASE):
    p=prior.read(Path(base)/'protocol.json');prior.verify(Path(p['previous_trial']))
    if p['rules']!=RULES:raise ValueError('rules changed')
    for path,digest in {**p['source_hashes'],**p['dependencies']}.items():
        if prior.sha(path)!=digest:raise ValueError('frozen dependency changed '+path)
    return p


def run(base=BASE,session='left'):
    base=Path(base);p=verify(base);dest=base/session
    if dest.exists():raise FileExistsError('do not overwrite trial')
    dest.mkdir();prior.write(dest/'started.json',{'status':'started_not_proof_of_completion'})
    original=prior.read(prior.BASE/session/'index.json')
    with np.load(prior.BASE/session/'measurement.npz') as z:old={k:z[k] for k in z.files if k!='normal_profiles'}
    video=p['videos'][session];rows=old['rows_y'];times=old['times_s'];shape=(len(times),len(rows))
    records=original['records'];transforms=old['transforms'];aligned=np.full(shape,np.nan)
    semantic={k:np.full(shape,np.nan) for k in ('x','seed','gradient','contrast')}
    cv2.setNumThreads(2);started=time.perf_counter();cap=cv2.VideoCapture(video['path'])
    if not cap.isOpened():raise ValueError('video unavailable')
    try:
        for i,record in enumerate(records):
            before=time.perf_counter();ok,raw=cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1!=record['frame_index']:raise ValueError('frame order')
            if prior.pixels_sha(raw)!=record['frame_sha256']:raise ValueError('raw hash mismatch')
            gray=cv2.cvtColor(prior.canonical_frame(raw,session),cv2.COLOR_BGR2GRAY)
            fields=semantic_from_seeds(gray,old['semantic_seed_x_current'][i],rows)
            for key in semantic:semantic[key][i]=fields[key]
            if old['pose_valid'][i]:aligned[i]=prior.align_x_to_reference(fields['x'],rows,transforms[i],rows)
            record['unit_correction_side_ms']=(time.perf_counter()-before)*1000
            if i%100==0:print(session,'unit side',i,round(time.perf_counter()-started,1),flush=True)
    finally:cap.release()
    baseline=build_fixed_baseline(times,rows,aligned,ready_s=RULES['startup_ready_s'])
    normal={k:np.full(shape,np.nan) for k in ('offset','gradient_signed','peak_ratio','contrast','clipped_fraction')}
    profiles=np.full((*shape,131),np.nan,np.float32)
    cap=cv2.VideoCapture(video['path'])
    try:
        for i,record in enumerate(records):
            before=time.perf_counter();ok,raw=cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1!=record['frame_index']:raise ValueError('normal frame order')
            if prior.pixels_sha(raw)!=record['frame_sha256']:raise ValueError('normal raw hash')
            if old['pose_valid'][i]:
                gray=cv2.cvtColor(prior.canonical_frame(raw,session),cv2.COLOR_BGR2GRAY)
                values=sample_unit_normal(gray,transforms[i],rows,baseline)
                for k in normal:normal[k][i]=values[k]
                profiles[i]=values['profile']
            record['unit_correction_normal_ms']=(time.perf_counter()-before)*1000
            if i%100==0:print(session,'unit normal',i,round(time.perf_counter()-started,1),flush=True)
    finally:cap.release()
    warm=times<=RULES['startup_ready_s'];offset=prior.supported_median(normal['offset'][warm])
    residual=normal['offset']-offset;residual[warm]=np.nan
    verify(base)
    np.savez_compressed(dest/'measurement.npz',times_s=times,rows_y=rows,
        aligned_semantic_x=aligned,semantic_x_current=semantic['x'],semantic_seed_x_current=semantic['seed'],
        semantic_gradient=semantic['gradient'],semantic_contrast=semantic['contrast'],
        pose_valid=old['pose_valid'],transforms=transforms,
        baseline_x=baseline['x_ref'],baseline_nx=baseline['nx'],baseline_ny=baseline['ny'],
        baseline_mad_px=baseline['baseline_mad_px'],
        normal_offset=normal['offset'],normal_gradient=normal['gradient_signed'],normal_peak_ratio=normal['peak_ratio'],
        normal_contrast=normal['contrast'],normal_clipped_fraction=normal['clipped_fraction'],normal_profiles=profiles,
        baseline_normal_offset=offset,normal_residual=residual)
    prior.write(dest/'index.json',dict(schema='v18_unit_corrected_spatial_measurements_1',video=video,records=records,
        rules=RULES,protocol_sha256=prior.sha(base/'protocol.json'),
        measurement_sha256=prior.sha(dest/'measurement.npz'),
        original_index_sha256=prior.sha(prior.BASE/session/'index.json'),
        baseline_support_rows=int(np.isfinite(offset).sum()),
        frame_digest_format='sha256(repr((shape,str(dtype))) + contiguous_Bgr_bytes)',
        known_occlusion_not_classified=True,material_identity_verified=False,
        output_status='complete_observations_only_no_events',runtime_s=time.perf_counter()-started))
    print('COMPLETE unit-corrected',session,'baseline rows',np.isfinite(offset).sum(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('freeze','verify','run'))
    p.add_argument('--base',type=Path,default=BASE);p.add_argument('--session',choices=('left','right'),default='left')
    a=p.parse_args()
    if a.stage=='freeze':freeze(a.base)
    elif a.stage=='verify':verify(a.base);print('V18 unit trial verified')
    else:run(a.base,a.session)


if __name__=='__main__':main()
