"""Frozen V12 finite CV refinement trial; no retraining or physical control.

Train-first non-degradation gate, with an explicitly GT-seeded error-repair
stress test. Ground truth never enters the actual V11 prediction refinement.
"""
from pathlib import Path
import argparse
import datetime
import json
import time

import numpy as np

from .v11_path_experiment import BASE as V11, MANIFEST, read, sha, write_json, verify as verify_v11
from .v12_profile_refine import METHODS, PARAMETERS, refine_paths
from .evaluate_centerline_multiscale_v9 import _dataset, score_decoded, aggregate
from .centerline_review_v5 import read_clean_image

LAB=Path(__file__).resolve().parent
BASE=LAB/'outputs/v12_profile_refinement'


def verify_baseline():
    evaluation=read(V11/'development/evaluation.json')
    exports=read(V11/'development/exports.json')
    report=read(V11/'development/report.json')
    for artifact in (evaluation,exports,report):
        if artifact['arm']!='development' or artifact['protocol_sha256']!=sha(V11/'protocol.json'):
            raise ValueError('Wrong V11 baseline arm/protocol')
        if artifact['checkpoint_sha256']!=sha(V11/'development/checkpoint.pt'):
            raise ValueError('V11 model mismatch')
    if evaluation['exports_sha256']!=sha(V11/'development/exports.json'):
        raise ValueError('V11 evaluation/export mismatch')
    if exports['training_report_sha256']!=sha(V11/'development/report.json'):
        raise ValueError('V11 training provenance mismatch')
    for artifact in (exports,report):
        if artifact['cache_index_sha256']!=sha(V11/'development_cache/index.json'):
            raise ValueError('V11 cache provenance mismatch')
    expected={r['sample_id']:(r['split'],r['group_id']) for r in read(MANIFEST)['samples']}
    actual={r['sample_id']:(r['split'],r['group_id']) for r in evaluation['rows']}
    if len(evaluation['rows'])!=len(expected) or actual!=expected:
        raise ValueError('V11 baseline dataset rows mismatch')


def freeze(output):
    if output.exists(): raise FileExistsError('Never replace a previous experiment')
    verify_v11(V11)
    verify_baseline()
    source=read(MANIFEST)
    if '20260925' in json.dumps(source): raise ValueError('Reserved blind source prohibited')
    artifacts=[V11/'protocol.json', V11/'development/evaluation.json',V11/'development/exports.json',
               V11/'development/checkpoint.pt', V11/'development/report.json',V11/'development_cache/index.json',MANIFEST]
    protocol=dict(version=12,frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        question='Can image-only bounded normal-profile refinement correct optical center bias without damaging correct paths?',
        methods=list(METHODS),parameters=PARAMETERS,input='original RGB + saved V11 segments + independent free-span optical width',
        changes_model=False,refiner_uses_labels=False,fixed_thresholds=True,
        main_arms=['original_v11','resampled_control',*METHODS],
        gate=dict(train_coverage_max_drop=.01,train_precision_max_drop=.01,need_at_least_one_accepted_point=True),
        gate_scope='non-degradation prerequisite only, not deployment acceptance',
        heldout_policy='Only arms passing train gate; no threshold change after observing scores',
        stress_offsets_widths=[-.45,.45],
        stress_label='GT-derived shifted reference on TRAIN ONLY; not model performance or realistic guaranteed error distribution',
        stress_used_for_selection=False,
        score='unchanged V6 5 native-pixel arc-length full/downstream, all trusted denominator retained',
        fallback='Retain original candidate and mark NOT CV confirmed; separately report accepted points/segments',
        caveat='Optical ridge/edge midpoint is not necessarily rope body center; width prior is not metric diameter',
        representative_train_ids=['dual_20260924_111500__left__f00246','C07__R_run_stop_run_20260922_01__t012.033__f000368'],
        source_snapshot={str(LAB/name):sha(LAB/name) for name in ('v12_refine_experiment.py','v12_profile_refine.py')},
        artifact_snapshot={str(path):sha(path) for path in artifacts},hardware_access=False,force_command=None)
    write_json(output/'protocol.json',protocol)
    for split in ('train','holdout'):
        minimal=dict(source)
        minimal['samples']=[s for s in source['samples'] if s['split']==split]
        write_json(output/(split+'_manifest.json'),minimal)
    print(json.dumps(dict(protocol=str(output/'protocol.json'),sha256=sha(output/'protocol.json'))))


def verify(output):
    verify_v11(V11)
    verify_baseline()
    protocol=read(output/'protocol.json')
    for kind in ('source_snapshot','artifact_snapshot'):
        for path,digest in protocol[kind].items():
            if sha(path)!=digest: raise ValueError('Frozen input/source changed: '+path)
    if json.loads(json.dumps(PARAMETERS))!=protocol['parameters'] or list(METHODS)!=protocol['methods']:
        raise ValueError('Parameters changed after freeze')
    source=read(MANIFEST)
    for split in ('train','holdout'):
        expected=dict(source);expected['samples']=[s for s in source['samples'] if s['split']==split]
        if read(output/(split+'_manifest.json'))!=expected: raise ValueError('Split scope changed')
    return protocol


def _result_row(record, base, method, segments, result=None, milliseconds=0.):
    sample=record['sample']
    decoded=dict(segments=segments,reason='saved_v11_baseline' if result is None else 'bounded_photometric_candidate_with_explicit_fallback',
                 identity_confirmed=False,force_command=None)
    score=score_decoded(record['label'],record['quality'],decoded,None,sample['image_shape_hw'])
    # Spatial support is unchanged by this experiment; never interpret None
    # oracle/support masks as zero actual field-of-view support.
    score['coverage_support']=base['coverage_support']
    return dict(sample_id=sample['sample_id'],split=sample['split'],group_id=sample['group_id'],cohort=record['cohort'],
        method=method,prediction_status='loaded',independent_width=base['independent_width'],
        decoded=decoded,**score,refinement=result,elapsed_ms=milliseconds,
        support_note='inherited V11 source observation support; score denominator never cropped',
        target_input=False,quality_input=False,physical_center_confirmed=False)


def _resampled_inputs(result):
    segments=[]
    for segment_index in sorted({p['segment_index'] for p in result['diagnostics']['records']}):
        points=[p['input_xy'] for p in result['diagnostics']['records'] if p['segment_index']==segment_index]
        if len(points)>=2: segments.append(points)
    return segments


def _shifted_reference(segments,width,amount):
    """Explicit GT-input stressor only. Not a detector and never heldout input."""
    shifted=[]
    for segment in segments:
        points=np.asarray(segment,float)
        if len(points)<2: continue
        tangent=np.gradient(points,axis=0)
        tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
        normal=np.stack([-tangent[:,1],tangent[:,0]],axis=1)
        shifted.append((points+normal*(width*amount)).tolist())
    return shifted


def _counts(rows):
    records=[p for row in rows if row.get('refinement') is not None for p in row['refinement']['diagnostics']['records']]
    accepted=[p for p in records if p['accepted']]
    reasons={}
    for point in records: reasons[point['reason']]=reasons.get(point['reason'],0)+1
    return dict(total_sampled_points=len(records),accepted_points=len(accepted),fallback_points=len(records)-len(accepted),
        accepted_fraction=len(accepted)/len(records) if records else 0.,reason_counts=reasons,
        median_accepted_absolute_shift_px=float(np.median([abs(p['applied_offset_px']) for p in accepted])) if accepted else None,
        maximum_absolute_shift_px=max([abs(p['applied_offset_px']) for p in accepted],default=0.),
        fallback_is_cv_confirmation=False)


def run(output,split):
    protocol=verify(output)
    methods=list(METHODS)
    if split=='holdout':
        gate=read(output/'gate.json')
        if gate['protocol_sha256']!=sha(output/'protocol.json') or gate['train_results_sha256']!=sha(output/'train/results.json'):
            raise ValueError('Gate source mismatch')
        expected_rows=gate_decisions(read(output/'train/results.json'),protocol)
        if gate['rows']!=expected_rows or gate['passing_methods']!=[r['method'] for r in expected_rows if r['passed']]:
            raise ValueError('Gate decisions do not match the fixed non-degradation rule')
        methods=gate['passing_methods']
        if not methods:
            print('No method passed the training non-degradation gate; holdout images not evaluated.');return
    folder=output/split
    if folder.exists(): raise FileExistsError('Do not overwrite or silently resume a run')
    import cv2
    cv2.setNumThreads(2)
    _,records,_=_dataset(output/(split+'_manifest.json'))
    if len(records)!=(10 if split=='train' else 8): raise ValueError('Wrong split scope')
    saved={r['sample_id']:r for r in read(V11/'development/evaluation.json')['rows']}
    actual_rows, stress_rows=[],[]
    for record in records:
        sample=record['sample'];name=sample['sample_id'];base=saved[name]
        image=np.asarray(read_clean_image(sample))
        width=base['independent_width']['width_px']
        if width is None: raise ValueError('Known baseline width missing; do not drop this sample')
        original=base['decoded']['segments']
        row=_result_row(record,base,'original_v11',original)
        actual_rows.append(row)
        write_json(folder/'rows'/f'{name}__original_v11.json',row)
        for method in methods:
            started=time.perf_counter()
            result=refine_paths(image,original,width,method)
            elapsed=(time.perf_counter()-started)*1000
            if method==methods[0]:
                control=_result_row(record,base,'resampled_control',_resampled_inputs(result))
                actual_rows.append(control)
                write_json(folder/'rows'/f'{name}__resampled_control.json',control)
            row=_result_row(record,base,method,result['segments'],result,elapsed)
            actual_rows.append(row)
            write_json(folder/'rows'/f'{name}__{method}.json',row)
        if split=='train':
            reference=base['metrics']['score_support']['full_reference_segments']
            for offset in protocol['stress_offsets_widths']:
                shifted=_shifted_reference(reference,width,offset)
                initial=_result_row(record,base,'shifted_reference',shifted)
                stress_rows.append(dict(sample_id=name,offset_widths=offset,method='unrefined',
                    metrics=initial['metrics'],gt_used_for_input=True,model_performance=False))
                for method in methods:
                    result=refine_paths(image,shifted,width,method)
                    checked=_result_row(record,base,method,result['segments'],result)
                    stress_rows.append(dict(sample_id=name,offset_widths=offset,method=method,metrics=checked['metrics'],
                        counts=_counts([checked]),gt_used_for_input=True,model_performance=False))
        print(json.dumps(dict(sample=name,split=split,downstream={r['method']:r['metrics']['downstream']['covered_fraction']
            for r in actual_rows if r['sample_id']==name}),ensure_ascii=False),flush=True)
    arms=['original_v11','resampled_control',*methods]
    summaries={method:aggregate([r for r in actual_rows if r['method']==method]) for method in arms}
    compact=[]
    for row in actual_rows:
        compact.append({k:v for k,v in row.items() if k!='refinement'})
    result=dict(split=split,protocol_sha256=sha(output/'protocol.json'),rows=compact,summaries=summaries,
        per_point_records='separate rows/*__method.json; accepted/fallback are never conflated',
        counts={method:_counts([r for r in actual_rows if r['method']==method]) for method in methods},
        median_refinement_ms={method:float(np.median([r['elapsed_ms'] for r in actual_rows if r['method']==method])) for method in methods},
        stress_rows=stress_rows,stress_warning='GT-derived training probes; not actual prediction performance, not used for gates',
        all_rows_retained=True,hardware_access=False,force_command=None)
    write_json(folder/'results.json',result)
    verify(output)


def gate_decisions(source,protocol):
    def summary(method):
        return next(r for r in source['summaries'][method] if r['scope']=='downstream' and r['cohort']=='all' and r['split']=='train')
    baseline=summary('original_v11');entries=[]
    for method in METHODS:
        score=summary(method)
        dc=score['covered_fraction']-baseline['covered_fraction']
        dp=score['prediction_precision_at_5px']-baseline['prediction_precision_at_5px']
        accepted=source['counts'][method]['accepted_points']
        passed=dc>=-protocol['gate']['train_coverage_max_drop'] and dp>=-protocol['gate']['train_precision_max_drop'] and accepted>0
        entries.append(dict(method=method,coverage_change=dc,precision_change=dp,accepted_points=accepted,
            passed=passed,criterion='train coverage/precision drops <=1 percentage point and nonzero accepted evidence'))
    return entries


def gate(output):
    protocol=verify(output)
    source=read(output/'train/results.json')
    if source['protocol_sha256']!=sha(output/'protocol.json'): raise ValueError('Wrong train experiment')
    entries=gate_decisions(source,protocol)
    result=dict(protocol_sha256=sha(output/'protocol.json'),train_results_sha256=sha(output/'train/results.json'),
        rows=entries,passing_methods=[r['method'] for r in entries if r['passed']],
        deployment_allowed=False,stress_scores_used=False)
    write_json(output/'gate.json',result)
    print(json.dumps(result,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','train','gate','holdout','verify'))
    parser.add_argument('--output',type=Path,default=BASE)
    args=parser.parse_args()
    if args.stage=='freeze': freeze(args.output)
    elif args.stage in ('train','holdout'): run(args.output,args.stage)
    elif args.stage=='gate': gate(args.output)
    else: verify(args.output);print('V12 sources and baseline artifacts verified')


if __name__=='__main__': main()
