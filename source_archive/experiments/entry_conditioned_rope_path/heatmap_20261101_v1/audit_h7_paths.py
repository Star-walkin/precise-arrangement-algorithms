"""Independent H7 saved-path audit, reusing frozen independent H6 arithmetic.

New free observations are recomputed from a new model response and need not
equal H5/H6 observations. Historical paths are never used as target truth.
No production detector, model, annotation, or hardware module is imported.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
H6=HERE.parent/'postprocess_b_20261031_v6'
spec=importlib.util.spec_from_file_location('independent_h6_audit_for_h7',H6/'audit_h6.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
prior=old.prior;load=old.load;sha=old.sha;pts=old.pts


def check_response(entry, expected, job, model, variant, hashes):
    path=Path(entry['response_file'])
    assert path.resolve()==Path(expected['response_file']).resolve()
    if str(path) not in hashes:hashes[str(path)]=sha(path)
    assert hashes[str(path)]==expected['response_sha256']==entry['response_sha256']
    info=load(path.with_name('inference.json'))
    assert info['response_sha256']==entry['response_sha256'] and Path(info['response_file']).resolve()==path.resolve()
    assert info['model_sha256']==model['sha256'] and Path(info['model_path']).resolve()==Path(model['path']).resolve()
    assert info['variant']==variant and info['image_sha256']==job['image_sha256']
    assert info['seed_xy']==entry['seed']['xy']==expected['seed_xy']
    assert info['entry_index']==entry['entry_index']==expected['entry_index']
    assert info['available'] and not info['labels_read'] and info['bbox']==[0,1050,4096,3000]
    assert info['shape']==[975,2048] and info['scale']==.5 and info['dtype']=='float32'
    assert info['mapping'].startswith('model_xy=(original_xy-[0,1050])*.5')
    with np.load(path,allow_pickle=False) as z:
        response=z['response']
        assert response.shape==(975,2048) and response.dtype==np.float32
        assert np.isfinite(response).all() and 0<=response.min()<=response.max()<=1


def verify_free(full, record, heat_parameters):
    p=prior.aligned_points(full['points_xy']);n=len(p)
    for field in ('supported','protect_support','point_status','raw_a7_supported','observed_straight_support','B_conflict'):
        assert len(full[field])==n,('free_array_length',field)
    assert len(full['segment_status'])==max(0,n-1)
    assert all(len(v)==n for v in full.get('evidence',{}).values())
    assert not (np.asarray(full['supported'],bool)&~np.asarray(full['protect_support'],bool)).any()
    assert np.array_equal(np.asarray(full['observed_straight_support'],bool),
                          np.asarray(full['protect_support'],bool)|np.asarray(full['raw_a7_supported'],bool))
    if n:
        assert full['diagnostics']['parameters']==heat_parameters
        assert prior.line_residual(p)<1.e-5
        prior.validate_raw_a7(full,record)
        prior.validate_r4_pool(full)  # No comparison to a previous B-only axis.
    return n


def verify_selection(record):
    ranked=[(c['rank_score'],ei,ci,c) for ei,e in enumerate(record['entries'])
            for ci,c in enumerate(e['connection']['candidates'])]
    ranked.sort(key=lambda x:(record['entries'][x[1]]['task_direction']['selection_allowed'],
                              x[0],x[3]['metrics'].get('free',{}).get('quality',0)),reverse=True)
    if ranked:
        best=ranked[0]
        assert (record['selected_entry'],record['selected_candidate'])==(best[1],best[2])
        assert record['entries'][best[1]]['task_direction']['selection_allowed']
    else:
        assert record['selected_candidate'] is None
    shown=record['displayed_candidates']
    assert len(shown)==min(3,len(ranked))
    for row,expected in zip(shown,ranked):
        assert row['entry']==expected[1] and row['candidate']==expected[2]
        assert np.isclose(row['score'],expected[0],atol=1.e-9)
        assert row['task_direction_allowed']==record['entries'][expected[1]]['task_direction']['selection_allowed']


def audit(run,out):
    run=Path(run)
    if not run.is_absolute():run=HERE/'postprocess'/run
    run=run.resolve();start=load(run/'run_start.json');summary=load(run/'summary.json')
    protocol=load(HERE/'protocol.json');choice=load(HERE/'selection.json')
    manifest=load(HERE/'batch_inference/manifest.json');batch_start=load(HERE/'batch_inference/run_start.json')
    jobs={j['id']:j for j in protocol['jobs']};responses={r['id']:r for r in manifest['rows']}
    errors=[];rows=[];counts=Counter();hashes={};surface_points=0
    try:
        assert len(jobs)==len(responses)==292 and set(jobs)==set(responses)
        assert sum(len(j['seeds']) for j in jobs.values())==manifest['entry_conditions']==301
        assert sum(len(r['entries']) for r in responses.values())==301
        assert protocol['source_count']==400 and protocol['unresolved_old_entries']==108
        assert manifest['completed']==292 and manifest['exit_code']==0
        assert manifest['sources_unchanged'] and manifest['model_unchanged']
        assert summary['exit_code']==0 and not summary['failures'] and summary['source_unchanged']
        assert summary['completed']==summary['planned']==start['planned']==len(start['input_ids'])
        assert len(set(start['input_ids']))==len(start['input_ids']) and set(start['input_ids'])<=set(jobs)
        if run.name=='all_001':assert set(start['input_ids'])==set(jobs)
        assert sha(HERE/'protocol.json')==start['protocol_sha256']
        assert sha(HERE/'selection.json')==start['selection_sha256']==batch_start['selection_sha256']
        assert sha(HERE/'batch_inference/manifest.json')==start['response_manifest_sha256']
        assert start['variant']==manifest['variant']==choice['selected_variant']
        model=manifest['model'];assert model==choice['models'][choice['selected_variant']]
        assert sha(model['path'])==model['sha256']
        frozen=load(H6/'freeze_final.json');assert frozen==start['frozen_H6']
        assert start['parameters']==frozen['parameters'] and start['topology_parameters']==frozen['topology_parameters']
        assert not start['new_training_during_postprocess']
        for source,digest in (start['sources']|batch_start['sources']|frozen['sources']|frozen['reused_sources']).items():
            assert sha(source)==digest,('source_changed',source)
        assert len(summary['rows'])==len(start['input_ids'])
    except Exception as exc:
        errors.append({'scope':'run','error':str(exc),'traceback':traceback.format_exc()})
    output_rows={r['id']:r for r in summary['rows']}
    for sid in start['input_ids']:
        try:
            record=load(run/'记录'/(sid+'.json.gz'));job=jobs[sid];frozen=start['frozen_H6']
            assert record['id']==sid and record['heatmap_variant']==manifest['variant']
            assert record['source_image_sha256']==job['image_sha256'] and job['shape']==[3000,4096]
            assert record['coordinate_system']=='crop_half_xy; original=2*xy+[0,1050]'
            assert not record['response_entry_conditions_merged'] and not record['complete_identity_verified']
            assert not record['postprocess_parameters_changed'] and not record['labels_read'] and not record['devices']
            tracks=old.verify_tracks(record,frozen['surface_parameters'],hashes);counts['center_track_fragments']+=len(tracks)
            assert len(record['entries'])==len(job['seeds'])==len(responses[sid]['entries'])
            local=[]
            for ei,(entry,expected) in enumerate(zip(record['entries'],responses[sid]['entries'])):
                counts['entry_conditions']+=1
                assert entry['entry_index']==ei and entry['seed']==job['seeds'][ei]
                check_response(entry,expected,job,manifest['model'],manifest['variant'],hashes)
                counts['new_model_response_bindings_verified']+=1
                verify_free(entry['full_free_observation'],record,frozen['heat_parameters'])
                projection=old.verify_projection(entry,list(tracks.values()),frozen['topology_parameters'])
                counts['trimmed_conditions']+=projection['trimmed']
                counts['licensed_projection_events_retained']+=projection['retained_events']
                free=entry['free'];cs=entry['connection']['candidates'];assert len(cs)<=3
                assert entry['connection']['parameters']==start['parameters']
                if len(free['points_xy'])<3:assert not cs
                for c in cs:
                    counts['candidates']+=1;cp=prior.aligned_points(c['points_xy']);n=len(cp)
                    assert all(len(c[k])==n for k in ('supported','point_status','stage','width_half_px'))
                    assert not c['identity_verified']
                    prior.validate_score(c,entry['connection']['parameters'],free)
                    nf=int(np.sum(np.asarray(c['stage'])=='free'))
                    assert np.array_equal(cp[:nf],pts(free['points_xy'])[:nf])
                    assert c['point_status'][:nf]==free['point_status'][:nf] and c['supported'][:nf]==free['supported'][:nf]
                    if c['kind']=='free_fragment_no_resolved_connection':
                        assert np.array_equal(cp,pts(free['points_xy']))
                        if projection['trimmed']:assert c['termination']=='unresolved_surface_projection'
                    else:
                        assert c['kind']=='straight_free_to_F3_center_track'
                        surface_points+=old.verify_attachment(c,tracks);counts['surface_candidates']+=1
                        assert np.array_equal(entry['candidate_track_geometry'][c['track_id']],tracks[c['track_id']]['points_xy'])
                local.append({'entry':ei,'candidate_count':len(cs),**projection})
            counts['task_direction_excluded_conditions']+=prior.validate_direction(record,start['parameters'])
            verify_selection(record)
            si=record['selected_entry'];ci=record['selected_candidate']
            kind=None if ci is None else record['entries'][si]['connection']['candidates'][ci]['kind']
            counts['selected_'+str(kind)]+=1
            comparison=record['old_B_comparison'];assert comparison['entry_index']==si
            assert Path(comparison['path']).resolve()==Path(job['response_files'][si]).resolve()
            if comparison['path'] not in hashes:hashes[comparison['path']]=sha(comparison['path'])
            assert hashes[comparison['path']]==comparison['sha256']
            for path,size,digest in ((run/'四栏检测结果'/(sid+'.jpg'),(8192,1655),output_rows[sid]['sha256']),
                                     (run/'原图_旧B_新热力图'/(sid+'.jpg'),(6144,1655),output_rows[sid]['comparison_sha256'])):
                assert sha(path)==digest
                with Image.open(path) as im:assert im.size==size;im.verify()
            rows.append({'id':sid,'selected_entry':si,'selected_kind':kind,'entries':local})
        except Exception as exc:
            errors.append({'id':sid,'error':str(exc),'traceback':traceback.format_exc()})
    report={'status':'passed' if not errors else 'failed','exit_code':int(bool(errors)),
        'created_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,'run':str(run),
        'audit_sha256':sha(__file__),'reused_independent_H6_audit_sha256':sha(H6/'audit_h6.py'),
        'reused_independent_H5_audit_sha256':sha(old.H5/'audit_h5.py'),
        'planned_images':len(start['input_ids']),'checked_images':len(rows),'full_population_images':292,
        'full_population_conditions':301,'source_images':400,'unresolved_entrance_source_images':108,
        'counts':dict(counts),'surface_points_exact_fixed_pair_source_indices':surface_points,
        'errors':errors,'records':rows,
        'scope':'Saved outputs: new-model response provenance, unchanged automatic entrance, finite fixed-pair F3 midpoint source, complete bridge/tail intersections and evidence-licensed free projection.',
        'not_required':'New response hashes and new free observations are not required to equal old B/H5/H6.',
        'limits':'Artifact and topology audit only. No labels, no identity accuracy, no claim all292 visually reviewed; licensed free projection does not establish physical contact.'}
    dest=Path(out)
    if not dest.is_absolute():dest=run/dest
    with dest.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','errors')},ensure_ascii=False))
    if errors:
        print(json.dumps(errors,ensure_ascii=False));raise SystemExit(1)


def check():
    # Synthetic topology checks exercise the independent reusable verifier,
    # especially the previous target-leave-and-return/coincident-end loophole.
    track={'track_id':'T','points_xy':[[0.,0.],[5.,3.],[10.,0.],[20.,0.]]}
    ev=old.events([[0.,0.],[10.,0.]],[track])
    assert len(ev)==2 and all(e['kind']=='touch' for e in ev)
    q={'track_id':'N','points_xy':[[0.,2.],[10.,2.]]}
    ev=old.events([[0.,0.],[10.,4.]],[q]);assert len(ev)==1 and ev[0]['kind']=='cross'
    q={'track_id':'N','points_xy':[[0.,0.],[10.,0.]]}
    assert old.events([[0.,0.],[10.,0.]],[q])[0]['kind']=='overlap'
    bad={'points_xy':[[0.,0.],[10.,0.],[20.,0.]],'stage':['free','transition','surface'],
         'join_start_xy':[0.,0.],'join_end_xy':[10.,0.],'track_id':'T','surface_source_indices':[3],
         'topology':{'allowed':True,'stage_labels_used_to_skip_geometry':False,'other_track_events':[],'premature_target_events':[]}}
    caught=False
    try:old.verify_attachment(bad,{'T':track})
    except AssertionError:caught=True
    assert caught
    goodtrack={'track_id':'T','points_xy':[[0.,0.],[10.,0.],[20.,0.]]}
    good={**bad,'surface_source_indices':[2]}
    assert old.verify_attachment(good,{'T':goodtrack})==1
    f={'points_xy':[[0.,0.],[24.,0.],[48.,0.]],'supported':[True,True,True],
       'raw_a7_supported':[False,False,False],'point_status':['joint_heat_image_supported']*3}
    event={'progress_half_px':24.,'kind':'cross'}
    params={'free_crossing_window_original_px':96.,'free_crossing_continuous_original_px':48.}
    assert old.license_at(f,event,params)[0]
    f['supported'][1]=False;assert not old.license_at(f,event,params)[0]
    print(json.dumps({'passed':True,'checks':7,'scope':'synthetic geometry only; no image, model, or annotation read',
                      'audit_sha256':sha(__file__)}))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--out',default='independent_geometry_audit.json');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:check()
    elif a.run:audit(a.run,a.out)
    else:p.error('--run or --check is required')
