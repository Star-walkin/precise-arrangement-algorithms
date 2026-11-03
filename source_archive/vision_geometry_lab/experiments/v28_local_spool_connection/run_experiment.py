"""Run one fixed V28 offline image experiment; every output root must be new."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from vision_geometry_lab.rope_detector import detect_rope
from vision_geometry_lab.arc_detector import detect_arc
from vision_geometry_lab.geometry import rope_exclusion_band
from vision_geometry_lab.v16_video_measurement import automatic_seed
from vision_geometry_lab.experiments.v28_local_spool_connection import data_eval, pipeline, structure

LAB=ROOT/'vision_geometry_lab'
HERE=Path(__file__).resolve().parent


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reserve_output(path):
    path=Path(path)
    path.mkdir(parents=True,exist_ok=False)
    return path


def source_files():
    return [HERE/n for n in ('run_experiment.py','pipeline.py','structure.py','data_eval.py','render_results.py')]+[
        LAB/n for n in ('__init__.py','rope_detector.py','arc_detector.py','geometry.py',
           'v16_video_measurement.py','wound_rope_tracker.py','v10_ridge_candidates.py',
           'v12_profile_refine.py','centerline_path_metrics_v6.py','centerline_quality_v5.py')]


def freeze(output):
    output=reserve_output(output)
    inputs=data_eval.list_inputs()
    config=dict(pipeline=pipeline.DEFAULT_CONFIG,structure=structure.DEFAULT_CONFIG,
                evaluation=data_eval.EVALUATION_PROTOCOL)
    # Configuration and allowlist exist before any image inference or score.
    write_json(output/'config.json',config)
    write_json(output/'input_manifest.json',inputs)
    snapshots={}
    for path in source_files():
        relative=path.relative_to(ROOT)
        target=output/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f: f.write(path.read_bytes())
        snapshots[str(path)]=data_eval.sha256(path)
    write_json(output/'protocol.json',dict(
        schema='v28_1',frozen_at=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        experiment='local spool finite connection hypotheses, G0/G1/G2 frozen pool ablation',
        input_selection='all eight historical open development images, fixed before new scoring',
        image_count=8,independent_events=None,debug_images=[],
        config_sha256=data_eval.sha256(output/'config.json'),
        input_manifest_sha256=data_eval.sha256(output/'input_manifest.json'),
        source_hashes=snapshots,git_repository=False,
        interpreter=sys.executable,python=sys.version,opencv=cv2.__version__,numpy=np.__version__,
        old_module_reuse=['free detector','observed flange arc','automatic seed',
                         'multiscale bright/dark ridge','paired boundary profiles','trusted path metrics'],
        phases='AUTO complete and saved, then independent free_seed CONDITIONAL, then target evaluation',
        label_isolation='AUTO receives image-only allowlist and current image; scorer never reads target labels',
        no_new_network=True,no_model_download=True,no_hardware=True,no_holdout_decode=True,
        no_online_changes=True,new_layer_confirmed=False,physical_contact_confirmed=False,force_command=None))
    return output


def verify(output):
    p=read_json(output/'protocol.json')
    for path, expected in p['source_hashes'].items():
        if data_eval.sha256(path)!=expected:
            raise ValueError('Frozen source changed: '+path+'; create a new run, never overwrite')
    if data_eval.sha256(output/'config.json')!=p['config_sha256']:
        raise ValueError('Frozen configuration changed')
    if data_eval.sha256(output/'input_manifest.json')!=p['input_manifest_sha256']:
        raise ValueError('Frozen input list changed')
    for row in read_json(output/'input_manifest.json'):
        if data_eval.sha256(row['image_path'])!=row['image_sha256']:
            raise ValueError('Frozen image changed')
    return p


def detect_input(image, sample_id, mode):
    hint=data_eval.load_conditional_seed(sample_id) if mode=='CONDITIONAL' else None
    rope=detect_rope(image,seed_points=hint)
    exclusion,_=rope_exclusion_band(image.shape,rope)
    arc=detect_arc(image,exclusion_mask=exclusion)
    if mode=='AUTO':
        seed,reason=automatic_seed(rope,arc)
    else:
        # This diagnostic only replaces free-span geometry with an existing
        # independent hint. Width is remeasured on the original image.
        seed=hint if rope.get('valid') and rope.get('width_px') else None
        reason='existing_independent_free_seed_and_current_image_width' if seed else 'conditional_width_measurement_failed'
    entry=dict(available=seed is not None,free_seed=seed,width_px=rope.get('width_px'),
               reason=reason,rope=rope,arc=arc,mode=mode,
               endpoint_is_visible_support_not_contact=True,
               conditional_target_bend_used=False)
    return entry


def run_mode(output,mode):
    verify(output)
    cfg=read_json(output/'config.json')
    records=[]
    for item in read_json(output/'input_manifest.json'):
        start=time.perf_counter(); sid=item['sample_id']
        folder=output/mode/sid; folder.mkdir(parents=True,exist_ok=False)
        image=cv2.imdecode(np.fromfile(item['image_path'],np.uint8),cv2.IMREAD_COLOR)
        if image is None or list(image.shape[:2])!=item['image_shape_hw']:
            raise ValueError('Original image decode mismatch: '+sid)
        entry=detect_input(image,sid,mode)
        write_json(folder/'entry.json',entry)
        if entry['available']:
            seed=np.asarray(entry['free_seed']); direction=pipeline.unit(seed[-1]-seed[0])
            local=structure.extract_local_structure(image,seed[-1],direction,entry['width_px'],
                config=cfg['structure'],arc_points_xy=entry['arc'].get('points',[]))
            pool=pipeline.generate_pool(image,seed,entry['width_px'],local,cfg['pipeline'])
        else:
            local=dict(roi_xyxy=[0,0,image.shape[1],image.shape[0]],structures=[],
                       status='ENTRY_UNAVAILABLE',diagnostics={})
            pool=dict(schema='v28_pool_1',candidates=[],coordinate_system='full_original_image_xy',
                image_shape_hw=list(image.shape[:2]),origin_xy=[0.,0.],direction_xy=[1.,0.],normal_xy=[0.,1.],
                free_seed=None,width_px=1.,candidate_cap=cfg['pipeline']['max_candidates'],
                truncated=False,input_uses_labels=False,geometry_completion_is_evidence=False)
        pool['automatic_entry_success']=entry['available']; pool['input_mode']=mode
        write_json(folder/'structure.json',local)
        write_json(folder/'pool.json',pool)  # HARD boundary: scoring can only see this frozen pool.
        pool=read_json(folder/'pool.json')
        scores=pipeline.score_pool(image,pool,cfg['pipeline'])
        write_json(folder/'scores.json',scores)
        record=dict(sample_id=sid,mode=mode,entry=entry,
                    pool_path=str((folder/'pool.json').resolve()),
                    scores_path=str((folder/'scores.json').resolve()),
                    structure_path=str((folder/'structure.json').resolve()),
                    elapsed_s=time.perf_counter()-start)
        write_json(folder/'record.json',record)
        records.append(record)
        print(json.dumps(dict(mode=mode,sample_id=sid,entry=entry['available'],
                         structures=len(local['structures']),candidates=len(pool['candidates']),
                         selected={g:scores[g]['selected_id'] for g in scores},seconds=round(record['elapsed_s'],1)),ensure_ascii=False),flush=True)
    write_json(output/(mode+'_records.json'),records)
    verify(output)
    return records


def evaluate(output,records):
    rows=[]
    for row in records:
        pool=read_json(row['pool_path']); scores=read_json(row['scores_path'])
        ev=data_eval.evaluate_case(row['sample_id'],pool,scores,row['mode'],persisted_pool_path=row['pool_path'])
        rows.append(ev)
    result=dict(rows=rows,summary=data_eval.summarize(rows),
                physical_identity_confirmed=False,development_only=True)
    write_json(output/'evaluation.json',result)
    with (output/'results.csv').open('x',encoding='utf-8-sig',newline='') as f:
        fields=['mode','sample_id','entry_success','candidate_count','pool_contains_trusted_bend',
                'visible_connection_evaluable','group','best_id','selected_id','status','visible_connection',
                'ribbon_geometry','bend_coverage_5px','margin','stability','reason']
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for record,ev in zip(records,rows):
            pool=read_json(record['pool_path']);scores=read_json(record['scores_path'])
            for g in ('G0','G1','G2'):
                s=scores[g]; c=ev['choices'][g]
                writer.writerow(dict(mode=ev['mode'],sample_id=ev['sample_id'],entry_success=ev['automatic_entry_success'],
                    candidate_count=len(pool['candidates']),pool_contains_trusted_bend=ev['candidate_pool_contains_trusted_bend_geometry'],
                    visible_connection_evaluable=ev['visible_connection_evaluable'],group=g,best_id=s['best_id'],
                    selected_id=s['selected_id'],status=s['status'],visible_connection=c['selected_visible_connection_status'],
                    ribbon_geometry=c['selected_visible_geometry_status'],bend_coverage_5px=c['bend_geometry_5px']['covered_fraction'],
                    margin=s['margin'],stability=s['stability'],reason=';'.join(s['reason'])))
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--skip-conditional',action='store_true')
    ap.add_argument('--verify',action='store_true')
    args=ap.parse_args(); output=args.output.resolve()
    cv2.setNumThreads(2)
    if args.verify:
        verify(output); print('verified'); return
    freeze(output)
    command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]
    write_json(output/'command_start.json',dict(argv=command,cwd=str(Path.cwd()),
        start_local=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()))
    try:
        records=run_mode(output,'AUTO')
        write_json(output/'auto_complete_before_conditional_and_evaluation.json',dict(
            auto_records_sha256=data_eval.sha256(output/'AUTO_records.json'),
            pool_hashes={r['sample_id']:data_eval.sha256(r['pool_path']) for r in records}))
        if not args.skip_conditional:
            records+=run_mode(output,'CONDITIONAL')
        write_json(output/'records.json',records)
        evaluate(output,records)
        from vision_geometry_lab.experiments.v28_local_spool_connection.render_results import render_run
        render_run(output)
        verify(output)
        write_json(output/'command_result.json',dict(exit_code=0,status='complete',argv=command))
    except Exception as exc:
        write_json(output/'command_result.json',dict(exit_code=1,status='failed_preserved',argv=command,
                    error_type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__':
    main()
