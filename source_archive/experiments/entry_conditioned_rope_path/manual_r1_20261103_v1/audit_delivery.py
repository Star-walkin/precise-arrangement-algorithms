"""Independent all-output audit. No training, inference or evaluation labels.

Does not import render_results.py. Heat pixels are reconstructed with an
independent four-neighbor interpolation equation in row chunks.
"""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,sys,time,traceback
from collections import Counter
import cv2,numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;BASE=HERE.parent
CONTEXT=None
Image.MAX_IMAGE_PIXELS=150_000_000

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def initialize():
    global CONTEXT
    cv2.setNumThreads(1)
    p=load(HERE/'protocol.json');s=load(HERE/'selected_checkpoints.json')
    inf=load(HERE/'inference/gallery/receipt.json');paths=load(HERE/'paths/gallery_receipt.json')
    render=load(HERE/'render_receipt.json');index=load(HERE/'dataset/index.json')
    known_pipeline_sources={sha(HERE/'r1_pipeline.py')}
    prior=HERE/'repairs/gallery_role_001/r1_pipeline_before.py'
    if prior.exists():known_pipeline_sources.add(sha(prior))
    CONTEXT=dict(protocol=p,selection=s,inference={(x['id'],x['entry_index']):x for x in inf['rows']},
        paths={(x['id'],x['entry_index']):x for x in paths['rows']},render={x['id']:x for x in render['rows']},
        relation=load(HERE/'relation/model/training_receipt.json'),freeze=load(HERE/'r1_freeze.json'),
        training_sha={x['image_sha256']:x for x in index['items']},
        training_groups={x['source_group'] for x in index['items']},eval_groups=set(p['eval_groups']),
        lut=np.load(BASE/'batch_ab_20261030_001/color_lut.npy'),pipeline_sources=known_pipeline_sources)

def middle_raster(response,lut):
    """Align-false x2, replicate border; float64 exact binary quarter weights.

    The four original float32 neighbors and .25/.75 weights are evaluated
    directly. This deliberately differs from the renderer's separable arrays.
    """
    h,w=response.shape;dest=np.full((3000,4096,3),45,np.uint8)
    sx=(np.arange(2*w,dtype=np.float64)+.5)/2-.5
    floorx=np.floor(sx).astype(int);fx=(sx-floorx)[None,:]
    xlo=np.clip(floorx,0,w-1);xhi=np.clip(floorx+1,0,w-1)
    for begin in range(0,h*2,128):
        stop=min(begin+128,h*2)
        sy=(np.arange(begin,stop,dtype=np.float64)+.5)/2-.5
        floory=np.floor(sy).astype(int);fy=(sy-floory)[:,None]
        ylo=np.clip(floory,0,h-1)[:,None];yhi=np.clip(floory+1,0,h-1)[:,None]
        q00=response[ylo,xlo[None,:]].astype(np.float64)
        q01=response[ylo,xhi[None,:]].astype(np.float64)
        q10=response[yhi,xlo[None,:]].astype(np.float64)
        q11=response[yhi,xhi[None,:]].astype(np.float64)
        values=(q00*(1-fx)*(1-fy)+q01*fx*(1-fy)+q10*(1-fx)*fy+q11*fx*fy).astype(np.float32)
        indices=np.rint(values*np.float32(255)).astype(np.uint8)
        dest[1050+begin:1050+stop]=lut[indices]
    return dest

def inspect_candidate(c):
    q=np.asarray(c['points_xy'],float);states=np.asarray(c['point_status'])
    assert q.ndim==2 and q.shape[1]==2 and len(q)>=2 and np.isfinite(q).all()
    assert len(states)==len(q) and set(states)<=set(['entry_condition','visible_supported','inferred'])
    assert ((q[:,0]>=0)&(q[:,0]<4096)&(q[:,1]>=1050)&(q[:,1]<3000)).all()
    lengths=np.linalg.norm(np.diff(q,axis=0),axis=1)
    assert np.isclose(lengths.sum(),c['length_px'],rtol=1e-10,atol=1e-7)
    same_lengths={}
    for state in ['entry_condition','visible_supported','inferred']:
        same_lengths[state]=float(lengths[(states[:-1]==state)&(states[1:]==state)].sum())
        assert np.isclose(same_lengths[state],c['status_length_px'][state],rtol=1e-10,atol=1e-7)
        flag={'entry_condition':'entry_condition','visible_supported':'supported','inferred':'inferred'}[state]
        if flag in c:assert c[flag]==(states==state).tolist()
    assert c['complete_identity_verified'] is False
    mixed=float(lengths[states[:-1]!=states[1:]].sum())
    assert np.isclose(sum(same_lengths.values())+mixed,lengths.sum(),atol=1e-6)
    return dict(points=len(q),length_px=float(lengths.sum()),same_state_edge_lengths=same_lengths,
                mixed_status_transition_length_px=mixed,inferred_points=int((states=='inferred').sum()))

def inspect_path(record,row,meta):
    assert sha(row['path'])==row['sha256']
    assert record['id']==meta['id'] and record['entry_index']==meta['entry_index'] and record['arm']=='R1_new'
    assert record['status']==row['status']
    bind=record['metadata'];ctx=CONTEXT;model=ctx['selection']['paths']['U1']
    assert bind['response_sha256']==meta['response_sha256'] and bind['image_sha256']==meta['image_sha256']
    assert bind['U1_model_sha256']==model['sha256'] and bind['relation_model_sha256']==ctx['relation']['portable_sha256']
    assert bind['entry_seed_xy']==meta['seed_xy']
    assert bind['target_labels_read'] is False and bind['evaluation_references_read'] is False
    assert bind['old_A_features_used'] is False and bind['selected_entry_not_truth_or_heat_ranked'] is True
    assert bind['pipeline_source_sha256'] in ctx['pipeline_sources']
    assert bind['frozen_R1_sha256']==sha(HERE/'r1_freeze.json')
    assert record['parameters']==ctx['freeze']['parameters']
    result=record['result'];candidates=result['candidates']
    assert len(candidates)<=3 and len({x['id'] for x in candidates})==len(candidates)
    assert result['relation_enabled'] is True and 'before pool' in result['relation_insertion']
    stats=result['search'];assert stats['generated']==stats['relation_scored_before_pruning']
    details={c['id']:inspect_candidate(c) for c in candidates}
    byid={c['id']:c for c in candidates};selected=result['selected_candidate_id']
    assert selected==row['selected_candidate_id']
    if selected is None:assert record['selected'] is None
    else:
        assert selected in byid and record['selected']['id']==selected
        for k in ['points_xy','point_status','length_px','status_length_px','rank_score']:
            assert record['selected'][k]==byid[selected][k]
        inspect_candidate(record['selected'])
    assert {c['id'] for c in record['alternatives']}==set(byid)-({selected} if selected else set())
    for c in record['alternatives']:
        assert c['points_xy']==byid[c['id']]['points_xy'];inspect_candidate(c)
    assert result['entry_counted_as_recognition'] is False and result['full_target_identity_accuracy'] is None
    return dict(candidate_count=len(candidates),selected=selected,candidates=details,
                pipeline_source_sha256=bind['pipeline_source_sha256'])

def one(job):
    ctx=CONTEXT;row=ctx['render'][job['id']];model=ctx['selection']['paths']['U1'];condition_details=[]
    assert sha(job['image_path'])==job['image_sha256']==row['source_sha256']
    assert row['selected_entry']==job.get('main_display_entry_index',0)==0
    assert row['entry_count']==len(job['seeds']) and row['model']==model
    assert row['model_step']==ctx['selection']['common_step']==800
    assert row['R1_model']['sha256']==ctx['relation']['portable_sha256']
    assert row['size']==[12492,3370] and row['no_manual_labels_rendered'] is True and row['full_identity_verified'] is False
    response_main=None;path_main=None
    for entry,seed in enumerate(job['seeds']):
        key=(job['id'],entry);meta=ctx['inference'][key];pathrow=ctx['paths'][key]
        assert meta['id']==job['id'] and meta['entry_index']==entry
        assert meta['image_sha256']==job['image_sha256'] and meta['seed_xy']==seed['xy']
        assert meta['model_sha256']==model['sha256'] and meta['model_step']==800 and meta['branch']=='U1'
        assert meta['new_model_forward'] is True and meta['target_labels_read'] is False and meta['status']=='predicted'
        assert 'new trained U1 dec1' in meta['feature_source'] and 'SAME new forward' in meta['feature_source']
        assert sha(meta['response_path'])==meta['response_sha256']
        sibling=load(Path(meta['response_path']).with_suffix('.json'));assert sibling==meta
        with np.load(meta['response_path'],allow_pickle=False) as z:
            assert set(z.files)=={'response','features'}
            response=z['response'];features=z['features']
        assert response.dtype==np.float32 and response.shape==(975,2048) and np.isfinite(response).all()
        assert 0<=response.min()<=response.max()<=1
        assert features.dtype==np.float16 and features.shape==(8,248,512) and np.isfinite(features).all()
        path=load(pathrow['path']);checked=inspect_path(path,pathrow,meta)
        condition_details.append(dict(entry_index=entry,response_sha256=meta['response_sha256'],
            response_min=float(response.min()),response_max=float(response.max()),response_std=float(response.std()),
            features_finite_shape_dtype_verified=True,path_sha256=pathrow['sha256'],path=checked))
        condition=row['all_conditions'][entry]
        assert condition['entry_index']==entry and condition['response_sha256']==meta['response_sha256']
        assert condition['path_sha256']==pathrow['sha256'] and condition['status']==meta['status']
        if entry==0:response_main=response;path_main=path
    assert row['response_sha256']==ctx['inference'][(job['id'],0)]['response_sha256']
    assert row['path_record_sha256']==ctx['paths'][(job['id'],0)]['sha256']
    assert row['selected_candidate_id']==(path_main['selected']['id'] if path_main['selected'] else None)
    assert row['alternative_count']==len(path_main['alternatives'])
    match=ctx['training_sha'].get(job['image_sha256'])
    expected_use=('training_manual' if match['origin']=='manual' else 'training_old') if match else (
        'historical_open_evaluation_source_display' if job['source_group'] in ctx['eval_groups'] else
        'same_training_source_unused_frame_display' if job['source_group'] in ctx['training_groups'] else
        'other_open_untrained_source_display')
    assert row['source_use']==expected_use and row['source_is_training_image']==bool(match)
    assert sha(row['output'])==row['sha256'] and Path(row['output']).name==job['id']+'.png'
    with Image.open(job['image_path']) as f:raw=f.convert('RGB');raw.load()
    with Image.open(row['output']) as board:
        assert board.format=='PNG' and board.mode=='RGB' and board.size==(12492,3370)
        boxes=[[0,160,4096,3160],[4128,160,8224,3160],[8256,160,12352,3160]]
        assert row['image_viewports_xyxy']==boxes
        assert board.crop(tuple(boxes[0])).tobytes()==raw.tobytes(),'Original full viewport changed'
        actual=np.asarray(board.crop(tuple(boxes[1])))
        expected=middle_raster(response_main,ctx['lut'])
        assert np.array_equal(actual,expected),'Raw response raster differs (stretch/overlay/mapping/color)'
        assert (actual[:1050]==45).all()
        assert board.crop(tuple(boxes[2])).size==raw.size
    return dict(id=job['id'],source_use=expected_use,png_sha256=row['sha256'],
        full_left_original_pixel_identical=True,full_middle_independent_float_reconstruction_pixel_identical=True,
        outside_calculation_gray45_exact=True,three_native4096x3000_panels=True,
        right_path_JSON_model_entry_hash_consistent=True,entry_conditions=condition_details,
        complete_target_identity_verified=False)

def main():
    sys.stdout.reconfigure(encoding='utf-8');parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--output',type=Path,default=HERE/'audit_delivery_receipt.json')
    args=parser.parse_args();started=time.perf_counter();initialize();ctx=CONTEXT
    render=load(HERE/'render_receipt.json');inf=load(HERE/'inference/gallery/receipt.json');paths=load(HERE/'paths/gallery_receipt.json')
    assert inf['exit_code']==paths['exit_code']==render['exit_code']==0
    jobs=ctx['protocol']['gallery_jobs'];assert len(jobs)==len({j['id'] for j in jobs})==292
    expected={(j['id'],e) for j in jobs for e in range(len(j['seeds']))};assert len(expected)==301
    assert expected==set(ctx['inference'])==set(ctx['paths'])
    assert set(ctx['render'])=={j['id'] for j in jobs} and len(render['rows'])==292
    assert len(list(Path(render['output_directory']).glob('*.png')))==292
    assert render['U1_inference_receipt_sha256']==sha(HERE/'inference/gallery/receipt.json')
    assert render['R1_path_receipt_sha256']==sha(HERE/'paths/gallery_receipt.json')
    assert ctx['selection']['common_step']==800
    model=ctx['selection']['paths']['U1'];assert sha(model['path'])==model['sha256']
    r1=ctx['relation'];assert r1['exit_code']==0 and r1['steps']==600
    assert sha(HERE/'relation/model/final_portable.npz')==r1['portable_sha256']
    assert r1['U1_sha256']==model['sha256'] and r1['U1_frozen_for_relation_training'] is True
    official=cv2.applyColorMap(np.arange(256,dtype=np.uint8)[:,None],cv2.COLORMAP_TURBO)[:,0,::-1]
    assert np.array_equal(ctx['lut'],official),'Frozen LUT must be actual fixed TURBO RGB'
    for path,digest in ctx['freeze']['source_hashes'].items():assert sha(path)==digest
    rows=[];errors=[]
    with concurrent.futures.ProcessPoolExecutor(args.workers,initializer=initialize) as pool:
        pending={pool.submit(one,j):j['id'] for j in jobs}
        for future in concurrent.futures.as_completed(pending):
            try:rows.append(future.result())
            except Exception:errors.append(dict(id=pending[future],traceback=traceback.format_exc()))
            if (len(rows)+len(errors))%20==0 or len(rows)+len(errors)==292:
                print(dict(checked=len(rows),errors=len(errors),planned=292,seconds=round(time.perf_counter()-started,1)),flush=True)
    receipt=dict(exit_code=int(bool(errors)),command=sys.argv,seconds=time.perf_counter()-started,
        source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),
        render_receipt_sha256=sha(HERE/'render_receipt.json'),
        checked_unique_sources=len(rows),checked_entry_conditions=sum(len(r['entry_conditions']) for r in rows),
        exact292_unique_original_sources=len(rows)==292 and len({r['id'] for r in rows})==292,
        source_use_counts=dict(Counter(r['source_use'] for r in rows)),
        original_left_and_raw_heat_middle_all_pixels_checked=True,
        raw_middle_formula='independent direct4-neighbor alignFalse x2, replicate borders, castfloat32, rint255, fixed officialTURBO RGB',
        right_path_source_model_entry_and_candidates_checked=True,
        legacy_status_length_rule='same-state adjacent edges only; mixed transitions separately recorded, isolated inferred points may have zero inferred-edge length',
        no_renderer_import=True,no_training_or_inference_reexecuted=True,no_user_labels_or_evaluation_references_read=True,
        complete_target_identity_accuracy=None,errors=errors,rows=rows)
    args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='rows'},ensure_ascii=False,indent=2),flush=True)
    return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
