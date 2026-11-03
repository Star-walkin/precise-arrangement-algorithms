"""Freeze actual user labels and build partial positive/red-negative supervision."""
from pathlib import Path
from collections import Counter
import json, hashlib, sys, time
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=HERE.parents[2]
TASK=ROOT/'vision_training/task_active_rope_center_manual_20261103'
sys.path.insert(0,str(TASK));import annotation_core as manual
sys.path.insert(0,str(BASE));from prepare import response_from_lines

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    cv2.setNumThreads(2);started=time.perf_counter()
    prior=load(BASE/'data_expansion_d1_20261102_v1/protocol.json')
    manifest=manual.load_manifest(TASK/'manifest.json')
    oldix=load(BASE/'runs/run_20261030_002/data/index.json')
    old=[dict(x,origin='old') for x in oldix['items'] if x['split']=='train'];assert len(old)==10
    samples=manifest['samples'];assert len(samples)==24
    assert not {s['source_group'] for s in samples}&set(prior['eval_groups'])
    dest=HERE/'dataset';cache=dest/'cache';cache.mkdir(exist_ok=False)
    items=old[:];rows=[];lengths=Counter();status=Counter();annotation_hashes={}
    shape=(975,2048);bbox=[0,1050,4096,3000];scale=.5
    yy,xx=np.indices(shape)
    for s in samples:
        source=TASK/'annotations'/(s['id']+'.json')
        snapshot=HERE/'label_audit/annotation_snapshot'/(s['id']+'.json')
        assert source.read_bytes()==snapshot.read_bytes(),'User changed label after audit; freeze a new audit'
        a=manual.validate(load(snapshot),s);assert a['reviewed']
        assert sha(s['image_path'])==s['image_sha256']
        annotation_hashes[s['id']]=sha(snapshot);status[a['status']]+=1
        pos=np.zeros(shape,bool);neg=np.zeros(shape,bool);unknown=np.zeros(shape,bool)
        target=np.zeros(shape,np.float32);stage=np.zeros(shape,np.uint8);segment_rows=[]
        for seg in a['segments']:
            line=np.asarray(seg['points_xy'],float)
            assert np.all(line[:,1]>=1050),'Annotation extends above frozen search field'
            hm,d=response_from_lines([line],shape,bbox,scale,3.)
            radius=float(seg['valid_radius_px']);tube=d<=radius*scale
            if seg['kind']!='uncertain':
                for q in (line[[0,-1]]-[0,1050])*scale:
                    tube &= ((xx-q[0])**2+(yy-q[1])**2)>(radius*scale)**2
            ll=float(np.linalg.norm(np.diff(line,axis=0),axis=1).sum());lengths[seg['kind']]+=ll
            segment_rows.append(dict(kind=seg['kind'],length_original_px=ll,stage=seg['stage'],radius_original_px=radius))
            if seg['kind']=='target_center':
                pos|=tube;target=np.maximum(target,hm*tube);stage[tube]=seg['stage']
            elif seg['kind']=='neighbor_negative':neg|=tube;stage[tube]=seg['stage']
            else:unknown|=tube
        # Narrow tube edges can overlap even though positive and red centers do not.
        # Preserve user lines, remove contradictory pixels from BOTH classes.
        conflict=pos&neg;valid=(pos|neg)&~unknown&~conflict
        negative=neg&valid;target[~valid]=0;stage[~valid]=0
        assert np.any(valid&(target>=.5)) and negative.any()
        assert not np.any(negative&(target>0))
        active_stages={int(x) for x in np.unique(stage[valid&(target>=.5)])}
        assert all(int(x) in active_stages for x in np.unique(stage[negative])), 'Red supervision would be skipped by loss'
        bgr=cv2.imdecode(np.fromfile(s['image_path'],np.uint8),cv2.IMREAD_COLOR)
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
        image=cv2.resize(gray[1050:],(2048,975),interpolation=cv2.INTER_AREA)
        seed,_=response_from_lines([s['seed_xy']],shape,bbox,scale,4.)
        out=cache/(s['id']+'.npz')
        np.savez_compressed(out,image=image,seed=seed,target=target,valid=valid,stage=stage,
                            roi=np.ones(shape,bool),negative=negative,conflict_ignore=conflict)
        items.append(dict(id=s['id'],split='train',origin='manual',source_group=s['source_group'],
                          image_path=s['image_path'],image_sha256=s['image_sha256'],
                          cache=str(out),cache_sha256=sha(out),known_seed_xy=s['seed_xy'],bbox=bbox,
                          scale=.5,shape=[3000,4096],annotation_path=str(snapshot),annotation_sha256=sha(snapshot),
                          annotation_status='human_reviewed_partial_centers_and_explicit_neighbor_negatives',
                          source_annotation_path=str(source),surface_roi_path=s['roi_path'],surface_roi_sha256=s['roi_sha256']))
        rows.append(dict(id=s['id'],source_group=s['source_group'],segments=segment_rows,
                         valid_pixels=int(valid.sum()),red_negative_pixels=int(negative.sum()),
                         positive_high_pixels=int((valid&(target>=.5)).sum()),
                         conflict_ignore_model_grid_pixels=int(conflict.sum()),
                         conflict_high_pixels=int((conflict&(target>=.5)).sum()),
                         red_stages_active_in_loss=True,manual_stage_fields_preserved=True))
    stats=dict(manual_images=24,old_images=10,training_images=len(items),
               train_groups=sorted({x['source_group'] for x in items}),eval_groups=prior['eval_groups'],
               segment_counts=dict(Counter(seg['kind'] for s in samples for seg in load(HERE/'label_audit/annotation_snapshot'/(s['id']+'.json'))['segments'])),
               center_length_original_px=dict(lengths),status_counts=dict(status),
               manual_stage_fields_all_default_one=True,physical_phase_not_measured=True,
               conflict_policy='positive/red valid-tube overlap is ignored in both; no user coordinate changed',
               conflict_ignore_model_grid_pixels=sum(x['conflict_ignore_model_grid_pixels'] for x in rows),
               red_negative_pixels=sum(x['red_negative_pixels'] for x in rows),items=rows)
    dump(dest/'statistics.json',stats)
    dump(dest/'index.json',dict(frozen=True,labels_frozen_before_training=True,items=items,
                              manual_label_hashes=annotation_hashes,manual_manifest_sha256=sha(TASK/'manifest.json'),
                              unmarked_pixels_are_unknown=True,red_lines_are_confirmed_non_target_centers=True,
                              normalization='raw gray only; LCN assisted human viewing, not model input',
                              source_snapshot=str(HERE/'label_audit/annotation_snapshot')))
    protocol={**prior,'version':'IR-M1-R1.0.0','task':'manual complete-visible target centers plus explicit red neighbors, U-Net+matched R1',
              'artifact_root':'C:/precise_arrangement_results/manual_r1_20261103_v1',
              'manual_task_manifest':str(TASK/'manifest.json'),'manual_annotation_hashes':annotation_hashes,
              'training_budget':dict(unet_steps=800,unet_max_seconds=2000,smallcheck_steps=100,smallcheck_max_seconds=300,
                                     relation_steps=600,relation_max_seconds=120),
              'network_input':'unchanged original gray plus independent frozen short entry; no manual far-tail prompt',
              'manual_data_stage_policy':'keep UI default stage1; includes projected-on-coil target; not physical free-only',
              'main_render':'292 unique three native panels: raw source, raw U1 heat, U1+R1 path on source',
              'main_model_rule':'last saved U1 step, relation600 or final budget step; no eval/gallery selection',
              'new_results_bulk_location_reason':'D drive had2.37GB free; new bulk artifacts on C, original assets remain unchanged',
              'input_protocol_sha256':sha(BASE/'data_expansion_d1_20261102_v1/protocol.json')}
    dump(HERE/'protocol.json',protocol)
    dump(HERE/'prepare_receipt.json',dict(exit_code=0,seconds=time.perf_counter()-started,
         command=sys.argv,source_sha256=sha(__file__),dataset_index_sha256=sha(dest/'index.json'),
         protocol_sha256=sha(HERE/'protocol.json'),manual_source_files_unchanged=all(
             sha(TASK/'annotations'/(i+'.json'))==h for i,h in annotation_hashes.items()),
         manual_lines_never_used_for_auto_entry=True,stats=stats))
    print(json.dumps({k:v for k,v in stats.items() if k!='items'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
