"""Check frozen added-negative masks and unchanged original center supervision."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(run):
    run=Path(run).resolve();index=load(run/'data/index.json');previous=load(index['previous_index'])
    old={i['id']:i for i in previous['items']};rows=[];tg=set();eg=set()
    for item in index['items']:
        if item['split']=='eval':
            assert item==old[item['id']],'Evaluation metadata should remain exactly unchanged'
            eg.add(item['source_group']);continue
        tg.add(item['source_group']);assert sha(item['cache'])==item['cache_sha256']
        assert sha(item['supervision_cache'])==item['supervision_cache_sha256']
        with np.load(item['cache']) as a,np.load(item['supervision_cache']) as b:
            for src,dst in [('target','center_target'),('valid','center_valid'),('stage','center_stage')]:assert np.array_equal(a[src],b[dst])
            for k in ['center_valid','neighbor_mask','background_mask','extra_ignore']:assert b[k].dtype==bool and b[k].shape==a['image'].shape
            n=b['neighbor_mask'];g=b['background_mask']
            assert not np.any(n&g) and not np.any((n|g)&~a['roi']) and not np.any((n|g)&a['valid'])
            rows.append({'id':item['id'],'center_arrays_bitwise_equal':True,'new_groups_disjoint':True,
              'new_groups_inside_input_domain':True,'old_center_overlap_pixels':int(((n|g)&a['valid']).sum()),
              'neighbor_pixels':int(n.sum()),'background_pixels':int(g.sum()),
              'remaining_ignore_pixels':int((a['roi']&~(a['valid']|n|g)).sum())})
    assert len(rows)==10 and len(tg)==4 and len(eg)==3 and not tg&eg
    review={'original_training_images_viewed':[i['id'] for i in index['items'] if i['split']=='train'],
      'native_neighbor_raw_and_marked_pairs_viewed':[i['id'] for i in index['items'] if i['id'] in ['dual_20260924_111500__left__f00240','dual_20260924_111143__right__f00130','C01__dual_20260923_101517__pair0050__right','C02__dual_20260923_101517__pair0550__right']],
      'reviewer':'Codex; engineering visual review, not independent human ground truth',
      'pretraining_coordinate_refinement':'right111143f130 short negative moved6originalpx toward visible rope-body center after raw/overlay inspection; draft retained; no model result used',
      'new_positive_length_px':0,'no_evaluation_labels_or_predictions_read':True}
    result={'exit_code':0,'items':rows,'train_groups':sorted(tg),'eval_groups_metadata_only':sorted(eg),'review':review,
      'command':sys.argv,'source_sha256':sha(__file__),'index_sha256':sha(run/'data/index.json'),
      'warning':'Mask and source checks do not independently validate weak-label identity.'}
    (run/'supervision/validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'verified_training_caches':len(rows),'eval_rows_unchanged':len(index['items'])-len(rows),'remaining_ignore_pixels':sum(r['remaining_ignore_pixels'] for r in rows),'exit_code':0}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(p.parse_args().run)
