"""Read-only source/split/quality inventory; never decode images or evaluation labels."""
from pathlib import Path
import hashlib,json,sys

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'source_audit.json'
MANIFEST=ROOT/'vision_geometry_lab/outputs/v11_path_identity/development_manifest.json'
HEATMAP=ROOT/'experiments/unet_rope_heatmap/runs/run_20261029_001/data/index.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    assert not OUT.exists(),'Preserve prior audit; use a distinct version if sources change'
    manifest=load(MANIFEST);old=load(HEATMAP)['items'];samples=manifest['samples']
    assert len(samples)==18 and all('20260925' not in json.dumps(s) for s in samples)
    train=[s for s in samples if s['split']=='train'];evaluation=[s for s in samples if s['split']=='holdout']
    groups={k:sorted({s['group_id'] for s in rows}) for k,rows in [('train',train),('eval',evaluation)]}
    heatmapgroups={k:sorted({s['source_group'] for s in old if s['split']==k}) for k in ['train','eval']}
    assert len(train)==10 and len(evaluation)==8 and not set(groups['train'])&set(groups['eval'])
    assert not set(groups['train'])&set(heatmapgroups['eval'])
    assert not set(groups['eval'])&set(heatmapgroups['train'])
    qualities=[]
    for s in train:
        if not s.get('quality_sha256'):continue
        p=Path(s['quality_path']);q=load(p);assert sha(p)==s['quality_sha256']
        qualities.append({'id':s['sample_id'],'path':str(p),'sha256':sha(p),
          'region_kinds':[r['kind'] for r in q['regions']],
          'trusted_rule':q.get('trusted_supervision_rule'),
          'trusted_point_counts':{k:len(v) for k,v in q.get('trusted_supervision_polylines',{}).items()},
          'policy':'clip trusted geometry before rasterization/blur; excluded areas are neither positive nor negative'})
    result={'status':'completed','exit_code':0,'date_local':'2026-10-29','command':sys.argv,
      'manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST)},'prior_heatmap_index':{'path':str(HEATMAP),'sha256':sha(HEATMAP)},
      'source_groups':groups,'counts':{'train_images':len(train),'eval_images':len(evaluation),'train_groups':len(groups['train']),'eval_groups':len(groups['eval'])},
      'prior_heatmap_groups':heatmapgroups,'cross_experiment_split_conflict':False,
      'train_quality_rules':qualities,
      'evaluation_inventory':[{'id':s['sample_id'],'group_id':s['group_id'],'annotation_path':s['annotation_path'],
        'expected_annotation_sha256':s['annotation_sha256'],'quality_sha256':s['quality_sha256']} for s in evaluation],
      'evaluation_reference_scope':{'discovery_observation':'Prior read-only discovery checked metadata/hashes of all 8 existing evaluation labels; all reviewed=true and status=partial, with centerline, edge_flange, edge_surface, free_seed; no geometric coordinates exported to model implementation.',
        'this_audit_script_reads_eval_label_content':False,
        'future_scoring_gate':'paths.json and response.npy must already exist and be hashed before each evaluation label is opened',
        'supports':'limited visible target center and two rope-side boundaries, including previously studied bend/attached portions',
        'missing':['independently confirmed target arc endpoint at true spool edge','complete current-target identity reference through ambiguous terminal areas','explicit four-stage labels','exhaustive background and neighboring-rope negative labels','independent adjacent rope identities'],
        'edge_flange_meaning':'one side of the target rope, not the task terminal spool edge',
        'edge_surface_meaning':'other side of the target rope, not the task terminal spool edge',
        'partial_endpoint_is_true_spool_edge':False},
      'excluded_diagnostic_ids':manifest['diagnostic_only_sample_ids'],
      'C07_C08_rule':'retain center through C7 and both edges cut by the same confirmed normal section; inferred tail never positive/background; derive targets only after geometry clipping',
      'withdrawn_event_claim':'27.3s is not a confirmed t1/change-layer positive. Existing static visible geometry is not an event annotation.',
      'holdout_semantics':'historically opened development evaluation by source group, not pristine blind test; remains evaluation only',
      'protected_data_policy':'9/25 and other explicit reserved sets not decoded/read; no device access',
      'minimum_additional_reference':['correct target anchor after the competing contact/attached junction','same target near the actual spool edge or independently confirmed last-visible endpoint'],
      'safe_pure_helpers':{'quality':'vision_geometry_lab.centerline_quality_v5.load_label_quality',
        'geometry_clip':'vision_geometry_lab.centerline_path_metrics_v6.clip_scoring_segments',
        'eval_reference':'vision_geometry_lab.experiments.v28_local_spool_connection.data_eval.load_evaluation_reference',
        'conditional_seed_only':'vision_geometry_lab.experiments.v28_local_spool_connection.data_eval.load_conditional_seed'},
      'unsafe_for_training_preparation':'evaluate_centerline_multiscale_v9._dataset reads every evaluation label; filter train rows before loading annotations instead',
      'source_sha256':sha(__file__)}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'completed','counts':result['counts'],'quality_rules':len(qualities),'cross_experiment_split_conflict':False}))
if __name__=='__main__':main()
