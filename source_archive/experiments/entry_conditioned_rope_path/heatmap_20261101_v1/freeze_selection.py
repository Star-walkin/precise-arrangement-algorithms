"""Make the predeclared training-only choice immutable before evaluation."""
from datetime import datetime,timezone
from common import HERE,H6,load,dump,sha
def main():
    assert not (HERE/'selection.json').exists()
    protocol=load(HERE/'protocol.json');paired=load(HERE/'paired_comparison.json');common=load(HERE/'common_checkpoints.json')
    assert paired['status']=='frozen_training_only_selection' and paired['same_origin_and_initial_state']
    assert paired['shared_input_target_masks_hashes_exact'] and not paired['evaluation_or_gallery_used_for_choice']
    assert paired['main_comparison_step']==common['step']<=400
    assert load(HERE/'smallcheck/result.json')['status']=='passed'
    assert load(HERE/'baseline_replay/receipt.json')['status']=='passed'
    assert load(HERE/'h6_adapter_replay_002/receipt.json')['status']=='passed'
    for v,m in common['models'].items():assert sha(m['path'])==m['sha256']
    for p,s in protocol['source_H6_core'].items():assert sha(p)==s
    frozen={str(HERE/n):sha(HERE/n) for n in ['network.py','preprocessing.py','train_heatmap.py','infer_heatmaps.py','postprocess_heatmap.py','legacy_h6.py','common.py','protocol.json','evaluate_heatmaps.py','evaluation_protocol_final.json']}
    dump(HERE/'selection.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),selected_variant=paired['selected_variant'],
        models=common['models'],common_additional_step=common['step'],guards=paired['guards'],accepted_by_training_guards=paired['accepted_by_training_guards'],
        scores=paired['scores'],source_hashes=frozen,paired_comparison_sha256=sha(HERE/'paired_comparison.json'),training_fit_sha256=sha(HERE/'common_training_fit.json'),
        policy=protocol['selection'],evaluation_references_used_for_selection=False,
        evaluation_scope_predefined_from_frozen_legacy_protocol=True,selected_for='This bounded experimental comparison/gallery only',
        established_B_replaced=False,production_model_changed=False))
    print('Frozen training-only choice',paired['selected_variant'],'at common step',common['step'])
if __name__=='__main__':main()
