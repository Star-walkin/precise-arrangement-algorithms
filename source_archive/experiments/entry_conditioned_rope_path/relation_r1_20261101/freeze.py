"""Freeze unique292,301 conditions, supervision scope and finite budgets."""
from common import *
import sys,shutil,time
def main():
    assert not (HERE/'protocol.json').exists()
    old=load(H6/'inputs.json');ix=load(PAIR/'data/index.json');pair=load(PAIR/'paired_comparison.json')
    assert len(old['jobs'])==292 and len({j['id'] for j in old['jobs']})==292
    assert sum(len(j['seeds']) for j in old['jobs'])==301
    tg={i['source_group'] for i in ix['items'] if i['split']=='train'};eg={i['source_group'] for i in ix['items'] if i['split']=='eval'}
    assert len(tg)==4 and len(eg)==3 and not tg&eg
    for k in ['A','B']:assert sha(pair['main_checkpoints'][k]['path'])==pair['main_checkpoints'][k]['sha256']
    for j in old['jobs']:
        assert '20260925' not in j['image_path'] and 'reserved' not in j['dataset_use_status'].lower()
        assert sha(j['image_path'])==j['image_sha256']
        # Stable pre-inference entry rule. No score or reference-based choice.
        j['main_display_entry_index']=0
    p=dict(version='IR-R1.0.0',created_local='2026-11-01',task_request_path=r'C:\Users\Administrator\.codex\attachments\6feb6127-61fe-4a87-9c29-558a5d578736\已粘贴的文本.txt',
        images=292,entry_conditions=301,open400_missing_entry=108,jobs=old['jobs'],index_path=str(PAIR/'data/index.json'),index_sha256=sha(PAIR/'data/index.json'),
        models=pair['main_checkpoints'],train_ids=[i['id'] for i in ix['items'] if i['split']=='train'],eval_ids=[i['id'] for i in ix['items'] if i['split']=='eval'],
        train_groups=sorted(tg),eval_groups=sorted(eg),entry_display='stable saved entry index0 fixed before inference; all conditions retained separately; no max heat union',
        controlled_h6_ids=list(dict.fromkeys(old['pilot_ids']+old['extra_ids']+['L_run_20260923_01__f000585__left','dual_20260923_101517__pair0015__right'])),
        h6_control='A750 and B750 same exact source image, entry, frozen H6 parameters, no response threshold calibration; B response reused with hash verification; no old-vs-new connector comparison',
        budgets=dict(smallcheck_steps=100,smallcheck_seconds=60,formal_steps=600,formal_seconds=120,checkpoint_selection='last completed fixed-budget update, no evaluation selection',
            model='MLP input dimensionality determined by fixed spatial descriptor then64,32,1',seed=20261101,
            search_max_steps=150,search_step_original_px=24.,beam_width=8,maximum_proposals_per_parent=30,maximum_final_candidates=3,
            development_rounds=1,implementation_repair_rounds=2,no_large_network_or_budget_escalation=True),
        training_target='reviewed continuous trusted centers with existing valid low-response center negatives; four historical Codex engineering weak non-target fragments explicitly tagged weak; no new independent human truth',
        source_rules='train labels only during training/diagnostic construction;8 old eval references only after predictions are frozen; C07/C08 trusted clipped source only; S1 and reserved untouched',
        feature_cache='frozen A750 dec1 spatial feature8 channels average pooled4 model px; no full-image feature average; current/bridge/next/seed spatial sampling plus original-resolution gray profiles',
        R1_insertion='all generated edges scored before pool sort, merge and beam pruning; inference uses its own predicted previous path only',
        ablations=['frozen A+H6 vs frozen B+H6','same fixed relation pair pools untrained vs trained MLP','same candidate search relation_weight0 vs trained R1'],
        geometry_policy='valid image/search domain remains hard; no target-reference candidate insertion; weak F3 used as quality-weighted prior not an identity gate; raw continuous projection overlap not physical contact',
        outputs=dict(main_png_count=292,layout='exactly two full original4096x3000 panels plus outside header/footer/colorbar',heat='unaltered A750 center response fixed0..1 TURBO; neutral uncomputed domain',
            path='thin supported solid and weak bridge dashed; ambiguity/competitors distinguished; no body dilation'),
        no_devices=True,no_reserved_access=True,baseline_replaced=False)
    dump(HERE/'protocol.json',p)
    dest=HERE/'source_snapshot';dest.mkdir()
    for name in ['model.py','train.py','connect.py','scene.py','evaluate.py','prepare.py']:
        shutil.copyfile(BASE/name,dest/name)
    dump(HERE/'upstream_hashes.json',{str(BASE/n):sha(BASE/n) for n in ['model.py','train.py','connect.py','scene.py','evaluate.py','prepare.py']})
    print('Frozen292 sources,301 conditions; A750 frozen; R1 only trains. H6 control images',len(p['controlled_h6_ids']))
if __name__=='__main__':main()
