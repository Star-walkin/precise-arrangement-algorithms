"""Freeze source groups, budget and comparisons before the new training."""
import sys
from collections import Counter
from datetime import datetime,timezone
from common import HERE,H6,OLD_INDEX,B_MODEL,load,dump,sha
def main():
    assert not (HERE/'protocol.json').exists()
    index=load(OLD_INDEX);jobs=load(H6/'inputs.json');train=[i for i in index['items'] if i['split']=='train'];ev=[i for i in index['items'] if i['split']=='eval']
    tg=sorted({i['source_group'] for i in train});eg=sorted({i['source_group'] for i in ev})
    assert len(train)==10 and len(ev)==8 and not set(tg)&set(eg)
    assert len(jobs['jobs'])==292 and sum(len(j['seeds']) for j in jobs['jobs'])==301
    for j in jobs['jobs']:
        assert '20260925' not in j['image_path'] and sha(j['image_path'])==j['image_sha256']
    p=dict(version='IR-H7.0.0',created_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        purpose='Diagnose and improve raw target-center response over the spool, not recolor output or paint selected paths into it.',
        legacy_index=dict(path=str(OLD_INDEX),sha256=sha(OLD_INDEX)),source_B_model=dict(path=str(B_MODEL),sha256=sha(B_MODEL)),
        source_H6_inputs_sha256=sha(H6/'inputs.json'),source_H6_core=load(H6/'freeze_final.json')['sources'],
        train_ids=[i['id'] for i in train],train_source_groups=tg,eval_ids=[i['id'] for i in ev],eval_source_groups=eg,
        training_supervision='Frozen existing reviewed partial centers and explicit weak negatives only; no H6 selected paths or new post-contact pseudo truth.',
        supervision_sources=[dict(id=i['id'],cache=i['cache'],cache_sha256=sha(i['cache']),supervision_cache=i['supervision_cache'],supervision_cache_sha256=sha(i['supervision_cache'])) for i in train],
        budget=dict(overfit_steps=60,overfit_seconds=120,matched_arms=['C','D'],maximum_steps_each=400,maximum_seconds_each=720,checkpoint_interval=50,repeat_policy='At most one repair for an implementation failure; no repeated tuning on evaluation or all292 results.'),
        arms=dict(B='Frozen legacy step750 baseline',C='Same B initialization; restore relative positive supervision weight with lower engineering negative weights; raw-gray plus seed',D='Matched C training plus a fixed float local-contrast auxiliary channel; original gray is retained'),
        model='Existing base8 entry-conditioned U-Net; auxiliary first-layer weights initialized to zero; all branches initially exactly reproduce B',
        input=dict(shape=[975,2048],bbox_original_xyxy=[0,1050,4096,3000],scale=.5,photometry_order='Nonnegative raw photometry then weighted full-search-ROI LCN then neutral padding; no gamma on standardized or signed residual tensors',
            lcn_sigma_original_px=32,lcn_std_floor_gray255=2.5,lcn_clip=[-3,3],lcn_scale='predefined texture contrast scale, not calibrated physical rope width',true_zero_kept=True),
        selection=dict(scope='training10 only, largest common saved step; evaluation images never select model or parameters',
            score='same stage-balanced high/low center MSE + .10 neighbor zero MSE + .05 explicit-background zero MSE',
            guards='No trained positive-stage mean drop >.025 from B; explicit background mean <=max(.04,B+.02), neighbor mean <=max(.05,B+.02).',
            failure_policy='If neither candidate passes, keep B as the established baseline and export the best experimental candidate clearly marked unaccepted; do not claim the requested near-universal outcome.'),
        user_cases=['L_run_20260923_01__f000585__left','dual_20260923_101517__pair0015__right'],
        gallery_fixed_diagnostic_ids=list(dict.fromkeys(jobs['pilot_ids']+jobs['extra_ids']+['L_run_20260923_01__f000585__left','dual_20260923_101517__pair0015__right'])),
        source_count=400,run_images=292,entry_conditions=301,unresolved_old_entries=108,
        jobs=jobs['jobs'],source_group_counts=dict(Counter(j['source_group'] for j in jobs['jobs'])),
        heat_display='fixed0..1 same TURBO; no per-image stretch, no skeleton/path raster fed back to network response',
        evaluation='Freeze candidate choice and all inference before existing8 evaluation references are scored; known-entry and AUTO separately; same open development groups, not blind test.',
        downstream='H6 geometric and image gates remain frozen, rerun from new response; old paths are comparison only, never learned target identity.',
        reserved_access=False,hardware=False,old_sources_unchanged=True)
    dump(HERE/'protocol.json',p);print('Frozen:10 train/4 groups;8 eval/3 groups;2x400-step candidates;292/301 batch')
if __name__=='__main__':main()
