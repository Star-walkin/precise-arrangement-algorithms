import shutil
from common import *
OLD=PARENT/'RopeEdgeSuite_R18_20261110'
cases=[
 ('dual_20260923_113745__left',[310,370,397,400,500,599],'Current rope follows the visible incoming rope; the 397/400 reflection error is absent. Tool-covered edge is withheld or explicitly inferred, with count unconfirmed.'),
 ('dual_20260923_113305__right',[2,300,499,540,580,599],'Convex initialization follows the silver/yellow interface. The false strict texture component at 499/540 is rejected; the accepted rope segment is upstream and does not establish a full path.'),
 ('dual_20260924_111500__left',[150,200,356,400,500,600],'Visible rope segments remain aligned before and after the old long tracking loss; the upper coil outline follows the visible arc.'),
 ('dual_20260923_113029__left',[1,150,200,300,450,596],'Guide-separated rope recovered on the coil side; current visible segments align with the rope. Initial boundary waits remain explicitly marked.'),
 ('dual_20260924_111143__right',[150,300,382,514,580,600],'At 15 s the tool is excluded rather than used as the coil edge. Later visible rope and arc samples show no obvious object swap.'),
 ('dual_20260923_101517__right',[93,248,400,500,580,600],'The upstream rope segment is identified. Guide-side and coil-side axes differ; extrapolation to d is not validated by this review.'),
 ('dual_20260923_101643__right',[10,50,150,300,450,540,600],'Current rope and coil arc align in inspected samples; scalar gate rejections elsewhere are not labelled as true or false detections.'),
 ('dual_20260923_113029__right',[0,400,425,428,450,550,595],'Rope initially absent or only short fragments visible near the bottom. Long enough current segments appear later; accepted late segments align with the actual rope.'),
 ('dual_20260925_161234__right',[8],'Short source: current rope and visible arc align; occluded portions are estimates.'),
 ('dual_20260925_212641__right',[3],'Four-frame source: rope and visible arc align in the last frame; no complete-minute claim.'),
 ('dual_20260925_232941__right',[4],'BG11 repairs initial shape gating. The visible arc follows the edge; rope-covered portions remain inferred and physical contact is unconfirmed.')]
review=[]
for id,frames,note in cases:
    run=ROOT/'runs'/id if (ROOT/'runs'/id/'receipt.json').exists() else OLD/'runs'/id
    receipt=read(run/'receipt.json');assert receipt['status']=='complete'
    review.append(dict(id=id,frames=frames,run=str(run),reviewer='assistant visual inspection',finding=note,
        reviewed_original_and_overlay=True,subpixel_error_measured=False,independent_ground_truth=False))
dump(ROOT/'visual_review.json',dict(scope='53 selected final-output frames across all 11 sources, plus targeted intermediate experiments; not an exhaustive human-labelled accuracy study',
    selected_final_frames=sum(len(x['frames']) for x in review),videos=review,
    limitations=['Full-path identity, d recalibration, F3 root counts and physical contact were not validated.',
        'Gate availability is not detection accuracy. Extra rejections may be false negatives.',
        'Upstream straight segments may differ from the axis after the guide.']))
out=ROOT/'frozen';out.mkdir(exist_ok=True);hashes={}
names=['common.py','tracker_sg5.py','boundary_bg9.py','boundary_bg10.py','boundary_bg11.py','camera_cg2.py','engine.py',
       'run_one.py','run_fresh.py','render.py','audit_invariance.py','plan.json']
for name in names:
    p=ROOT/name;shutil.copy2(p,out/name);hashes[str(p)]=sha(p)
for p in [R12/'tracker_sg2.py',R11/'region_bg5_refined.py',PARENT/'RopeEdgeSuite_R15_20261109/tracker_sg3.py']:
    hashes[str(p)]=sha(p)
equivalent={name:sha(ROOT/name)==sha(OLD/name) for name in ['common.py','tracker_sg5.py','boundary_bg9.py','boundary_bg10.py']}
source_equivalent={name:(ROOT/name).read_text(encoding='utf8')==(OLD/name).read_text(encoding='utf8') for name in equivalent}
assert all(source_equivalent.values()),'Parent algorithm source differs beyond newline encoding'
audit=read(ROOT/'trace_invariance.json');assert len(audit['videos'])==11
assert [r['id'] for r in audit['videos'] if not r['numerically_invariant']]==['dual_20260925_232941__right']
assert audit['candidate_sha256']==sha(ROOT/'boundary_bg11.py')
assert len(read(ROOT/'sg5_regression_test.json')['cases'])==15
assert sum(len(r['checks']) for r in read(OLD/'fresh_replay_parity.json')['tests'])==24
assert sum(len(r['frames']) for r in read(ROOT/'revision_fresh_parity.json')['tests'])==21
dump(out/'hashes.json',dict(algorithm='RES-1.8.1/SG5-BG10-BG11-CG2',files=hashes,parent_component_byte_equivalence=equivalent,
    parent_component_source_equivalence_after_newline_normalization=source_equivalent,
    newline_note='boundary_bg10.py was copied with CRLF rather than LF; source is otherwise identical.',
    validated_BG10_full_batch=str(OLD),newly_recomputed_source='dual_20260925_232941__right',
    other_outputs_reused_only_after_exhaustive_trace_invariance_audit=True))
dump(ROOT/'iteration_log.json',dict(trials=[
    dict(version='SG3/BG8',finding='Recovered missed ropes, but reflection drift at 113745 397/400; candidate rejected.'),
    dict(version='SG4/BG9',finding='Strict priority fixed one drift but retained SG2 texture error at 113305 499/540; candidate rejected.'),
    dict(version='SG5/BG9',finding='Both branches checked at current fixed entry. Two false-identity classes rejected; current supported alternatives retained.'),
    dict(version='SG5/BG10',finding='Convex spatial consensus fixes initial outlier contamination. Full 11-source evaluation, 4819 frames.'),
    dict(version='SG5/BG11',finding='Bounded startup exception repairs short near-edge source, 0/9 to 6/9 boundary gates. Other 10 sources numerically invariant by recorded-transition audit.')],
    not_claimed=['All-frame accuracy','Full F3/root-count validation','Real-time throughput','Physical contact calibration']))
print('Evidence finalized:',sum(len(x['frames']) for x in review),'selected frames; 11 sources',flush=True)
