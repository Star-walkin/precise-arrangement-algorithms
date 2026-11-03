from pathlib import Path
import hashlib,json,csv
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'camera_condition_whole_incoming_20261028_v4'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
extra=json.loads((OLD/'protocol_addendum_002.json').read_text(encoding='utf-8'))['posthoc_cases']
pilot=list(dict.fromkeys(old['pilot_inputs']+extra))
data=HERE.parent/'camera_condition_images_20261009_v2'
rows=list(csv.DictReader((data/'manifest.csv').open(encoding='utf-8-sig')))
review=[]
for role in ['left','right']:
 pool=[r for r in rows if r['camera_role']==role and r['output_file'] not in pilot]
 pool.sort(key=lambda r:hashlib.sha256(('A5fixedextra29:'+r['output_file']).encode()).hexdigest())
 groups=set()
 for r in pool:
  if r['source_group'] in groups:continue
  review.append(r['output_file']);groups.add(r['source_group'])
  if len(groups)==4:break
p={'version':'IR-A5.0.0','name':'差分轨道 A5','created':'2026-10-29','input_manifest_sha256':old['input_manifest_sha256'],'parent_manifest_sha256':old['parent_manifest_sha256'],
 'pilot_inputs':pilot,'extra_review_inputs':review,'denominator':400,
 'user_feedback':'User estimates A4 accuracy around50%; qualitative estimate without fixed truth/denominator, not adopted as measured accuracy. n-1 rationale withdrawn as main decision.',
 'budget':'Maximum3 fixed20 development pilots with explicit diagnosed revisions; then freeze8 additional preset open-development reviews and same400. Implementation failures preserved. No per-image parameter tuning, training or heldout access.',
 'architecture':['Keep original external free segment remeasurement.','Compare whole-width inner intensity and shadows against both neighboring background bands; preserve bright/dark polarity and uncertainty.','Use formal F3 merged mask to authorize boundary rows, form adjacent paired-edge rope lanes, generate finite straight-to-lane transitions with competitors.','Score shared candidate pool with geometry-only G0 and image evidence G2; n-1 disabled.','Keep unsupported/competing curved segments review-marked; do not merge different paths into a single rope mask.'],
 'continue_stop':'Continue a revision only for a diagnosed sampling/candidate/scoring failure with visible or structural evidence. Do not claim identity accuracy without independent truth; do not force invisible tails. No new network or label requirement.',
 'fixed_evaluation':'Save predictions before review. Compare original/A4/A5 at fixed20 and preset8; report restored visible span, wrong turns, early termination and unknown. Free-entry failures stay in400 denominator.',
 'source_parent_sha256':{str(OLD/f):sha(OLD/f) for f in ['whole_io.py','free_rope.py','whole_path.py','boundary_occlusion.py','run_whole.py','freeze_all_002.json','protocol.json','protocol_addendum_002.json']},
 'A4_output_manifest_sha256':sha(OLD/'all_002/output_manifest.json'),'A4_output_root':str(OLD/'all_002'),'F3_sources_freeze_sha256':sha(HERE.parent/'camera_condition_full_surface_joint_20261028_v3/freeze_all.json'),
 'equipment':False,'reserved_data':False,'HTML':False,'accuracy_ground_truth_available':False}
assert not (HERE/'protocol.json').exists()
(HERE/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
f=HERE/'whole_io.py';s=f.read_text(encoding='utf-8').replace("VERSION='IR-A4.0.1'","VERSION='IR-A5.0.0'")
s=s.replace("return rows if mode=='all' else [r for r in rows if r['output_file'] in p['pilot_inputs']]","return rows if mode=='all' else [r for r in rows if r['output_file'] in p['extra_review_inputs' if mode=='extra' else 'pilot_inputs']]")
f.write_text(s,encoding='utf-8')
print('Protocol saved: pilot',len(pilot),'extra',len(review),'fixed400.')
