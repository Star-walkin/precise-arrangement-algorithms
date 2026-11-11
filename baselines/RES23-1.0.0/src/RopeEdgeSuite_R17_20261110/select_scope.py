from common import *
plan=read(ROOT/'plan.json')
if not (ROOT/'inventory_all_sources.json').exists():dump(ROOT/'inventory_all_sources.json',plan)
videos=plan['videos'];active=[v for v in videos if v['visual_role'].startswith('active')]
controls=[v for v in videos if not v['visual_role'].startswith('active')]
assert len(active)==11 and sum(v['planned_frames'] for v in active)==4819
plan['videos']=active;plan['control_sources']=controls
plan['current_scope']='All 11 active-rope sources in full decodable extent; other-camera sources are sampled negative controls, not full candidate evaluation'
plan['evaluation_planned_frames']=4819
dump(ROOT/'plan.json',plan)
old=PARENT/'RopeEdgeSuite_R15_20261109'
states=[]
for item in controls:
    paths=sorted((old/'runs'/item['id']/'frames').glob('*.json.gz'))
    states.append(dict(id=item['id'],available_frame_count=len(paths),full_source_frames=item['planned_frames'],
        scope='sampled continuous initial segment; not full evaluation',frozen_output_dir=str(old/'runs'/item['id'])))
dump(ROOT/'control_scope.json',dict(reason='Task delivery is all active-rope videos; controls are a bounded false-positive check',sources=states))
dump(old/'negative_control_early_stop.json',dict(reason='Optional negative-control full reruns stopped after collecting continuous initial segments. Active-rope videos continue to full completion.',sources=states))
print('Full active scope',len(active),4819,'sampled controls',len(controls))
