"""Mechanism tests with known synthetic paths; no real-image accuracy claims."""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,sys
import numpy as np
from causal_gate import GateConfig,TemporalRopeGate,compare_paths

HERE=Path(__file__).resolve().parent

def candidate(offset=0.,score=.8,ident='a',length=600.,start=0.,tail_bend=0.,supported=True):
    x=np.linspace(start,start+length,121)
    y=.12*x+offset+tail_bend*np.clip((x-start)/length-.65,0,1)
    return dict(id=ident,points_xy=np.column_stack([x,y]).tolist(),image_score=score,image_supported=supported)

def run():
    checks=[]
    def record(name,**result):checks.append(dict(name=name,passed=True,**result))

    # Motion of one24px projected pitch over3s should pass at5Hz.
    g=TemporalRopeGate();states=[]
    for t in np.arange(0.,3.01,.2):states.append(g.update(float(t),[candidate(8*t)],pitch_px=24))
    assert all(s['accepted'] for s in states)
    assert len(g.history)==5 and max(r['time_s'] for r in g.history)==states[-1]['time_s']
    record('smooth_expected_motion',frames=len(states),accepted=sum(s['accepted'] for s in states))

    # An isolated wrong lane has a higher image score but cannot enter history.
    before=g.snapshot();bad=g.update(3.2,[candidate(8*3.2+60.,.99,'jump')],pitch_px=24)
    assert bad['status']=='STALE' and bad['points_xy'] is None and bad['reference_points_xy']
    assert g.snapshot()['accepted_history']==before['accepted_history']
    back=g.update(3.4,[candidate(8*3.4,.8,'back')],pitch_px=24)
    assert back['accepted'] and back['accepted_candidate_id']=='back'
    record('isolated_jump_rejected_without_history_contamination',reason=bad['candidate_diagnostics'][0]['temporal']['reasons'])

    # Compare candidates without nearest-neighbour relabeling: exactly one-pitch
    # parallel displacement is rejected even though both paths are straight.
    g=TemporalRopeGate();g.update(0,[candidate()],pitch_px=24)
    r=g.update(.2,[candidate(24.,.99,'wrong'),candidate(1.6,.6,'supported')],pitch_px=24)
    assert r['accepted_candidate_id']=='supported'
    assert not r['candidate_diagnostics'][0]['temporal']['passed']
    record('lower_rank_consistent_candidate_beats_parallel_lane_jump')

    # A persistent new image-supported location only becomes a new reference
    # after old output expiry; it never asserts same identity through the gap.
    g=TemporalRopeGate();g.update(0,[candidate()],pitch_px=24)
    rejected=[]
    for t in [.4,.8,1.2]:rejected.append(g.update(t,[candidate(80.,.9,'persistent')],pitch_px=24))
    assert rejected[0]['status']=='STALE' and rejected[1]['status']=='STALE'
    assert rejected[2]['status']=='REACQUIRED' and rejected[2]['epoch']==1
    assert rejected[2]['identity_continuity_through_gap'] is False
    assert len(g.history)==1 and g.history[0]['id']=='persistent'
    record('persistent_change_reacquired_as_new_epoch',statuses=[r['status'] for r in rejected])

    # Missing/unsupported detections cannot refresh a displayed historic line.
    g=TemporalRopeGate();g.update(0,[candidate()],pitch_px=24)
    r1=g.update(.5,[],pitch_px=24);r2=g.update(1.1,[candidate(supported=False)],pitch_px=24)
    assert r1['status']=='STALE' and r1['reference_age_s']==.5
    assert r2['status']=='UNKNOWN' and r2['points_xy'] is None and r2['reference_points_xy'] is None
    assert g.history[-1]['time_s']==0 and not g.pending
    record('occlusion_expires_and_unsupported_prediction_never_accepted')

    # Explicit scene boundary and input-time gap must not inherit a stale rope.
    g=TemporalRopeGate();g.update(0,[candidate()],pitch_px=24)
    cut=g.update(.2,[candidate(300.)],pitch_px=24,scene_cut=True)
    gap=g.update(4.,[candidate(-200.)],pitch_px=24)
    assert cut['reset_reason']=='scene_cut' and cut['epoch']==1 and cut['accepted']
    assert gap['reset_reason']=='large_input_time_gap' and gap['epoch']==2 and gap['accepted']
    record('scene_cut_and_large_timestamp_gap_reset_without_old_identity')

    # Reject nonmonotonic timestamps without mutating state.
    before=json.dumps(g.snapshot(),sort_keys=True)
    for timestamp in [4.,3.9,float('nan')]:
        try:g.update(timestamp,[candidate()],pitch_px=24)
        except ValueError:pass
        else:raise AssertionError('bad timestamp accepted')
    assert json.dumps(g.snapshot(),sort_keys=True)==before
    record('duplicate_reverse_and_nonfinite_time_rejected_atomically')

    # End shortening alone is not a large position jump; untested tails remain
    # exposed in diagnostics, so partial visibility is not full coverage.
    g=TemporalRopeGate();g.update(0,[candidate(length=700.)],pitch_px=24)
    shorter=g.update(.2,[candidate(1.6,length=350.)],pitch_px=24)
    assert shorter['accepted']
    q=shorter['candidate_diagnostics'][0]['temporal']
    assert q['reference_uncompared_length_px']>300 and q['common_length_px']>300
    record('partial_tail_not_conflated_with_global_jump',untested_reference_px=q['reference_uncompared_length_px'])

    # A later bend at the same entrance must not be hidden by a stable median.
    g=TemporalRopeGate();g.update(0,[candidate()],pitch_px=24)
    turn=g.update(.2,[candidate(tail_bend=220.)],pitch_px=24)
    assert not turn['accepted'] and 'excessive_lateral_jump' in turn['candidate_diagnostics'][0]['temporal']['reasons']
    record('large_tail_departure_rejected_despite_unchanged_entrance')

    # Prove causal prefix invariance under different future paths.
    streams=[]
    for future in [0.,200.]:
        h=TemporalRopeGate();out=[]
        for t in [0.,.2,.4,.6]:out.append(h.update(t,[candidate(8*t if t<.6 else future)],pitch_px=24))
        streams.append(out)
    assert streams[0][:3]==streams[1][:3]
    record('future_input_cannot_change_saved_prefix_outputs')

    # Candidate cap and malformed paths are bounded and visible.
    g=TemporalRopeGate()
    try:g.update(0,[candidate() for _ in range(4)],pitch_px=24)
    except ValueError:pass
    else:raise AssertionError('unbounded candidates accepted')
    q=g.update(0,[dict(id='bad',points_xy=[[0,0]],image_score=.9,image_supported=True)])
    assert q['status']=='UNKNOWN' and q['pitch_source']=='explicit_uncalibrated_fallback'
    record('bounded_candidates_invalid_geometry_and_fallback_scale')

    result=dict(exit_code=0,checks=checks,passed=len(checks),config=asdict(GateConfig()),
        command=sys.argv,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'causal_gate.py',Path(__file__)]},
        scope='Synthetic temporal mechanism tests only; no imagery, hardware, labels, future frames or identity accuracy tested.')
    (HERE/'test_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(exit_code=0,passed=len(checks))))

if __name__=='__main__':run()
