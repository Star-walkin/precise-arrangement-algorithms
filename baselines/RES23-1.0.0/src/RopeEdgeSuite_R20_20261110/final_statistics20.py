"""Summarize availability and limitations without equating them to accuracy."""
import collections
from common20 import *

def near_windows(rows,fps):
    result=[];current=[]
    def close():
        if len(current)>=3:
            result.append(dict(start_s=current[0]['time_s'],end_s=current[-1]['time_s']+1/fps,
                start_frame=current[0]['frame'],end_frame=current[-1]['frame'],frames=len(current),
                states=dict(collections.Counter(x['contact']['state'] for x in current)),
                semantics='continuous_current_near_edge_observation_window_not_confirmed_material_contact',
                physical_contact_confirmed=False,bridged_missing_frames=0))
    for r in rows:
        eligible=r['contact']['state'] in ('near_edge_uncertain','visual_contact_candidate')
        if current and (not eligible or r['epoch']!=current[-1]['epoch']):close();current=[]
        if eligible:current.append(r)
    close();return result

def main():
    assert read(ROOT/'delivery_receipt.json')['status']=='complete'
    videos=[]
    for item in plan():
        id=item['id'];out=ROOT/'delivery'/id;rows=read(out/'frames.json');s=read(out/'summary.json');cal=s['calibration'];categories=collections.Counter();jumps=[]
        for k,r in enumerate(rows):
            if r['contact']['state']=='projection_conflict':category='projection_geometry_conflict_with_signed_diagnostic'
            elif r['available']:category='observed_F3_phase' if r['complete_F3_phase'] else 'conditional_geometry_with_current_F3_pitch'
            elif r['d'] is None:category='fixed_d_unidentified'
            elif r['status']=='projection_geometry_inconsistent':category='projection_geometry_inconsistent'
            else:category='insufficient_current_image_evidence'
            categories[category]+=1
            if k and r['available'] and rows[k-1]['available'] and r['epoch']==rows[k-1]['epoch']:
                change=r['N']-rows[k-1]['N']
                if abs(change)>.75:jumps.append(dict(frame=k,change_N=change))
        near=near_windows(rows,item['fps']);dump(out/'near_observation_windows.json',near)
        videos.append(dict(id=id,frames=len(rows),categories=dict(categories),current_contact_states=dict(collections.Counter(r['contact']['state'] for r in rows)),near_observation_windows=near,
            predicted_frames=sum(r['filtered']['source']=='short_prediction_max_0_5s' for r in rows),large_adjacent_changes=jumps,
            contact_windows=s['contact_windows'],fixed_d_phases=sum(p['d'] is not None for p in cal['epochs']),
            calibration_references=sum(len(p['far_guard_accepted_frames']) for p in cal['epochs'] if p['d'] is not None)))
    repairs=[]
    for name in ('roi22_correction_receipt.json','fragments23_correction_receipt.json','fragments24_correction_receipt.json'):
        repairs.extend(dict(x,receipt=name) for x in read(ROOT/name)['promoted'])
    assert len({(x['id'],x['frame']) for x in repairs})==len(repairs)
    result=dict(videos=videos,total_frames=sum(x['frames'] for x in videos),
        categories=dict(sum((collections.Counter(x['categories']) for x in videos),collections.Counter())),
        restored_current_measurements_after_initial_full_run=len(repairs),restored_frames=repairs,
        contact_states=dict(sum((collections.Counter(x['current_contact_states']) for x in videos),collections.Counter())),
        independent_count_ground_truth_available=False,independent_d_ground_truth_available=False,
        independent_material_contact_truth_available=False,accuracy_not_estimated=True)
    dump(ROOT/'final_statistics.json',result)
    print({k:v for k,v in result.items() if k not in ('videos','restored_frames')},flush=True)

if __name__=='__main__':main()
