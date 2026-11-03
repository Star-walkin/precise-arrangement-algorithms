"""Collect recorded visual observations; does not replace actual image viewing."""
from pathlib import Path
import hashlib,json
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'audit'/'visual_review_agent'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    expected=[v['video_id'] for v in json.loads((ROOT/'source_trace'/'source_trace.json').read_text(encoding='utf8'))['videos']]
    reviewed=[];batches=[]
    for p in sorted(FOLDER.glob('batch_*_review.json')):
        observations=json.loads(p.read_text(encoding='utf8'))
        manifest_path=FOLDER/(observations['batch']+'.json')
        manifest=json.loads(manifest_path.read_text(encoding='utf8'))
        if not observations['actually_viewed_contact_sheet']:continue
        if sha(Path(manifest['contact_sheet']))!=manifest['contact_sheet_sha256']:raise ValueError('Contact sheet changed')
        samples={v['video_id']:v for v in manifest['videos']}
        for v in observations['videos']:
            original=samples[v['video_id']]
            if v['positions_viewed']!=[s['position'] for s in original['samples']]:raise ValueError('Incomplete visual review')
            for s in original['samples']:
                if sha(Path(s['path']))!=s['sha256']:raise ValueError('Preview changed')
            reviewed.append(original|v|dict(review_record=str(p),review_record_sha256=sha(p)))
        batches.append(dict(contact_sheet=manifest['contact_sheet'],contact_sheet_sha256=manifest['contact_sheet_sha256'],
            review_record=str(p),review_record_sha256=sha(p),additional_full_preview_viewed=observations.get('additional_full_preview_viewed',[])))
    ids=[v['video_id'] for v in reviewed]
    assert len(ids)==len(set(ids)) and set(ids).issubset(expected)
    result=dict(status='complete' if set(ids)==set(expected) else 'in_progress',scope='Fixed-position visual review of existing offline four-panel previews; not identity or detection accuracy.',
        selection='Nearest saved preview to anchor+3s, source midpoint and source end, for every one of 16 prescribed source videos, including UNKNOWN. No outcome-based sample exclusion.',
        planned_videos=len(expected),reviewed_videos=len(ids),reviewed_fixed_positions=sum(len(v['samples']) for v in reviewed),
        pending_videos=[x for x in expected if x not in ids],batches=batches,videos=reviewed,
        sampled_display_state_counts=dict(Counter(state for v in reviewed for state in v['sample_statuses'])),
        sampled_state_counts_are_not_accuracy=True,
        supplemental_existing_output_previews=[s['path'] for v in reviewed for s in v['samples'] if s.get('preview_kind')=='decoded_existing_four_panel_output'],
        source_video_decoding_by_reviewer=False,new_model_forward=False,identity_accuracy_evaluated=False,
        principal_visible_limitations=[
            dict(video_id='dual_20260923_101643__right',media_time_s=30.0,issue='Accepted prefix stops within the coil while a longer dark oblique extension remains visible to the left; full-size preview also inspected.'),
            dict(video_id='L_run_20260923_01__left',positions=['anchor_plus_3s','middle','end'],issue='B response over the coil remains crowded/weak and every fixed sample is UNKNOWN despite an oblique rope being visible.'),
            dict(video_id='R_run_stop_run_20260922_01__right',position='end',issue='Accepted prefix stops near the right coil approach although the original oblique projection continues farther left.'),
            dict(video_id='L_run_stop_run_20260922_01__left',positions=['middle','end'],issue='Midpoint is UNKNOWN with visible rope; final accepted prefix ends near the left coil approach while the original projection extends farther right.'),
            dict(video_id='R_run_20260923_01__right',position='end',issue='STALE reference is correctly displayed as old evidence, not a current measured position.')],
        limitations=['Three still positions do not measure temporal smoothness between them.',
            'Gray references are historical scene tracks, not independent current-frame rope truth.',
            'Accepted color denotes supplied evidence/gate state, not complete same-rope identity.',
            'UNKNOWN is a refused/unavailable output, not evidence that no rope is present.'])
    (ROOT/'audit'/'visual_review_agent.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','planned_videos','reviewed_videos','reviewed_fixed_positions','pending_videos']}))

if __name__=='__main__':main()
