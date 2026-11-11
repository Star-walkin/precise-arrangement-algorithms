"""Recover only bookkeeping after all 601 frozen detector tasks completed.

The original runner's final receipt referenced the inner-scope `prog` variable.
No detector result or mask is modified here. Recovery verifies frozen sources,
all frame indices, the original video hash and the saved final progress first.
"""
import json
from run_full import ROOT,SOURCE,EXPECTED,sha,dump
if __name__=='__main__':
    run=ROOT/'full_002';binding=json.loads((run/'binding.json').read_text(encoding='utf8'))
    assert all(sha(p)==v for p,v in binding['code_sha256'].items())
    assert sha(SOURCE)==EXPECTED
    progress=json.loads((run/'progress.json').read_text(encoding='utf8'));assert progress['processed']==601
    rows=[]
    for i in range(601):
        r=json.loads((run/'frames'/f'{i:06d}.json').read_text(encoding='utf8'))
        assert r['frame_index']==i and r['source_video_sha256']==EXPECTED
        rows.append(dict(frame=i,region_valid=r['region']['valid'],F3_called=r['F3_called'],count_valid=r['count_valid'],
                         display_N=r['display_N'],reasons=r['reasons'],seconds=r['elapsed_seconds']))
    receipt=dict(status='completed',**progress,frame_rows=rows,source_unchanged=True,
                 gate_blocks=sum(not r['region_valid'] for r in rows),
                 region_failed_F3_calls=sum(r['F3_called'] and not r['region_valid'] for r in rows),
                 finalization_recovered=True,original_runner_exit_code=1,
                 original_runner_error="NameError: prog was local to consume() in receipt finalization",
                 recovery_script_sha256=sha(__file__),detector_outputs_changed_by_recovery=False)
    dump(run/'receipt.json',receipt)
    print({k:v for k,v in receipt.items() if k!='frame_rows'},flush=True)
    # Fixed entry point for future runs; keep the actual executed runner intact
    # so its frozen hash remains directly verifiable.
    original=(ROOT/'run_full.py').read_text(encoding='utf8')
    fixed=original.replace("dump(out/'receipt.json',dict(status='completed',**prog,frame_rows=rows", "prog=json.loads((out/'progress.json').read_text(encoding='utf8'))\n    dump(out/'receipt.json',dict(status='completed',**prog,frame_rows=rows")
    assert fixed!=original
    (ROOT/'run_full_fixed.py').write_text(fixed,encoding='utf8')
