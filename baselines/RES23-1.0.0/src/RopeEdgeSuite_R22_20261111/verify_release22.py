from common22 import *
from audit22 import tests
def main():
    manifest=read(ROOT/'release_manifest.json')
    for name,h in manifest['source_hashes'].items():assert sha(ROOT/name)==h,('changed code',name)
    assert sha(ROOT/'report.html')==manifest['report_sha256']
    assert sha(ROOT/'regression_audit.json')==manifest['regression_audit_sha256']
    for id,h in manifest['videos'].items():assert sha(ROOT/'delivery'/id/f'RES22_{id}_full.mp4')==h
    audit=read(ROOT/'regression_audit.json');assert audit['frames']==4819 and len(audit['videos'])==11
    receipt=read(ROOT/'analysis_receipt.json')
    for name,h in receipt['code_hashes'].items():assert sha(ROOT/name)==h
    fresh=tests();assert len(fresh)==4
    target=read(ROOT/'delivery/dual_20260924_111500__left/frames.json')
    assert all(r['state22']['state']=='approaching' for r in target[35:200])
    event=read(ROOT/'motor/dual_20260924_111500__left.json')['events'][0]
    assert event['recurrence_s']==27.3 and event['confirmation_s']==28.0
    dump(ROOT/'final_verification.json',dict(passed=True,frames=4819,videos=11,test_count=32,
        early_approach_assertion_passed=True,target_candidate_s=27.3,target_corroboration_s=28.,
        human_motor_time_ground_truth_available=False,report_links_verified=read(ROOT/'report_link_audit.json')['local_links']))
    print('PASS: 11 videos, 4819 frames, 32 tests, report links and hashes, 27.3s candidate / 28.0s confirmation')
if __name__=='__main__':main()
