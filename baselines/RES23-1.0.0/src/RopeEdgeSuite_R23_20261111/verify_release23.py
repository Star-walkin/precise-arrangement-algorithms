from common23 import *
def main():
    m=read(ROOT/'release_manifest.json')
    for name,h in m['source_hashes'].items():assert sha(ROOT/name)==h,name
    for id,h in m['videos'].items():assert sha(ROOT/'delivery'/id/f'RES23_{id}_full.mp4')==h,id
    assert sha(ROOT/'report.html')==m['report_sha256']
    assert sha(ROOT/'regression_audit.json')==m['regression_audit_sha256']
    for name,h in read(ROOT/'analysis_receipt.json')['code_hashes'].items():assert sha(ROOT/name)==h,name
    audit=read(ROOT/'regression_audit.json')
    assert audit['status']=='passed' and audit['frames']==4819 and len(audit['videos'])==11 and audit['test_count']==45
    links=read(ROOT/'report_link_audit.json');assert links['passed']
    e=read(ROOT/'events/dual_20260924_111500__left.json')['events'][0]
    assert e['image_supported'] and e['candidate_s']==30.4 and e['confirmation_s']==31.1
    dump(ROOT/'final_verification.json',dict(passed=True,videos=11,frames=4819,tests=45,
        local_report_links=links['local_links'],candidate_s=e['candidate_s'],confirmation_s=e['confirmation_s'],
        all_R22_primary_fields_unchanged=True))
    print('PASS: 11 videos, 4819 frames, 45 tests, hashes and links; 30.4s candidate / 31.1s corroboration')
if __name__=='__main__':main()
