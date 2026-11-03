from common import *
from spatial import P
def main():
    assert not (HERE/'selection_freeze.json').exists()
    d=load(HERE/'training_diagnostics.json');r=load(HERE/'model/training_receipt.json')
    assert r['steps']==600 and r['smallcheck_passed']
    dump(HERE/'selection_freeze.json',dict(date_local='2026-11-02',main_display='frozen A750 response + experimental R1 path; never replace raw response',
        relation_checkpoint='last600 fixed-budget update',relation_sha256=sha(HERE/'model/final_portable.npz'),parameters=P,
        train_R0_attached=d['aggregate']['R0']['attached']['coverage'],train_R1_attached=d['aggregate']['R1']['attached']['coverage'],
        near_edge_training_coverage_both=0.,candidate_upgrader='bounded original-gray and A peak proposals already added; no next-point CNN',
        reason='Training partial-reference postcontact coverage improves but near-edge remains missing. Freeze as an unaccepted experiment, keep failures and report evaluation regressions. No evaluation selection.',
        eval_references_opened=False,no_further_parameter_search=True,
        code_hashes={str(HERE/n):sha(HERE/n) for n in ['common.py','spatial.py','search_r1.py','run_paths.py','train_relation.py']}))
    print('Frozen R1 before8-case evaluation; no baseline replacement')
if __name__=='__main__':main()
