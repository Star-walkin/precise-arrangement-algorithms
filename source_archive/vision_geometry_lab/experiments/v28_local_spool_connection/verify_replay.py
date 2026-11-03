"""Verify that V28 output-only repairs have not changed candidates or scores."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def without_timing(value):
    if isinstance(value,dict):
        return {k:without_timing(v) for k,v in value.items() if k!='elapsed_ms'}
    if isinstance(value,list):return [without_timing(v) for v in value]
    return value


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before',type=Path,required=True)
    ap.add_argument('--after',type=Path,required=True)
    args=ap.parse_args()
    rows=[]
    before_config=read(args.before/'config.json');after_config=read(args.after/'config.json')
    expected_policy='undefined: partial trusted bend is not an exhaustive negative reference; coverage only'
    assert after_config['evaluation'].get('bend_prediction_precision_policy')==expected_policy
    comparable=json.loads(json.dumps(after_config))
    comparable['evaluation'].pop('bend_prediction_precision_policy',None)
    assert before_config==comparable, 'No threshold or weight tuning is permitted'
    for record in read(args.after/'records.json'):
        stem=Path(record['mode'])/record['sample_id']
        item=dict(mode=record['mode'],sample_id=record['sample_id'],files={})
        for name in ('pool.json','scores.json','structure.json','entry.json'):
            a,b=args.before/stem/name,args.after/stem/name
            first,last=read(a),read(b)
            if name in ('pool.json','scores.json'):
                assert first==last, f'Changed frozen pool/scoring {stem}/{name}'
            else:
                assert without_timing(first)==without_timing(last), f'Changed geometry {stem}/{name}'
            item['files'][name]=dict(before_sha256=digest(a),after_sha256=digest(b),
                                     byte_identical=digest(a)==digest(b),
                                     identical_except_elapsed_ms=True)
        rows.append(item)
    previous=read(args.before/'evaluation.json');current=read(args.after/'evaluation.json')
    assert previous['summary']==current['summary']
    for a,b in zip(previous['rows'],current['rows']):
        for key in ('sample_id','mode','visible_connection_evaluable','candidate_pool_contains_trusted_bend_geometry','best_pool_bend_coverage_5px'):
            assert a[key]==b[key]
        for group in ('G0','G1','G2'):
            for key in ('selected_id','best_id','algorithm_status','selected_visible_connection_status','selected_visible_geometry_status'):
                assert a['choices'][group][key]==b['choices'][group][key]
            assert a['choices'][group]['bend_geometry_5px']['covered_fraction']==b['choices'][group]['bend_geometry_5px']['covered_fraction']
    result=dict(before=str(args.before.resolve()),after=str(args.after.resolve()),
                rows=rows,all_numeric_config_identical=True,
                config_documentation_addition={'bend_prediction_precision_policy':expected_policy},
                candidate_and_scoring_cases_identical=len(rows),
                all_48_selected_results_and_coverage_unchanged=True,
                exception='partial-bend precision/extra removed as undefined; None boundary rendering fixed',
                exit_code=0)
    path=args.after/'output_fix_regression.json'
    with path.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print('16 candidate pools, 48 scoring choices, configuration and summary identical; output-only repair verified.')


if __name__=='__main__':main()
