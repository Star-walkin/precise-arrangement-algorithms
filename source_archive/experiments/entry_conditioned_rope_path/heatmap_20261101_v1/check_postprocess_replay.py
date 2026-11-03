"""Verify the H7 adapter preserves H6 paths when fed unchanged B responses."""
import argparse,gzip,json,time
from common import HERE,H6,load,dump,sha
from postprocess_heatmap import process
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='h6_adapter_replay_002');args=parser.parse_args()
    out=HERE/args.run;out.mkdir(exist_ok=False);(out/'记录').mkdir();p=load(HERE/'protocol.json');rows=[];tick=time.time()
    def logical(value):
        if isinstance(value,dict):return {k:logical(v) for k,v in value.items() if k!='elapsed_seconds'}
        if isinstance(value,list):return [logical(v) for v in value]
        return value
    ids=p['user_cases']+['dual_20260923_101517__pair0345__right','dual_20260923_101643__pair0255__right']
    for job in p['jobs']:
        if job['id'] not in ids:continue
        rs=[dict(entry_index=k,seed_xy=s['xy'],response_file=r,response_sha256=sha(r)) for k,(s,r) in enumerate(zip(job['seeds'],job['response_files']))]
        process(job,rs,out,'B_REPLAY')
        with gzip.open(H6/'all_001/记录'/(job['id']+'.json.gz'),'rt',encoding='utf8') as f:old=json.load(f)
        with gzip.open(out/'记录'/(job['id']+'.json.gz'),'rt',encoding='utf8') as f:new=json.load(f)
        assert (old['selected_entry'],old['selected_candidate'])==(new['selected_entry'],new['selected_candidate'])
        for oe,ne in zip(old['entries'],new['entries']):
            assert logical(oe['free'])==logical(ne['free']) and logical(oe['full_free_observation'])==logical(ne['full_free_observation'])
            assert oe['free_projection']==ne['free_projection']
            assert oe['connection']['candidates']==ne['connection']['candidates']
        rows.append(dict(id=job['id'],same_condition_free_candidates_and_selection=True))
    assert len(rows)==4
    dump(out/'receipt.json',dict(command=__file__,rows=rows,status='passed',exit_code=0,seconds=time.time()-tick,
        changed_only='titles, comparison panel and elapsed wall-clock diagnostics; no H6 algorithm or numerical behavior changed',source_sha256=sha(HERE/'postprocess_heatmap.py')))
    print('four H6 response/path replays exactly matched')
if __name__=='__main__':main()
