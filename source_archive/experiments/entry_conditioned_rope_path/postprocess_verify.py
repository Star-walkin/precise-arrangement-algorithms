"""Read-only final artifact verification for the frozen postprocessing experiment."""
from pathlib import Path
import argparse, json, hashlib, time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS=250_000_000
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def check_response(row):
    assert sha(row['response_path'])==row['response_sha256'],row['response_path']
    response=np.load(row['response_path'],mmap_mode='r')
    assert response.shape==(3000,4096) and response.dtype==np.float32
    assert np.isnan(response[:1050]).all() and np.isfinite(response[1050:]).all()
    assert float(np.min(response[1050:]))>=0 and float(np.max(response[1050:]))<=1
    return row['response_path']
def main(run):
    tick=time.perf_counter();run=Path(run).resolve();code=Path(__file__).parent
    inf=read(run/'batch/inference_receipt.json');post=read(run/'batch/postprocess_receipt.json')
    manifest=read(run/'batch/manifest.json');evaluation=read(run/'evaluation/summary.json')
    assert inf['forward_calls']==598 and post['images']==292 and post['conditions']==299
    assert evaluation['formal_saved_pairs']==64 and evaluation['exit_code']==0
    for model in inf['models'].values():assert sha(model['path'])==model['sha256']
    freeze=read(run/'p1_freeze.json')
    assert sha(code/'p1_segments.py')==read(run/'calibration_report.json')['source_sha256']
    with ThreadPoolExecutor(max_workers=3) as pool:
        verified=list(pool.map(check_response,inf['records']))
    counts={};caps=Counter();segments=[];times=[];all_paths=0;bridges=[]
    for image in post['records']:
        for row in image['records']:
            for method,record in row['postprocessors'].items():
                assert sha(record['paths_path'])==record['paths_sha256']
                data=read(record['paths_path']);all_paths+=1
                assert data['response_sha256']==row['response_sha256']
                key=data['variant']+'/'+method
                stat=counts.setdefault(key,Counter())
                stat['conditions']+=1;stat['selected']+=bool(data['selected_candidate_id'])
                stat['rejected']+=data['selected_candidate_id'] is None
                stat['candidates']+=len(data['candidates'])
                assert len(data['candidates'])<=3
                if method=='P1':
                    assert data['source_sha256']==sha(code/'p1_segments.py')
                    diag=data.get('extraction_diagnostics',{})
                    caps.update(diag.get('budget_events',[]));segments.append(diag.get('segments',0))
                    times.append(data['elapsed_seconds'])
                    for c in data['candidates']:
                        xy=np.asarray(c['points_xy']);assert len(c['segment_status'])==len(xy)-1
                        assert np.isfinite(xy).all() and (xy[:,0]>=0).all() and (xy[:,0]<4096).all()
                        assert (xy[:,1]>=1050).all() and (xy[:,1]<3000).all()
                        assert c['bridge_length_px']<=120.01
                        assert all(b['length_px']<=40.01 for b in c['bridges'])
                        bridges.extend(b['length_px'] for b in c['bridges'])
    assert all_paths==1196
    result={'status':'passed','exit_code':0,'verified_full_float_responses':len(verified),
      'verified_batch_paths':all_paths,'formal_cases':64,'network_weights_unchanged':True,
      'P1_source_frozen':True,'batch_counts':counts,'P1_extraction_budget_events':dict(caps),
      'P1_segments_range':[min(segments),max(segments)],
      'P1_batch_wall_seconds_range':[min(times),max(times)],
      'maximum_single_bridge_px':max(bridges,default=0),
      'coordinate_nan_padding_and_bridge_bounds_passed':True,'seconds':time.perf_counter()-tick,
      'interpretation':'artifact and numerical contracts only; no full target identity claim'}
    render=run/'batch/render_receipt.json'
    if render.exists():
        r=read(render);assert r['count']==292
        assert len(list(Path(r['output']).glob('*.jpg')))==292
        for row in r['records']:
            with Image.open(row['output']) as im:assert list(im.size)==row['dimensions'];im.verify()
        ids=sorted(set([0,1,73,145,218,291]))
        for n in ids:
            with Image.open(r['records'][n]['output']) as im:im.load()
        result.update(output_jpegs_verified=292,full_pixel_decode_samples=ids)
    missing=read(run/'missing_entries_receipt.json')
    result['missing_entry_receipt_path']=str(run/'missing_entries_receipt.json')
    result['source_and_output_code_hashes']={p.name:sha(p) for p in code.glob('*.py') if p.name.startswith(('postprocess_','p1_','calibrate_p1','example_postprocess'))}
    (run/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);main(a.parse_args().run)
