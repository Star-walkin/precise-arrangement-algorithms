"""Check saved response coordinates, finite domains and hashes, without labels."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main(run,phase):
    run=Path(run).resolve();summary=json.loads((run/f'inference_{phase}_summary.json').read_text(encoding='utf-8'));rows=[]
    for r in summary['items']:
        a=np.load(r['response_path'],mmap_mode='r');b=np.load(r['processed_response_path'],mmap_mode='r')
        info=json.loads((Path(r['response_path']).parent/'inference.json').read_text(encoding='utf-8'))
        assert a.dtype==b.dtype==np.float32 and list(a.shape)==list(b.shape)==info['shape_original_hw']
        assert sha(r['response_path'])==r['response_sha256'] and sha(r['processed_response_path'])==r['processed_response_sha256']
        x0,y0,x1,y1=info['bbox_original_xyxy'];domain=np.zeros(a.shape,bool);domain[y0:y1,x0:x1]=True
        if r['status']=='entry_unavailable':assert np.isnan(a).all() and np.all(b==0)
        else:
            assert np.isnan(a[~domain]).all() and np.isfinite(a[domain]).all()
            assert a[domain].min()>=0 and a[domain].max()<=1
            assert np.array_equal(a[domain],b[domain]) and np.all(b[~domain]==0)
        rows.append({'id':r['id'],'mode':r['mode'],'status':r['status'],'mapping_and_hash_pass':True})
    out={'phase':phase,'count':len(rows),'items':rows,'exit_code':0,'command':sys.argv,'source_sha256':sha(__file__),
         'scope':'numeric/artifact mapping verification only; no target accuracy or identity claim','evaluation_labels_read':False}
    (run/f'inference_{phase}_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'phase':phase,'verified':len(rows),'exit_code':0}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--phase',required=True)
    a=p.parse_args();main(a.run,a.phase)
