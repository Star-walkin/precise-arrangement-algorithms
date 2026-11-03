"""Cache AUTO entry hints using exactly the preparation Gaussian; no labels read."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from prepare import response_from_lines,PARAMETERS

def main(run):
    data=Path(run).resolve()/'data';index=json.loads((data/'index.json').read_text(encoding='utf-8'));rows=[]
    for item in index['items']:
        if item['split']!='eval':continue
        path=Path(item['auto_entry_path']);entry=json.loads(path.read_text(encoding='utf-8'))
        if not entry['available']:
            rows.append({'id':item['id'],'available':False,'reason':entry['reason']});continue
        with np.load(item['cache'],allow_pickle=False) as z:shape=z['image'].shape
        seed,_=response_from_lines([entry['free_seed']],shape,item['bbox'],item['scale'],PARAMETERS['seed_sigma_original_px'])
        assert seed.sum()>0 and np.isfinite(seed).all()
        out=path.with_suffix('.seed.npy');np.save(out,seed.astype(np.float32))
        rows.append({'id':item['id'],'available':True,'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'shape':list(shape)})
    result={'command':sys.argv,'exit_code':0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'gaussian_sigma_original_px':PARAMETERS['seed_sigma_original_px'],'evaluation_labels_read':False,'items':rows}
    (data/'auto_seed_cache_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'auto_seed_caches':sum(r['available'] for r in rows),'unavailable':sum(not r['available'] for r in rows)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(p.parse_args().run)
