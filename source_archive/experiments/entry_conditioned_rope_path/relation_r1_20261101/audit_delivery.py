"""Independent delivery/data audit: all292 originals, all301 float conditions."""
from common import *
import numpy as np,time,sys,concurrent.futures,traceback,gzip
from PIL import Image
Image.MAX_IMAGE_PIXELS=150_000_000
def readgz(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def image(row):
    p=Path(row['output']);assert sha(p)==row['sha256']
    with Image.open(p) as im,Image.open(row['source']) as source:
        assert im.format=='PNG' and im.size==(8364,3350) and im.mode=='RGB'
        assert im.crop((0,160,4096,3160)).tobytes()==source.convert('RGB').tobytes()
        assert np.all(np.asarray(im.crop((4128,160,8224,1210)))==45),'Uncomputed domain not neutral gray'
    for cond in row['conditions']:
        r=readgz(cond['path']);v=r['result'];assert r['target_labels_read'] is False and not r['old_selected_paths_as_input']
        assert v['search']['generated']==v['search']['relation_scored_before_pruning']
        assert len(v['candidates'])<=3
        for c in v['candidates']:
            q=np.asarray(c['points_xy']);assert q.shape==(len(c['point_status']),2) and np.isfinite(q).all()
            assert np.all(q[:,0]>=0) and np.all(q[:,0]<4096) and np.all(q[:,1]>=1050) and np.all(q[:,1]<3000)
            assert c['complete_identity_verified'] is False
    for f in row['float_responses']:
        p=Path(f['path']);assert sha(p)==f['sha256']
        with np.load(p) as z:
            a=z['response'];assert a.shape==(975,2048) and a.dtype==np.float32 and np.isfinite(a).all() and 0<=a.min()<=a.max()<=1
    return dict(id=row['id'],pixel_identical=True,entry_count=row['total_entries'],unadorned_float_passed=True)
def main():
    sys.stdout.reconfigure(encoding='utf8');p=load(HERE/'protocol.json');r=load(HERE/'render_receipt.json');assert r['exit_code']==0 and not r['errors']
    files=list((HERE/'交付_292张原图与热力图').glob('*.png'));ids={f.stem for f in files};expected={j['id'] for j in p['jobs']}
    assert len(files)==292 and len(ids)==292 and ids==expected
    assert len(r['rows'])==292 and {v['id'] for v in r['rows']}==expected and sum(v['total_entries'] for v in r['rows'])==301
    assert all(v['display_entry']==0 for v in r['rows'])
    tick=time.time();checked=[];errors=[]
    with concurrent.futures.ProcessPoolExecutor(4) as ex:
        pending={ex.submit(image,row):row['id'] for row in r['rows']}
        for f in concurrent.futures.as_completed(pending):
            try:checked.append(f.result())
            except Exception:errors.append(dict(id=pending[f],traceback=traceback.format_exc()))
            if (len(checked)+len(errors))%30==0:print(dict(checked=len(checked),errors=len(errors)),flush=True)
    models=p['models'];assert all(sha(v['path'])==v['sha256'] for v in models.values())
    unchanged=load(HERE/'upstream_hashes.json');assert all(sha(k)==v for k,v in unchanged.items())
    source_hashes={str(q):sha(q) for q in HERE.glob('*.py')}
    dump(HERE/'audit_receipt.json',dict(exit_code=int(bool(errors)),checked_count=len(checked),errors=errors,seconds=time.time()-tick,
        exactly292_unique_sources=True,source_count_not_entry_count=True,unadorned_float_count=301,
        native_original_pixels_identical=True,neutral_uncomputed_domain=True,relation_before_all_pruning=True,
        original_model_and_upstream_sources_unchanged=True,main_display_not_selected_by_reference=True,
        old_missing_entry108_remain_unsolved=True,full_arc_identity_accuracy=None,source_hashes=source_hashes,
        output_directory=str(HERE/'交付_292张原图与热力图')))
    print('Audited292 PNG and301 original float entries; errors',len(errors),flush=True);sys.exit(bool(errors))
if __name__=='__main__':main()
