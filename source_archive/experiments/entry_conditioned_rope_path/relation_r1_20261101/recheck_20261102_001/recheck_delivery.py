"""Fresh audit of the completed exact-attachment R1 delivery.

All reads remain on the frozen protocol paths. Writes are confined to this new
recheck directory. No detector, labels, model forward, training or device import.
"""
from pathlib import Path
import sys,json,time,hashlib,concurrent.futures,traceback
HERE=Path(__file__).resolve().parent
R1=HERE.parent
sys.path.insert(0,str(R1))
from audit_delivery import image

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    start=time.perf_counter();errors=[];checked=[];hashchecks=[]
    p=load(R1/'protocol.json');r=load(R1/'render_receipt.json')
    expected={j['id'] for j in p['jobs']}
    files=list((R1/'交付_292张原图与热力图').glob('*.png'))
    assert len(p['jobs'])==len(expected)==len(files)==292
    assert {f.stem for f in files}==expected
    assert len(r['rows'])==292 and {v['id'] for v in r['rows']}==expected
    assert sum(len(j['seeds']) for j in p['jobs'])==sum(v['total_entries'] for v in r['rows'])==301
    assert all(v['display_entry']==0 for v in r['rows'])
    assert r['exit_code']==0 and not r['errors']
    with concurrent.futures.ProcessPoolExecutor(4) as ex:
        pending={ex.submit(image,row):row['id'] for row in r['rows']}
        for future in concurrent.futures.as_completed(pending):
            try:checked.append(future.result())
            except Exception:errors.append(dict(id=pending[future],traceback=traceback.format_exc()))
            if (len(checked)+len(errors))%30==0 or len(checked)+len(errors)==292:
                print(json.dumps(dict(checked=len(checked),errors=len(errors),seconds=time.perf_counter()-start)),flush=True)
    # Pixel equality was checked against decoded sources above; now also bind
    # their unchanged bytes to the whitelist, without reading any other image.
    for j in p['jobs']:
        assert '20260925' not in j['image_path']
        assert j['dataset_use_status']=='unlabeled_open_development_no_new_test_split'
        actual=sha(j['image_path']);assert actual==j['image_sha256'],j['id']
        hashchecks.append(dict(kind='source',path=j['image_path'],sha256=actual))
    for kind,items in [('A_B_models',p['models']),('upstream',load(R1/'upstream_hashes.json')),('R1_sources',load(R1/'final_source_hashes.json'))]:
        for k,v in items.items():
            path,expected_sha=(v['path'],v['sha256']) if kind=='A_B_models' else (k,v)
            actual=sha(path);assert actual==expected_sha,path
            hashchecks.append(dict(kind=kind,path=path,sha256=actual))
    train=load(R1/'model/training_receipt.json')
    assert sha(R1/'model/final.pt')==train['model_sha256']
    assert sha(R1/'model/final_portable.npz')==train['portable_sha256']
    # Cache portability check includes all existing available fixed and gallery
    # entries. The three automatic-entry failures have no fabricated npz.
    cj=load(R1/'cache_jobs.json');available=0;missing=0
    for job in cj:
        dest=Path(job['cache_path']);meta=load(dest.with_suffix('.json'))
        assert meta['job']==job
        if meta['available']:
            actual=sha(dest);assert actual==meta['sha256'],str(dest)
            available+=1
        else:missing+=1
    floats=[f for row in r['rows'] for f in row['float_responses']]
    assert len(floats)==301 and len({f['path'] for f in floats})==301
    receipt=dict(exit_code=int(bool(errors)),seconds=time.perf_counter()-start,checked_count=len(checked),errors=errors,
        exact_request_path=p['task_request_path'],exactly292_unique_png=True,total_conditions=301,
        dimensions=[8364,3350],left_original_dimensions=[4096,3000],left_native_pixels_identical=True,
        uncomputed_gray_is_unmodified=True,all301_floats_hashes_shapes_dtypes_ranges_passed=True,
        all301_path_coordinates_and_prepruning_relation_counts_passed=True,source_model_and_upstream_hashes_passed=True,
        available_A750_caches_verified=available,unavailable_auto_fixed_entries=missing,
        cache_response_shape=[975,2048],cache_feature_shape=[8,248,512],
        checker_source=str(R1/'audit_delivery.py'),checker_sha256=sha(R1/'audit_delivery.py'),
        new_checker_sha256=sha(__file__),checked_rows=checked,hash_checks=hashchecks,
        delivery_directory=str(R1/'交付_292张原图与热力图'),
        no_new_training=True,no_model_forward=True,no_reserved_access=True,no_labels_read=True,no_devices=True,
        full_target_identity_verified=False,full_arc_accuracy=None,
        limitation='Fresh delivery and data integrity audit only; old R1 identity failures retained and not reclassified.')
    target=HERE/'recheck_receipt.json';assert not target.exists()
    target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ['exit_code','seconds','checked_count','total_conditions','available_A750_caches_verified','unavailable_auto_fixed_entries','delivery_directory']},ensure_ascii=False),flush=True)
    sys.exit(bool(errors))
if __name__=='__main__':main()
