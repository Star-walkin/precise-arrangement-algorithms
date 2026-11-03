"""Independent numeric audit, without invoking the inference or path code."""
from pathlib import Path
import hashlib
import json
import sys
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'batch_ab_20261030_001'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(2**20),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    dest=HERE/'numeric_response_audit.json'
    assert not dest.exists()
    start=time.perf_counter();torch.set_num_threads(8)
    checks=[];failures=[];sources=[]
    proto=load(HERE/'protocol_inputs.json')
    gallery=load(OLD/'receipt.json');oldjobs=load(OLD/'jobs.json')
    records={r['id']:r for r in gallery['records']}
    jobs={j['id']:j for j in oldjobs['jobs']}
    for source in proto['frozen_sources']:
        actual=sha(source['path'])
        ok=actual==source['sha256'];sources.append(source|{'current_sha256':actual,'matches':ok})
        if not ok:failures.append({'source_hash_changed':source['path']})
    assert len(proto['jobs'])==len(jobs)==292
    assert len(oldjobs['skipped'])==108
    for number,job in enumerate(proto['jobs'],1):
        try:
            old=jobs[job['id']]
            assert job['seeds']==old['seeds']
            assert sha(job['image_path'])==job['image_sha256']==old['image_sha256']
            directory=HERE/'responses'/job['id'];meta=load(directory/'inference.json')
            assert meta['id']==job['id'] and meta['image_sha256']==job['image_sha256']
            assert meta['model_sha256']==proto['model_B']['sha256']
            assert meta['source_sha256']==proto['inference_helper_sha256']
            assert meta['protocol_sha256']==sha(HERE/'protocol_inputs.json')
            assert not meta['entry_conditions_merged'] and not meta['target_labels_read']
            assert len(meta['entry_conditions'])==len(job['seeds'])==len(records[job['id']]['ranges'])
            assert len(list(directory.glob('entry_*.npz')))==len(job['seeds'])
            for k,entry in enumerate(meta['entry_conditions']):
                response_path=directory/f'entry_{k:02d}.npz'
                assert Path(entry['response_path']).resolve()==response_path.resolve()
                assert sha(response_path)==entry['response_sha256']
                assert entry['seed']==job['seeds'][k]
                with np.load(response_path,allow_pickle=False) as z:
                    assert z.files==['response']
                    response=z['response']
                assert response.shape==(975,2048) and response.dtype==np.float32
                assert np.isfinite(response).all() and 0<=response.min()<=response.max()<=1
                # Compare exactly the same coordinate transformation used to
                # generate the old gallery's numerical min/max receipt. This
                # never derives response values from JPEG colors.
                with torch.inference_mode():
                    restored=F.interpolate(torch.from_numpy(response)[None,None],size=(1950,4096),
                                           mode='bilinear',align_corners=False)[0,0]
                    actual=[float(restored.min()),float(restored.max())]
                expected=records[job['id']]['ranges'][k]['B']
                error=float(np.max(np.abs(np.array(actual)-expected)))
                assert error<=3e-7, (job['id'],k,actual,expected,error)
                checks.append({'id':job['id'],'entry_index':k,'shape_hw':list(response.shape),
                               'dtype':str(response.dtype),'native_range':[float(response.min()),float(response.max())],
                               'original_gallery_B_range':expected,'reproduced_original_size_range':actual,
                               'maximum_range_absolute_error':error,'response_sha256':entry['response_sha256']})
        except Exception:
            failures.append({'id':job['id'],'error':traceback.format_exc()})
        if number==1 or number%40==0:
            print(json.dumps({'audited_images':number,'total':len(proto['jobs']),'failures':len(failures)}),flush=True)
    actual_dirs={p.name for p in (HERE/'responses').iterdir() if p.is_dir()}
    if actual_dirs!=set(jobs):failures.append({'unexpected_or_missing_image_dirs':sorted(actual_dirs^set(jobs))})
    if len(checks)!=299:failures.append({'expected_entries':299,'checked_entries':len(checks)})
    code=1 if failures else 0
    report={'version':'IR-H1.0.0','scope':'numeric response reproduction and provenance only, not visual rope identity',
            'exit_code':code,'source_count':400,'eligible_images':292,'upstream_skipped':108,
            'checked_entry_conditions':len(checks),'model_B_sha256':proto['model_B']['sha256'],
            'maximum_gallery_B_range_error':max([r['maximum_range_absolute_error'] for r in checks],default=None),
            'all_finite_native_float32_975x2048':len(checks)==299,
            'range_comparison_note':'same torch bilinear restoration as original gallery receipt; no JPEG inversion',
            'range_agreement_is_not_full_array_equivalence':True,
            'sources':sources,'checks':checks,'failures':failures,'seconds':time.perf_counter()-start,
            'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'numpy':np.__version__,
            'command':sys.argv,'audit_source_sha256':sha(__file__),'target_labels_read':False}
    with dest.open('x',encoding='utf-8') as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'exit_code':code,'images':292,'checked_entry_conditions':len(checks),
                      'maximum_gallery_B_range_error':report['maximum_gallery_B_range_error'],'failures':len(failures)}),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
