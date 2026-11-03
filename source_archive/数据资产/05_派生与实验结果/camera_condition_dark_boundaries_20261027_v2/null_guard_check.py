"""只核查真实空锚失败输入与一个非空回归输入，不调整算法。"""
from pathlib import Path
import numpy as np
from data_io import HERE,C3,selection,inputs,read_json,read_image,dump,sources,sha
from anchored_dark import detect_anchored
from run import small

rows,_=selection('all')
checks=[]
for name,empty in [('images/right_camera/R_run_20260923_01__f001348__right.png',True),
                   ('images/right_camera/ruler_right_05.png',False)]:
    row=next(r for r in rows if r['output_file']==name)
    im,roi,en,_,pitch,_,deps=inputs(row);p=Path(name)
    c3=read_json(C3/'boundaries/records'/p.parent.name/(p.stem+'.json'))
    result=detect_anchored(im,en,roi,pitch,c3)
    if empty:
        assert c3['fit']['model'] is None
        assert not result['observed_lines'] and not result['fit']['curves']
        assert not np.any(result['observed_mask']) and not np.any(result['fit']['smooth_mask'])
    else:
        previous=read_json(HERE/'pilot_006/记录/cases'/p.parent.name/(p.stem+'.json'))
        assert small(result['observed_lines'])==previous['observed_lines']
        assert small(result['fit'])==previous['fit']
        for key,value in [('observed',result['observed_mask']),('smooth',result['fit']['smooth_mask']),('inferred',result['inferred_mask'])]:
            assert np.array_equal(value,read_image(HERE/'pilot_006/记录/masks'/key/p.parent.name/p.name,0))
    checks.append({'image':name,'expected_empty_anchors':empty,'passed':True,'input_dependency_sha256':deps})
dump(HERE/'null_guard_check.json',{'passed':2,'failed':0,'checks':checks,'source_sha256':sources()})
dump(HERE/'freeze_all_null_guard.json',{'version':'DR-D2.1.1','variant':'anchored',
    'source_sha256':sources(),'prior_freeze_sha256':sha(HERE/'freeze_all.json'),
    'guard_check_sha256':sha(HERE/'null_guard_check.json'),
    'independent_pilot_review_sha256':sha(HERE/'independent_visual_qa_pilot_006.json'),
    'scope':'One null-model handling guard only. all_001 failure retained; all_002 full rerun with399 prediction equivalence verification.'})
print('null guard rejection and nonempty prediction equivalence: 2/2 passed')
