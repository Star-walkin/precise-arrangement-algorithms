"""同参数关闭图像软代价的机制对照；距离/能量不能充当真值准确率。"""
from pathlib import Path
import sys
import numpy as np
from data_io import HERE,selection,read_json,dump,sha

image_run,control_run=HERE/sys.argv[1],HERE/sys.argv[2]
target=HERE/sys.argv[3]
if target.exists():raise FileExistsError(target)
rows,_=selection('pilot')
records=[]
for row in rows:
    rel=Path(row['output_file']);part=Path('记录/cases')/rel.parent/(rel.stem+'.json')
    # output_file begins images/; cases only need its immediate camera directory.
    part=Path('记录/cases')/rel.parent.name/(rel.stem+'.json')
    a,b=read_json(image_run/part),read_json(control_run/part)
    assert a['observed_lines']==b['observed_lines']
    maps=[]
    for case in [a,b]:
        maps.append({(c['path_id'],c['side'],int(x)):int(y)
                     for c in case['fit']['curves'] for x,y in c['sampled_points_xy']})
    common=maps[0].keys()&maps[1].keys()
    delta=np.asarray([abs(maps[0][k]-maps[1][k]) for k in common],float)
    da=a['record']['path_diagnostics'];db=b['record']['path_diagnostics']
    records.append({'output_file':row['output_file'],
      'case_sha256':{'image':sha(image_run/part),'control':sha(control_run/part)},
      'yellow_count':a['record']['observed_point_count'],
      'common_path_columns':len(common),'image_only_columns':len(maps[0])-len(common),
      'control_only_columns':len(maps[1])-len(common),
      'changed_common_columns':int(np.count_nonzero(delta)),
      'displacement_mean_px':float(delta.mean()) if len(delta) else None,
      'displacement_p90_px':float(np.percentile(delta,90)) if len(delta) else None,
      'image_full_yellow_metric':da.get('yellow_full_denominator'),
      'control_full_yellow_metric':db.get('yellow_full_denominator'),
      'image_pseudo_count':sum(d.get('coherent_image_pseudo_count',0) for d in da.get('path_diagnostics',[])),
      'control_pseudo_count':sum(d.get('coherent_image_pseudo_count',0) for d in db.get('path_diagnostics',[]))})
dump(target,{'image_run':str(image_run),'control_run':str(control_run),'cases':records,
    'fixed_input_count':len(rows),'yellow_unchanged':True,
    'meaning':'Mechanism ablation only. Control disables image pseudo observations in fitting; both retain original image support gates. Common-path displacement is not truth accuracy; missing columns reported separately.',
    'ground_truth_accuracy':None})
print({'cases':len(records),'image_pseudo_count':sum(r['image_pseudo_count'] for r in records),
       'changed_common_columns':sum(r['changed_common_columns'] for r in records),
       'common_path_columns':sum(r['common_path_columns'] for r in records)})
