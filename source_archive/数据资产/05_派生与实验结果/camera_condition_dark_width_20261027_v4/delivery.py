"""只在全量审计与冻结后质性复查完成后登记交付，不修改预测。"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json
from data_io import HERE,ROOT,DATA,D1,read_json,dump,sha,sources

run=HERE/'all_001'
summary=read_json(run/'summary.json');audit=read_json(run/'audit.json')
assert summary['completed']==400 and summary['failed']==0 and summary['source_unchanged']
assert audit['passed']==400 and audit['failed']==0 and not audit['global_errors']
assert audit['limited_body_cases']==0
assert sources()==read_json(HERE/'freeze_all.json')['source_sha256']
reviews=[read_json(run/'记录'/f'qa_additional_review_{side}.json') for side in ('left','right')]
rows=[json.loads(v) for v in (run/'run_log.jsonl').read_text(encoding='utf8').splitlines()]
inventory={}
for stage in ('01_阴影二值化','02_去细股阴影','03_新黄点','04_边界结果'):
    inventory[stage]={role:len(list((run/'图像结果'/stage/role).glob('*.png'))) for role in ('left_camera','right_camera')}
    assert inventory[stage]=={'left_camera':200,'right_camera':200}
report={
 'version':'DR-D4.0.0','name':'沟宽筛选 D4','finished_utc':datetime.now(timezone.utc).isoformat(),
 'summary':summary,'output_image_counts':inventory,
 'source_groups':dict(Counter(r['source_group'] for r in rows)),
 'scope':'Same400 open development images. Offline dark-region boundary experiment. No claim of independent400conditions.',
 'input_manifest_sha256':sha(DATA/'manifest.csv'),
 'd1_fixed_output_manifest_sha256':sha(D1/'all_001/output_manifest.json'),
 'freeze_sha256':sha(HERE/'freeze_all.json'),
 'audit':{k:audit[k] for k in ('planned','passed','failed','limited_body_cases','images_without_accepted_curves','global_errors')},
 'audit_sha256':sha(run/'audit.json'),'output_manifest_sha256':sha(run/'output_manifest.json'),
 'synthetic_check':read_json(HERE/'selfcheck_records/final_algorithm_synthetic_001.json'),
 'pilots':[{'run':n,'summary':read_json(HERE/n/'summary.json'),
            'independent_visual_review':read_json(HERE/n/'记录/qa_independent_diagnosis.json')}
            for n in ('pilot_001','pilot_002','pilot_003')],
 'post_freeze_fixed_extra12_reviews':reviews,
 'method_changes':[
  'Fresh yellow candidates: coarse tangent averaging before adaptive threshold, width opening and raw-image bright flanks.',
  'Projected coarse/fine texture periodicity diagnostic and independently estimated broad bright-body barriers.',
  'New ordered local groups with pixel-path veto; old D2/D3 observations are not inputs.',
  'Local robust fits plus monotonic derivative projection, variable-curvature integral, all-group-point vertical alignment.',
  'Split and refit on body/unknown/long-evidence gaps; true raster ROI/unknown validation; no forced extrapolation.'
 ],
 'known_limitations':[
  'Fine-strand crossings can still mimic wide grooves; morphology cannot guarantee pure inter-rope shadows.',
  'Extreme dark block artifacts, side-edge foreshortening, partial shadow and competing phases remain uncertain.',
  'Rejection reduces misleading long curves but can also remove correct boundaries and reduce coverage.',
  'Current image-y convexity prior assumes this dataset camera orientation, not every future view.',
  'Image-supported smooth curves, periodicity and audit success do not establish physical material identity or contact.'
 ],
 'actual_boundary_accuracy':None,'baseline_replacement':False,'original_pixels_changed':False,
 'html_generated':False,'equipment_access':False,'reserved_access':False,
 'files_changed':'New D4 directory and appended three navigation indexes only; all20 baseline files verified unchanged.'}
if (HERE/'experiment_report.json').exists():raise FileExistsError('report already written')
dump(HERE/'experiment_report.json',report)
text=(
 '\n\n## 沟宽筛选 D4 / DR-D4.0.0（2026-10-27）\n\n'
 '[400张边界结果]({target}/all_001/图像结果/04_边界结果)及[实验记录]({target}/experiment_report.json)：'
 '同一开放400图，左右各200；D1欠曝光ROI/增强像素及C3/D2/D3原代码保持不变。'
 '按用户本轮授权重检黄点：粗切向滤波、动态阈值、宽度开运算，结合原图周期和宽亮绳体检查重新归类；'
 '变曲率凸性平滑后逐像素拒绝穿绳体/未知区路径，拆组重拟合，失败保留。'
 '三轮固定12图开发后冻结27项源码/依赖，400/400运行与独立结构审计通过；四个图像阶段各400张，ROI外0。'
 '目视固定样本细纹串珠/局部反弯减少，低信息区误连转拒识；侧缘漏段、碎片及邻沟歧义仍在，覆盖并非全面提升。'
 '冻结后额外12图另存逐图质性复查，记录改善、不足与覆盖变化，详见报告；点数、平滑性及结构审计不是真值准确率。'
 '本版供复核，不替换已认可基准；原失败轮次保留，无设备或保留集访问，不生成HTML。\n'
)
changes=[]
for p,target in [(ROOT/'数据资产/README.md','05_派生与实验结果/'+HERE.name),
                  (HERE.parent/'README.md',HERE.name),
                  (ROOT/'工程导航/05_实验版本档案/README.md','../../数据资产/05_派生与实验结果/'+HERE.name)]:
    before=sha(p)
    content=p.read_text(encoding='utf8')
    if '## 沟宽筛选 D4 / DR-D4.0.0' in content:raise ValueError('duplicate index entry')
    with p.open('a',encoding='utf8') as f:f.write(text.format(target=target))
    changes.append({'path':str(p),'before_sha256':before,'after_sha256':sha(p)})
dump(HERE/'delivery_receipt.json',{'version':'DR-D4.0.0','index_changes':changes,
 'experiment_report_sha256':sha(HERE/'experiment_report.json'),
 'new_source_files':[str(p) for p in sorted(HERE.glob('*.py'))],
 'baseline_unchanged':all(sha(p)==h for p,h in read_json(HERE/'protocol.json')['frozen_baseline_sha256'].items()),
 'image_count_per_stage':inventory,'equipment_access':False,'reserved_access':False})
print({'status':'delivered','images':inventory,'report':str(HERE/'experiment_report.json')})
