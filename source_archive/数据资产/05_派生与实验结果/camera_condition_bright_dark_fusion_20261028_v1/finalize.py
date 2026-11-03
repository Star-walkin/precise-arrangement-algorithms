"""仅在400图独立核验通过后写交付说明、来源记录和导航更新。"""
from datetime import datetime,timezone
from collections import Counter
import json
from fusion_io import HERE,ROOT,C3,D4,read_json,dump,sha,sources,baseline_unchanged

run=HERE/'all_001';summary=read_json(run/'summary.json');audit=read_json(run/'audit.json')
assert summary['completed']==400 and summary['failed']==0 and summary['source_unchanged']
assert audit['passed']==400 and audit['failed']==0 and not audit['global_errors']
assert sources()==read_json(HERE/'freeze_all.json')['source_sha256'] and baseline_unchanged()
baseline=ROOT/'工程导航/05_实验版本档案/卷绳边界_欠曝光基准_D4_20261028'
receipt=read_json(baseline/'verification_receipt.json')
assert all(c['exit_code']==0 and c['result']['failed']==0 for c in receipt['checks'])
logs=[json.loads(x) for x in (run/'run_log.jsonl').read_text(encoding='utf8').splitlines()]
assert all(sum(1 for r in logs if r['camera_role']==c)==200 for c in ('left','right'))
pilot_match=[]
for rel in read_json(HERE/'pilot_001/run_context.json')['selection']:
    from pathlib import Path
    p=Path(rel);part=Path('图像结果')/p.parent.name/p.name
    pilot_match.append({'output_file':rel,'identical':sha(HERE/'pilot_001'/part)==sha(run/part)})
assert all(c['identical'] for c in pilot_match)
report={'version':'RB-F1.0.0','name':'亮暗融合 F1','date':'2026-10-28',
 'created_utc':datetime.now(timezone.utc).isoformat(),'summary':summary,
 'scope':'Existing open400 C3 bright and D4 dark saved results, same coordinates. Not a new detector or new boundary accuracy evaluation.',
 'baselines':{'bright':{'id':'RB-C3.0.0','run':str(C3)},
              'dark':{'id':'DR-D4.0.0','run':str(D4),'archive':str(baseline),
                      'manifest_sha256':sha(baseline/'baseline_manifest.json'),
                      'confirmation':'2026-10-28 user accepts D4 as dark-region development baseline; qualitative confirmation only.'}},
 'formal_rules':read_json(HERE/'protocol.json'),
 'audit':{k:audit[k] for k in ('planned','passed','failed','global_errors','exit_code','auditor_sha256')},
 'audit_sha256':sha(run/'audit.json'),'output_manifest_sha256':sha(run/'output_manifest.json'),
 'freeze_sha256':sha(HERE/'freeze_all.json'),'pilot_full_image_consistency':pilot_match,
 'visual_review':read_json(HERE/'pilot_001/记录/qa_visual/visual_diagnosis.json'),
 'by_source_group':dict(Counter(r['source_group'] for r in logs)),
 'roi_overlap_images':sum(r['input_roi_overlap_pixels']>0 for r in logs),
 'c3_clipped_boundary_pixels':sum(r['c3_boundary_clipped_pixels'] for r in logs),
 'd4_clipped_boundary_pixels':sum(r['d4_boundary_clipped_pixels'] for r in logs),
 'known_limits':['C3/D4 seam gaps and phase/endpoint disagreement remain; no newly connected same-rope identity.',
   'Existing SAM-union surface is reused. ruler_left_05 visibly misses physical surface above incoming rope; F1 does not expand it.',
   'Faint cyan/yellow on bright glare needs full-resolution inspection; dark enhancement is display only.',
   'All400 integrity success does not mean every physical boundary is detected or correct.'],
 'equipment_access':False,'reserved_access':False,'html_generated':False,'accuracy':None}
if (HERE/'experiment_report.json').exists():raise FileExistsError('report already exists')
dump(HERE/'experiment_report.json',report)
note='''亮暗融合 F1（RB-F1.0.0） / 2026-10-28

图像入口：all_001/图像结果，left_camera与right_camera各200张。
原尺寸4096×3000，不裁切、不缩放；既有卷面ROI外统一0。曝光良好区使用整组圆锥C3（RB-C3.0.0），欠曝光区使用沟宽筛选D4（DR-D4.0.0）。D4已按用户本次确认登记为欠曝光基准，归档在工程导航/05_实验版本档案/卷绳边界_欠曝光基准_D4_20261028。

正式边界来自两个版本已经保存的掩码，不从彩色QA截图取线，不重新拟合或移动边界。亮区归属=完整卷面与C3亮ROI相交；暗区归属=完整卷面与D1暗ROI相交，并排除亮ROI。若重叠，亮区由C3负责。两者均未覆盖的像素显示底图，不补线。
完整卷面采用既有C4 surface_masks（SAM2原mask与已接受亮ROI的并集），暗区显示采用已认可D1增强像素，其他卷面保留原色。青色是正式拟合线中非推断的部分；紫色虚线是原算法已有推断段；黄色十字来自已接受曲线的黄点，隔点显示减少遮挡。C3/D4对“有图像支持”的定义不同，颜色统一不代表其置信度相同。

记录/masks提供surface、亮暗归属、未覆盖带、C3边界、D4边界、融合边界、推断和接受黄点共9个机器可读0/255掩码。每图JSON保存来源、输入哈希、像素数与带来源前缀的曲线ID。未把亮暗端点就近连接或赋予跨区同根身份。

固定12预览后冻结4项实现，400/400运行及独立像素核验通过，左右各200；全部主图和正式掩码由独立脚本从旧源重新计算后逐像素一致，未新增边界像素。固定12张的pilot/all输出逐字节一致。此核验只证明融合正确，不是识别准确率。

继承限制：亮暗接缝仍有断口、极暗块仍漏线；ruler_left_05的旧SAM掩码漏掉活动绳上方可见卷面，本版原样保留。因此“全卷面融合图”指既有卷面ROI内展示两算法结果，不代表真实物理卷面全部覆盖或每条边界已完整连接。

复现时使用原vision_prototype/.venv环境，在本目录执行：
python -B run_fusion.py --mode all --output <本目录下新的运行绝对路径> --workers 6
python -B audit_fusion.py --run <新运行目录名> --workers 4
禁止覆盖既有运行目录。源码/参数/原输入清单、环境、日志、退出码与快照均有记录。
本次新增F1实现及D4档案、更新导航与基准说明；C3/D4原算法、原图和旧输出保持不变。无HTML、保留集访问或设备连接。
'''
(HERE/'结果说明.txt').write_text(note,encoding='utf8')
entry=('\n\n## 2026-10-28：D4欠曝光基准确认与亮暗融合F1\n\n'
 '[欠曝光基准D4]({base})已按用户本次质性认可封存，版本保持DR-D4.0.0；含27项冻结源码及必要证据，'
 '独立核验7765个唯一文件通过，未重复复制全量图像。曝光良好区域继续以RB-C3.0.0为基准，确认不等于逐根真值准确率。\n\n'
 '[亮暗融合F1（RB-F1.0.0）的400张图像]({f1}/all_001/图像结果)在原4096×3000坐标合并C3亮区与D4暗区正式掩码，'
 '左右各200；卷面外0，暗部沿用D1增强。400/400运行及独立像素审计通过，固定12图pilot/full逐字节一致；'
 '不重拟合、不新增跨区连接、曲线来源分别保留。亮暗接缝缺口和旧SAM漏分（ruler_left_05尤明显）仍保留，'
 '不称物理卷面全覆盖或同根接续已确认。详见[结果说明]({f1}/结果说明.txt)与[实验记录]({f1}/experiment_report.json)。无设备/保留集访问，无HTML。\n')
changes=[]
for path,b,f in [(ROOT/'数据资产/README.md','../工程导航/05_实验版本档案/卷绳边界_欠曝光基准_D4_20261028/README.md','05_派生与实验结果/'+HERE.name),
                 (HERE.parent/'README.md','../../工程导航/05_实验版本档案/卷绳边界_欠曝光基准_D4_20261028/README.md',HERE.name),
                 (ROOT/'工程导航/05_实验版本档案/README.md','卷绳边界_欠曝光基准_D4_20261028/README.md','../../数据资产/05_派生与实验结果/'+HERE.name)]:
    if '## 2026-10-28：D4欠曝光基准确认与亮暗融合F1' in path.read_text(encoding='utf8'):raise ValueError('duplicate index')
    before=sha(path)
    with path.open('a',encoding='utf8') as stream:stream.write(entry.format(base=b,f1=f))
    changes.append({'path':str(path),'before_sha256':before,'after_sha256':sha(path)})
rules=ROOT/'AGENTS.md';before=sha(rules);content=rules.read_text(encoding='utf8')
needle='以下是已有报告的事实边界，不是整个项目完成百分比；后续结论必须注明实际运行与证据来源。'
assert content.count(needle)==1
addition='\n\n离线卷面边界分区基准（2026-10-28）：曝光良好区采用[整链基准C3 / RB-C3.0.0](工程导航/05_实验版本档案/卷绳边界_整链基准_C3_20261026/README.md)，欠曝光区采用[欠曝光基准D4 / DR-D4.0.0](工程导航/05_实验版本档案/卷绳边界_欠曝光基准_D4_20261028/README.md)。两者均有用户对400图展示的质性认可；后续算法比较保留这两个原版本。亮暗结果合并另记为F1，不把融合、平滑或用户认可当作逐根真值、同根接续或t0/t1/t2验收。'
assert '离线卷面边界分区基准（2026-10-28）' not in content
content=content.replace(needle,needle+addition).replace('维护日期：2026-10-09。','维护日期：2026-10-28。',1)
rules.write_text(content,encoding='utf8')
changes.append({'path':str(rules),'before_sha256':before,'after_sha256':sha(rules)})
dump(HERE/'delivery_receipt.json',{'version':'RB-F1.0.0','changed_navigation':changes,
 'baseline_archive':str(baseline),'baseline_manifest_sha256':sha(baseline/'baseline_manifest.json'),
 'fusion_report_sha256':sha(HERE/'experiment_report.json'),'baseline_sources_unchanged':baseline_unchanged(),
 'new_implementation_files':['fusion_io.py','fusion.py','run_fusion.py','audit_fusion.py'],
 'images':400,'by_camera':{'left':200,'right':200},'equipment_access':False,'reserved_access':False})
print({'delivered':True,'images':400,'audit_passed':400,'old_sources_unchanged':baseline_unchanged()})
