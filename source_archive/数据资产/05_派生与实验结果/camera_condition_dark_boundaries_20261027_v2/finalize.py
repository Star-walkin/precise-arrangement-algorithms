"""冻结已审核的算法，登记离线结果；不修改原数据或历史实验。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
from data_io import HERE, ROOT, sources, sha, read_json, dump

def freeze(pilot):
    target=HERE/'freeze_all.json'
    if target.exists():raise FileExistsError(target)
    context=read_json(HERE/pilot/'run_context.json')
    audit=read_json(HERE/pilot/'independent_audit.json')
    qa=HERE/('independent_visual_qa_'+pilot+'.json')
    if not qa.exists():qa=HERE/('root_visual_review_'+pilot+'.json')
    if audit['status']!='passed' or context['source_sha256']!=sources() or not qa.exists():
        raise ValueError('Need unchanged reviewed pilot, passing audit, and visual review')
    dump(target,{'version':read_json(HERE/'protocol.json')['version'],
        'variant':'anchored','selected_pilot':pilot,'source_sha256':sources(),
        'pilot_audit_sha256':sha(HERE/pilot/'independent_audit.json'),
        'visual_review_sha256':sha(qa),'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'decision':'Freeze for all400 offline review. Visible dark-groove coverage improved; residual-inconsistent fits remain orange review candidates. This is not ground-truth accuracy acceptance.',
        'legend':{'yellow':'original-image-checked dark valley observations',
                  'cyan':'smooth fit within self-consistency budget, not ground-truth verified',
                  'purple_dashed':'short inferred gap',
                  'orange_dashed':'fit disagrees with observation points; needs review'}})

def finish():
    protocol=read_json(HERE/'protocol.json');run_name=protocol.get('final_run','all_001')
    out=HERE/run_name;s=read_json(out/'summary.json');a=read_json(out/'independent_audit.json')
    if s['completed']!=400 or s['failed'] or a['status']!='passed' or a['passed']!=400:
        raise ValueError('400 predictions and independent audit required')
    repair=read_json(out/'repair_equivalence.json')
    additional=read_json(out/'记录/qa_additional/visual_review.json')
    if (HERE/'experiment_report.json').exists():raise FileExistsError('report exists')
    histories=[]
    for name in ['control_001','pilot_001','pilot_002','pilot_003','pilot_004','pilot_005','pilot_006','all_001']:
        p=HERE/name/'summary.json'
        if p.exists():histories.append({'run':name,'summary':read_json(p),'sha256':sha(p)})
    report={'version':s['version'],'name':'暗纹锚定 D2','date':'2026-10-27',
        'source':'same frozen open400, left200/right200; original PNG hashes in case records',
        'roi_and_enhancement':'D1 approved cached pixels reused unchanged and verified against frozen manifest',
        'baseline':'RB-C3.0.0 original observation/grouping/fitting sources unchanged',
        'method':'C3 bright original-point anchors; directional averaging; adaptive noise thresholds and original bilateral-valley recheck; ordered dark point classes; shared cubic B-spline displacement/spacing plus smooth per-row residual and ordered spacing projection.',
        'summary':s,'audit':a,'experiment_history':histories,
        'limits':['No manual ground-truth accuracy rate. Same development samples, not a blind test.',
                  'Orange fits exceed p90 0.25pitch residual to detected points; that is self-consistency, not truth accuracy.',
                  'Dark compression blocks, slight parallel offsets, missing bright anchors and very weak grooves remain failure modes.',
                  'Long unsupported gaps stay blank; no contact-point or material-identity claim.'],
        'input_protocol_sha256':sha(HERE/'protocol.json'),
        'output_manifest_sha256':sha(out/'output_manifest.json'),
        'freeze_sha256':sha(HERE/protocol.get('freeze_manifest','freeze_all.json')),'output_images':str(out/'图像结果'),
        'independent_visual_review':read_json(HERE/'independent_visual_qa_pilot_006.json'),
        'repair_equivalence':repair,'additional_fixed12_visual_review':additional,
        'audit_source_sha256':sha(HERE/'audit_dark.py'),
        'repair_check_source_sha256':sha(HERE/'compare_repair.py'),
        'empty_observation_images':400-s['images_with_observations'],
        'equipment_access':False,'reserved_access':False,'html_generated':False}
    dump(HERE/'experiment_report.json',report)
    legend=f'暗纹锚定 D2 / {s["version"]}\n左右各200张；ROI外统一为0。\n黄色十字：经原图暗谷检查的观测点。青线：与观测较一致的平滑拟合。紫虚线：短缺口推断。橙虚线：与观测点偏差较大的待复查拟合，不作为可靠边界。\n最暗块状区域和轻微平行偏位仍需复查。R_run_20260923_01__f001348__right缺乏C3亮区锚，仅保留ROI底图并拒识。无人工真值准确率，未接触设备或保留集。\n'
    (out/'图像结果'/'图例.txt').write_text(legend,encoding='utf-8-sig')
    tails={ROOT/'数据资产/README.md':'05_派生与实验结果/',
           HERE.parent/'README.md':'',
           ROOT/'工程导航/05_实验版本档案/README.md':'../../数据资产/05_派生与实验结果/'}
    changes=[]
    for path,prefix in tails.items():
        before=sha(path);old=path.read_text(encoding='utf-8')
        addition=f'\n\n## 暗纹锚定 D2 / {s["version"]}（2026-10-27）\n\n'
        addition+=f'[400张图像目录]({prefix}{HERE.name}/{run_name}/图像结果)：同一开放400图（左右各200），复用已认可的D1暗ROI/增强像素，C3原代码不改。C3亮区黄点限定行序，暗区原图检验沟槽；共享样条与有序间距约束拟合。400/400运行和独立结构审计通过；399张有边界，1张缺少亮区锚而保留ROI拒识。清晰暗带覆盖有目视改善，极暗块状区域及局部偏位仍存在。{s["uncertain_fit_curves"]}/{s["total_curves"]}条拟合超出相对观测点的残差门，橙虚线显式标待复查；这一比例不是真值错误率。旧失败轮次、空值错误运行、阈值对照、协议增补、源快照和逐图证据均保留，详见同目录experiment_report.json。未接触设备、保留集，不生成HTML，不替换C3基准。\n'
        if '## 暗纹锚定 D2 / '+s['version'] in old:raise ValueError('index already registered')
        with path.open('a',encoding='utf-8',newline='\n') as stream:stream.write(addition)
        changes.append({'path':str(path),'before_sha256':before,'after_sha256':sha(path)})
    dump(HERE/'index_changes.json',changes)
    print({'completed':400,'audit_passed':400,'output':str(out/'图像结果')})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','finish']);ap.add_argument('--pilot',default='pilot_006');args=ap.parse_args()
    freeze(args.pilot) if args.action=='freeze' else finish()
