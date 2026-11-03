"""冻结输入源码与登记产物；仅写简短JSON和原有资产索引。"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json
from data_io import HERE,ROOT,sha,read_json,dump,sources

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','package']);ap.add_argument('--pilot',default='pilot_001');a=ap.parse_args()
    if a.mode=='freeze':
        p=HERE/a.pilot;s=read_json(p/'summary.json');ctx=read_json(p/'run_context.json');audit=read_json(p/'independent_audit.json')
        if s['completed']!=12 or s['failed'] or not s['source_unchanged'] or ctx['source_sha256']!=sources():raise ValueError('Pilot gate')
        if audit['passed']!=12 or audit['failed'] or audit.get('global_errors'):raise ValueError('Pilot audit gate')
        target=HERE/'freeze_all.json'
        if target.exists():raise FileExistsError(target)
        dump(target,{'version':'DR-C3.1.0','frozen_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sources(),
            'pilot':a.pilot,'pilot_summary_sha256':sha(p/'summary.json'),'purpose':'Three image stages on same400; no contact or incoming-rope task.'})
        return
    run=HERE/'all_001';s=read_json(run/'summary.json');audit=read_json(run/'independent_audit.json')
    if s['completed']!=400 or s['failed'] or sources()!=read_json(HERE/'freeze_all.json')['source_sha256']:raise ValueError('Full run gate')
    if audit['passed']!=400 or audit['failed'] or audit.get('global_errors'):raise ValueError('Audit gate')
    rows=[json.loads(line) for line in (run/'run_log.jsonl').read_text(encoding='utf-8').splitlines()]
    contrasts=[]
    for row in rows:
        v=row['roi_and_enhancement'].get('vertical_adjacent_difference',{})
        contrasts.append({'image':row['output_file'],**v})
    comparable=[c for c in contrasts if c.get('paired_pixel_count',0)>0]
    dump(run/'contrast_check.json',{'checked_images':len(contrasts),'comparable_images':len(comparable),
        'p90_contrast_increased_images':sum(c['enhanced_abs_diff_p50_p90'][1]>c['raw_abs_diff_p50_p90'][1] for c in comparable),
        'comparison_scope':'Same vertical neighbor pairs inside dark ROI, original gray6..249; contrast metric, not boundary accuracy.',
        'cases':contrasts})
    before=read_json(HERE/'index_before.json')
    for path,h in before.items():
        if sha(path)!=h:raise ValueError('Index changed: '+path)
    updates=[]
    for path,prefix in [(ROOT/'数据资产/README.md','05_派生与实验结果/'),(HERE.parent/'README.md',''),(ROOT/'工程导航/05_实验版本档案/README.md','../../数据资产/05_派生与实验结果/')]:
        link=prefix+HERE.name+'/all_001/图像结果'
        text=f'\n\n## 欠曝光卷面三步实验 DR-C3.1.0（2026-10-26）\n\n[三组图像目录]({link})：同一开放400图，分别输出欠曝光原色ROI、灰度对比度增强ROI和新C3黄点/整组曲线叠图，各400张（左右各200），外部统一0。暗ROI来自C4卷面内的低频曝光场；全暗ROI直接检测，不依赖活动绳检出或旧走廊。运行400/400、异常0，像素及文件契约独立审计400/400；观测非空{s["images_with_observations"]}图、曲线非空{s["images_with_curves"]}图，仅为产物计数，非真值准确率。原图与C3/C4基准保留，未接触设备或保留集；按用户要求无HTML。源码、协议、参数、来源哈希与逐图记录在同实验目录。\n'
        with path.open('a',encoding='utf-8') as f:f.write(text)
        updates.append({'path':str(path),'before':before[str(path)],'after':sha(path)})
    dump(HERE/'index_update_receipt.json',{'files':updates})
    files=[{'path':p.relative_to(run).as_posix(),'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(run.rglob('*')) if p.is_file() and p.name!='output_manifest.json']
    dump(run/'output_manifest.json',{'version':'DR-C3.1.0','files':files,'count':len(files)})
    dump(HERE/'release_manifest.json',{'root_files':[{'path':p.name,'sha256':sha(p)} for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='release_manifest.json'],
        'output_manifest_sha256':sha(run/'output_manifest.json'),'image_groups':3,'images_per_group':400,
        'equipment_access':False,'reserved_access':False})
    print(json.dumps({'images':1200,'files':len(files),'audit_passed':audit['passed']}))

if __name__=='__main__':main()
