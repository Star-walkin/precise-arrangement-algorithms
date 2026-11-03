"""Register a completed offline run and preserve source/provenance snapshots."""
from pathlib import Path
import argparse, hashlib, json, shutil, sys
from datetime import datetime

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main(run):
    run=Path(run).resolve();src=Path(__file__).resolve().parent;root=src.parents[1]
    for name in ['model/final.pt','connection_receipt.json','REPORT.md','inference_final_validation.json']:
        assert (run/name).is_file(),name
    snapshots=run/'source_snapshot';snapshots.mkdir(exist_ok=False)
    source=[]
    for p in sorted(src.glob('*.py')):
        dest=snapshots/p.name;shutil.copy2(p,dest);source.append({'path':str(p),'sha256':sha(p),'snapshot':str(dest)})
    for name in ['source_audit.json','evaluation_contract_checks_v3.json']:
        if (src/name).exists():shutil.copy2(src/name,run/name)
    changes=[];nav=run/'navigation_before';nav.mkdir(exist_ok=False)
    marker='## 2026-10-30：入口条件化目标弧段 U-Net 离线实验'
    note='''

## 2026-10-30：入口条件化目标弧段 U-Net 离线实验

[实际运行报告](../experiments/entry_conditioned_rope_path/runs/run_20261029_001/REPORT.md) · [代码与复跑入口](../experiments/entry_conditioned_rope_path/README.md)。历史开放开发10训练图/4来源组、8评价图/3来源组；CPU实际训练843步/1200秒，另完成120步小样本检查。输入保留原灰度自由段、过渡及远端，入口条件传至全局；单中心响应头与原图多候选连接，26份已知/自动入口结果及浮点图、路径、固定局部图保存。评价仅使用质量裁切的部分中心参考，完整目标末端身份仍未验收；空间“后段”标签不等于物理接触。AUTO失败和歧义保留，不以边缘几何到达当成功，不替换F3/A6或连接控制。S1未训练规划与历史来源用途的冲突单列，不能声称其独立评价有效。未访问保留集、未连接设备，原始素材和历史输出不改。
'''
    for rel in ['数据资产/README.md','vision_geometry_lab/README.md']:
        p=root/rel;old=p.read_text(encoding='utf-8');assert marker not in old
        dest=nav/(p.parent.name+'_README.md');shutil.copy2(p,dest)
        before=sha(p);p.write_text(old.rstrip()+note,encoding='utf-8')
        changes.append({'path':str(p),'before_sha256':before,'after_sha256':sha(p),'before_copy':str(dest),'change':'append new evidence link, preserve historical text'})
    dump(run/'navigation_changes.json',changes)
    dump(run/'source_snapshot_manifest.json',source)
    files=[]
    # No junction following: this is a freshly created regular experiment run.
    for p in sorted(run.rglob('*')):
        if p.is_file() and not p.is_symlink():files.append({'path':str(p.relative_to(run)),'bytes':p.stat().st_size,'sha256':sha(p)})
    dump(run/'delivery_manifest.json',{'files':files,'file_count':len(files),'bytes':sum(x['bytes'] for x in files),
        'scope':'run files before delivery_manifest/execution_receipt; original data referenced by hashed manifests, not recopied'})
    dump(run/'execution_receipt.json',{'status':'completed','exit_code':0,'command':sys.argv,'local_time':datetime.now().isoformat(),
        'training':'843 steps,1200 seconds CPU; see training_receipt.json','devices_connected':False,'reserved_data_accessed':False,
        'original_data_modified':False,'historical_experiments_modified':False,'navigation_changes':changes,
        'source_hashes':source,'delivery_file_count':len(files),'delivery_bytes':sum(x['bytes'] for x in files),
        'scope':'offline experiment completed; complete target arc identity not accepted'})
    print(json.dumps({'exit_code':0,'files':len(files),'bytes':sum(x['bytes'] for x in files)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(p.parse_args().run)
