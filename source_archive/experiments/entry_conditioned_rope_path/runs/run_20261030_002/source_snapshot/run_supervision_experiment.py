"""One-command bounded supervision A/B, reusing both installed environments.

Use a NEW run directory. --resume skips completed gates but never overwrites
partial training or prediction files. Each subprocess command/exit is recorded.
"""
from __future__ import annotations
import argparse,json,shutil,subprocess,sys,time,hashlib
from pathlib import Path

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[1]
OLD=CODE/'runs/run_20261029_001'
VISION=ROOT/'vision_prototype/.venv/Scripts/python.exe'
TORCH=ROOT/'vision_training/.venv/Scripts/python.exe'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def run_command(run,python,script,args,gate,resume):
    if gate.exists():
        if not resume:raise FileExistsError(gate)
        print('completed gate retained:',gate,flush=True);return
    cmd=[str(python),'-B',str(CODE/script),*map(str,args)]
    start=time.time();print('RUN',subprocess.list2cmdline(cmd),flush=True)
    proc=subprocess.run(cmd,cwd=ROOT)
    row={'command':cmd,'start_unix':start,'seconds':time.time()-start,'exit_code':proc.returncode,'gate':str(gate)}
    with (run/'execution_commands.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    if proc.returncode:raise RuntimeError(f'{script} exited {proc.returncode}')
    if not gate.exists():raise RuntimeError(f'Missing completion gate: {gate}')

def branch_layout(run,key):
    branch=run/'branches'/key;data=branch/'data';data.mkdir(parents=True,exist_ok=True)
    for name in ['scenes','auto_entries','entry_hints']:
        target=data/name
        if not target.exists():shutil.copytree(OLD/'data'/name,target)
    for source,target in [(run/'data/index.json',data/'index.json'),(OLD/'path_config.json',branch/'path_config.json')]:
        if target.exists():assert digest(source)==digest(target),(source,target)
        else:shutil.copy2(source,target)
    return branch

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage',choices=['all','prepare','train','infer','evaluate'],default='all');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run=a.run.resolve();run.mkdir(parents=True,exist_ok=True)
    assert run!=OLD and run.parent==CODE/'runs','Use an independent experiment run folder'
    call=lambda py,s,args,gate:run_command(run,py,s,args,run/gate,a.resume)
    if a.stage in ['all','prepare']:
        call(VISION,'supervision_data.py',['--run',run],'data/index.json')
        call(VISION,'verify_supervision.py',['--run',run],'supervision/validation.json')
        call(VISION,'supervision_qa.py',['--run',run],'supervision_qa/summary.json')
        call(VISION,'freeze_evaluation_scope.py',['--run',run,'--old',OLD],'evaluation_scope/scope.json')
        call(VISION,'input_audit.py',['--run',run],'input_audit/audit.json')
        call(VISION,'input_audit.py',['--run',run,'--after-data'],'audit_with_supervision/audit.json')
        call(TORCH,'paired_train.py',['check','--out',run/'paired_numerical_checks_frozen.json'],'paired_numerical_checks_frozen.json')
        protocol=run/'protocol_parameters.json'
        if not protocol.exists():
            dump(protocol,{'frozen_unix':time.time(),'parameters':read(run/'paired_numerical_checks_frozen.json')['config'],
                'source_checkpoint_sha256':digest(OLD/'model/final.pt'),'evaluation_scope_sha256':digest(run/'evaluation_scope/scope.json'),
                'data_index_sha256':digest(run/'data/index.json'),'selection':'largest common saved update; no evaluation-based tuning',
                'supervision':'original center + engineering weak neighbor/background negatives; no new positive centers',
                'device_connections':False,'reserved_sets':False})
    if a.stage in ['all','train']:
        assert (run/'data/index.json').exists() and (run/'evaluation_scope/scope.json').exists()
        common=['--index',run/'data/index.json','--checkpoint',OLD/'model/final.pt']
        call(TORCH,'paired_train.py',['smallcheck',*common,'--out',run/'smallcheck'],'smallcheck/result.json')
        small=read(run/'smallcheck/result.json')
        before=small['initial']['macro_B_total_loss'];after=small['final']['macro_B_total_loss']
        assert after<before and all(small['present_negative_groups'].values()),'Smallcheck failed; inspect without increasing budget'
        for key in ['A','B']:
            branch=run/'branches'/key;branch.mkdir(parents=True,exist_ok=True)
            call(TORCH,'paired_train.py',['train',*common,'--variant',key,'--out',branch/'model'],f'branches/{key}/model/result.json')
        call(TORCH,'paired_train.py',['compare','--a',run/'branches/A/model','--b',run/'branches/B/model','--out',run/'paired_comparison.json'],'paired_comparison.json')
        call(TORCH,'common_training_fit.py',['--index',run/'data/index.json','--paired',run/'paired_comparison.json','--old-checkpoint',OLD/'model/final.pt','--out',run/'common_training_fit.json'],'common_training_fit.json')
        call(VISION,'plot_paired_training.py',['--run',run],'training_diagnostics/training_summary.json')
    if a.stage in ['all','infer']:
        paired=read(run/'paired_comparison.json')
        for key in ['A','B']:
            branch=branch_layout(run,key);checkpoint=Path(paired['main_checkpoints'][key]['path'])
            assert digest(checkpoint)==paired['main_checkpoints'][key]['sha256']
            call(TORCH,'infer.py',['--run',branch,'--model',checkpoint,'--phase','final','--threads','8'],f'branches/{key}/inference_final_summary.json')
            call(VISION,'verify_inference.py',['--run',branch,'--phase','final'],f'branches/{key}/inference_final_validation.json')
            call(VISION,'connect.py',['--run',branch],f'branches/{key}/connection_receipt.json')
    if a.stage in ['all','evaluate']:
        for key in ['A','B']:
            call(VISION,'evaluate.py',['--run',run/'branches'/key],f'branches/{key}/evaluation/summary.json')
        call(VISION,'compare_evaluate.py',['--run',run],'comparison/evaluation.json')
        call(VISION,'compare_visualize.py',['--run',run],'comparison/visualization_receipt.json')
        call(VISION,'trace_connection.py',['--run',run],'connection_trace/summary.json')
    print('stage complete',a.stage,str(run),flush=True)

if __name__=='__main__':main()
