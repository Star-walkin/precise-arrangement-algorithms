"""Fixed training-side diagnostics at the already-selected common update step."""
from pathlib import Path
import argparse,json,sys,time
import torch
from paired_train import evaluate_training,read_training
from train import load,load_model,dump,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True)
    p.add_argument('--paired',type=Path,required=True);p.add_argument('--old-checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    assert not args.out.exists()
    pair=load(args.paired);step=pair['main_comparison_step']
    assert sha(args.old_checkpoint)==pair['same_origin_checkpoint']
    selected={'old':args.old_checkpoint,**{v:Path(pair['main_checkpoints'][v]['path']) for v in ('A','B')}}
    cfg=load(selected['A'].parent.parent/'config.json');torch.set_num_threads(cfg['threads'])
    items,arrays,split=read_training(args.index);models={};start=time.perf_counter()
    for variant,checkpoint in selected.items():
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if variant in ('A','B'):
            assert cp['step']==step and sha(checkpoint)==pair['main_checkpoints'][variant]['sha256']
        net=load_model(checkpoint)
        models[variant]={'checkpoint':str(checkpoint.resolve()),'sha256':sha(checkpoint),'step':cp['step'],
                         'fit':evaluate_training(net,items,arrays,cfg)}
        del net,cp
        print('fixed_train_fit',variant,'completed',flush=True)
    result={'exit_code':0,'command':sys.argv,'source_sha256':sha(__file__),'index_sha256':sha(args.index),
            'paired_summary_sha256':sha(args.paired),'main_comparison_step':step,'models':models,
            'seconds':time.perf_counter()-start,'image_count':len(items),**split,
            'scope':'Fixed unaugmented training ROIs. Added neighbor/background masks are engineering weak supervision; fit is not independent identity accuracy.',
            'selection':'A/B selected only by maximum shared saved step. Old source checkpoint shown separately, not the formal continuation control.'}
    dump(args.out,result);print(json.dumps({'exit_code':0,'main_comparison_step':step,'seconds':result['seconds']}))


if __name__=='__main__':main()
