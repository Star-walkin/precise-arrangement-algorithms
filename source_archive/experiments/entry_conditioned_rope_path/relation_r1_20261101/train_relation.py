"""Train only R1. One pairwise softplus ranking loss; no A parameter updates."""
from common import *
import numpy as np,torch,time,random,sys
from torch import nn
class Ranker(nn.Module):
    def __init__(self,n):
        super().__init__();self.layers=nn.Sequential(nn.Linear(n,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,x):return self.layers(x).squeeze(-1)
def portable(net,mean,std,active,path):
    v={k.replace('.','_'):t.detach().cpu().numpy() for k,t in net.state_dict().items()}
    np.savez_compressed(path,**v,mean=mean,std=std,active=active)
def main():
    sys.stdout.reconfigure(encoding='utf8');torch.set_num_threads(2);torch.manual_seed(20261101);np.random.seed(20261101);random.seed(20261101)
    dest=HERE/'model';dest.mkdir(exist_ok=False);manifest=load(HERE/'pair_manifest.json');rows=manifest['rows']
    z=np.load(HERE/'pairs.npz');raw=z['x'];weights=torch.from_numpy(z['weight'])
    mean=raw.reshape(-1,raw.shape[-1]).mean(0);ss=raw.reshape(-1,raw.shape[-1]).std(0);active=(ss>1e-5).astype(np.float32);std=np.maximum(ss,.05)
    xx=((raw-mean)/std*active).clip(-12,12).astype(np.float32);x=torch.from_numpy(xx)
    net=Ranker(x.shape[-1]);initial={k:v.clone() for k,v in net.state_dict().items()}
    cfg=dict(seed=20261101,dimensions=x.shape[-1],layers=[x.shape[-1],64,32,1],parameters=sum(p.numel() for p in net.parameters()),
        pair_count=len(rows),source_groups=sorted({r['group'] for r in rows}),
        loss='weighted softplus(0.25 - score_correct + score_wrong); no other loss',optimizer='AdamW lr.002 weight_decay.0001',
        smallcheck_steps=100,smallcheck_seconds=60,formal_steps=600,formal_seconds=120,selection='last completed fixed-budget update',
        pair_sampling='uniform source group then pair; correlated local pairs are not independent images',
        inactive_feature_indices=np.flatnonzero(active==0).tolist(),inactive_reason='constant training feature columns forced to zero in inference too; F3-specific columns not learned without exact F3 in training images',
        A_frozen=True,evaluation_references_read=False,independent_identity_truth=False)
    dump(dest/'config.json',cfg);portable(net,mean,std,active,dest/'initial_portable.npz')
    def loss(ids):
        scores=net(x[ids]);return (torch.nn.functional.softplus(.25-scores[:,0]+scores[:,1])*weights[ids]).sum()/weights[ids].sum()
    @torch.no_grad()
    def evaluate(ids):
        v=net(x[ids]);d=v[:,0]-v[:,1]
        return dict(pair_loss=float(loss(ids)),pair_order_fraction=float((d>0).float().mean()),mean_margin=float(d.mean()),count=len(ids))
    ids=[i for i,r in enumerate(rows) if r['id'] in ['dual_20260924_111500__left__f00240','dual_20260924_111143__right__f00130'] and (r['phase']>=2 or r['kind']=='Codex_historical_engineering_weak_neighbor')]
    ids=np.asarray(ids[::max(1,len(ids)//32)][:40],int);assert len(ids)>=8
    start=evaluate(ids);optimizer=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0001);tick=time.time();small=[]
    for step in range(1,101):
        optimizer.zero_grad();ls=loss(ids);ls.backward();optimizer.step();small.append(dict(step=step,loss=float(ls.detach()),seconds=time.time()-tick))
        if time.time()-tick>=60:break
    final=evaluate(ids);gate=bool(final['pair_order_fraction']>=.9 and final['pair_loss']<start['pair_loss']*.6)
    dump(dest/'smallcheck.json',dict(initial=start,final=final,passed=gate,steps=step,seconds=time.time()-tick,training_pair_indices=ids.tolist(),independent_copy_discarded=True,log=small))
    assert gate,'Small-pair fit gate failed; do not hide by expanding training'
    net.load_state_dict(initial);optimizer=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0001);torch.manual_seed(20261101)
    groups=sorted({r['group'] for r in rows});bygroup={g:np.asarray([i for i,r in enumerate(rows) if r['group']==g]) for g in groups};rng=np.random.default_rng(20261101)
    tick=time.time();records=[];allids=np.arange(len(rows));before=evaluate(allids)
    with (dest/'train_log.jsonl').open('x',encoding='utf8') as f:
        for step in range(1,601):
            batch=np.asarray([rng.choice(bygroup[rng.choice(groups)]) for _ in range(64)])
            optimizer.zero_grad();ls=loss(batch);ls.backward();optimizer.step()
            rec=dict(step=step,loss=float(ls.detach()),seconds=time.time()-tick);f.write(json.dumps(rec)+'\n');f.flush()
            if step==1 or step%100==0:print(rec,flush=True)
            if time.time()-tick>=120:break
    seconds=time.time()-tick;after=evaluate(allids)
    torch.save(dict(state=net.state_dict(),step=step,config=cfg,mean=mean,std=std,active=active),dest/'final.pt');portable(net,mean,std,active,dest/'final_portable.npz')
    pergroup={g:evaluate(bygroup[g]) for g in groups}
    weak=np.asarray([i for i,r in enumerate(rows) if r['kind']=='Codex_historical_engineering_weak_neighbor'])
    dump(dest/'training_receipt.json',dict(exit_code=0,command=sys.argv,device='cpu',torch=torch.__version__,smallcheck_passed=gate,
        before=before,after=after,per_source_group=pergroup,weak_neighbor_pairs=evaluate(weak),
        steps=step,seconds=seconds,formal_initial_state_reset=True,checkpoint_policy=cfg['selection'],
        input_pair_sha256=sha(HERE/'pairs.npz'),model_sha256=sha(dest/'final.pt'),portable_sha256=sha(dest/'final_portable.npz'),
        frozen_A_model=load(HERE/'protocol.json')['models']['A'],A_gradients_or_optimizer_created=False,
        scope='training pair order only; not independent full-route identity accuracy'))
    print('Completed',before,after,'seconds',seconds,flush=True)
if __name__=='__main__':main()
