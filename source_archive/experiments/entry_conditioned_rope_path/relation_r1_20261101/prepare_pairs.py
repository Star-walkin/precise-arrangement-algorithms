"""TRAIN ONLY pair construction. Reviewed centers and explicit low/weak negatives.

No evaluation reference or prediction is read. Missing generated positives are
logged and skipped, never inserted as successful AUTO candidates.
"""
from common import *
import cv2,numpy as np,time
from spatial import Context,sample,P
def resample(line,step=4.):
    p=np.asarray(line,float);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    ss=np.arange(0,s[-1]+.01,step);return np.c_[np.interp(ss,s,p[:,0]),np.interp(ss,s,p[:,1])],ss
def closest(p,line):return np.sqrt(((np.asarray(p)[:,None]-line[None])**2).sum(-1)).min(1)
def gray(path):return cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
def main():
    assert not (HERE/'pairs.npz').exists();cv2.setNumThreads(1)
    tick=time.time();ix=load(PAIR/'data/index.json');rows=[];positive=[];negative=[];weight=[];diagnostics=[];inputs={};requests=[]
    for item in ix['items']:
        if item['split']!='train':continue
        assert item['source_group'] in load(HERE/'protocol.json')['train_groups']
        source=PAIR.parent/'run_20261029_001/data/label_sources'/(item['id']+'.json');s=load(source)
        assert s['reviewed'] is True;inputs[str(source)]=sha(source)
        with np.load(item['cache']) as z:target=z['target'];valid=z['valid'];stage=z['stage']
        with np.load(HERE/'cache/fixed/CONDITIONAL'/(item['id']+'.npz')) as z:response=z['response'];feat=z['features']
        ctx=Context(gray(item['image_path']),response,feat,item['known_seed_xy'])
        source_lines=[]
        for line in s['trusted_center_segments_xy']:
            p,_=resample(line)
            if (p[-1]-p[0])@ctx.d<0:p=p[::-1]
            source_lines.append(p)
        counts=dict(states=0,correct_generated=0,pairs=0,weak_neighbor_pairs=0)
        for li,p in enumerate(source_lines):
            for idx in range(24,len(p)-18,6):
                history=p[[idx-18,idx-12,idx-6,idx]];future=p[idx+6]
                candidates=ctx.propose(history);counts['states']+=1
                if not candidates:continue
                q=np.asarray([v for v,src in candidates]);dist=np.linalg.norm(q-future,axis=1)
                pp=np.flatnonzero(dist<=4.)
                if not len(pp):
                    requests.append(dict(id=item['id'],kind='generated_pool_missing_correct_next',current=history[-1].tolist(),future=future.tolist(),candidate_xy=q.tolist(),source='reviewed training reference posthoc only'))
                    continue
                counts['correct_generated']+=1
                qq=(q-[0,1050])*.5;v=sample(valid.astype(np.float32),qq)>.99;t=sample(target,qq)
                dline=closest(q,p);nn=np.flatnonzero(v&(t<.1)&(dline>=8))
                if not len(nn):continue
                desc,ev=ctx.descriptor(history,q);pi=int(pp[np.argmin(dist[pp])]);nn=sorted(nn,key=lambda k:ev['heat'][k],reverse=True)[:2]
                for ni in nn:
                    positive.append(desc[pi]);negative.append(desc[ni]);weight.append(1.)
                    rows.append(dict(id=item['id'],group=item['source_group'],kind='reviewed_target_vs_explicit_valid_low_center',
                        positive_origin='historical reviewed partial center',negative_origin='existing quality-clipped valid low center target; not arbitrary unlabelled space',
                        current_xy=history[-1].tolist(),positive_xy=q[pi].tolist(),negative_xy=q[ni].tolist(),positive_generated=True,
                        phase=int(stage[int(np.clip(round(qq[pi,1]),0,974)),int(np.clip(round(qq[pi,0]),0,2047))]),line=li))
                    counts['pairs']+=1
        annotation=load(item['supervision_annotation']);neighbor=annotation.get('neighbor')
        if neighbor:
            # Historical engineering weak pairs kept as a separately tagged group.
            n=np.asarray(neighbor['points_xy'],float)
            for center in annotation['reused_positive_pair_segments_xy']:
                center,_=resample(center)
                if len(center)<4:continue
                allp=max(source_lines,key=lambda p:-(np.linalg.norm(p[:,None]-center[None],axis=2).min()))
                for pp in center[::max(1,len(center)//5)]:
                    k=int(np.argmin(np.linalg.norm(allp-pp,axis=1)))
                    if k<18:continue
                    history=allp[[k-18,k-12,k-6,k]]
                    forward=allp[min(k+6,len(allp)-1)]
                    ni=int(np.argmin(abs((n-pp)@ctx.d)));wrong=n[ni]
                    if np.linalg.norm(wrong-forward)<8:continue
                    q=np.stack([forward,wrong]);desc,ev=ctx.descriptor(history,q)
                    positive.append(desc[0]);negative.append(desc[1]);weight.append(.5)
                    rows.append(dict(id=item['id'],group=item['source_group'],kind='Codex_historical_engineering_weak_neighbor',
                        positive_origin='reviewed center; source-only pairing',negative_origin=neighbor['source'],
                        current_xy=history[-1].tolist(),positive_xy=forward.tolist(),negative_xy=wrong.tolist(),positive_generated=False,phase=3,
                        no_independent_human_identity_verification=True))
                    counts['weak_neighbor_pairs']+=1
        diagnostics.append(dict(id=item['id'],group=item['source_group'],counts=counts,optical_width=ctx.width,width_meta=ctx.width_meta))
        print(item['id'],counts,flush=True)
    assert len(positive)>40,'Insufficient legal training pairs'
    x=np.stack([positive,negative],axis=1).astype(np.float32);w=np.asarray(weight,np.float32)
    np.savez_compressed(HERE/'pairs.npz',x=x,weight=w)
    dump(HERE/'pair_manifest.json',dict(rows=rows,diagnostics=diagnostics,input_hashes=inputs,count=len(rows),dimensions=x.shape[-1],seconds=time.time()-tick,
        evaluation_references_read=False,unknown_as_negative=False,training_groups=sorted({r['group'] for r in rows}),
        independent_identity_truth=False,negative_semantics='center-location low targets plus separately tagged historical engineering neighbor weak labels'))
    dump(HERE/'minimal_review_requests.json',dict(training_only=True,missing_candidate_cases=requests,
        request='Confirm current target after bend and one competing neighbor in two priority training frames; existing four neighbor fragments are engineering weak, not independent human truth.',
        priority_ids=['dual_20260924_111500__left__f00240','dual_20260924_111143__right__f00130']))
    dump(HERE/'search_parameters.json',P)
    print('Prepared pairs',x.shape,'seconds',time.time()-tick)
if __name__=='__main__':main()
