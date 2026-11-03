"""Numerical contracts, performance-repair equivalence and no-evidence refusal."""
from common import *
import numpy as np,sys,time
sys.path.append(str(ROOT/'vision_prototype/.venv/Lib/site-packages'))
import cv2,torch
from spatial import Context,sample
from search_r1 import PortableRanker,run
from train_relation import Ranker
from run_paths import load_gz
def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);tick=time.time()
    cp=torch.load(HERE/'model/final.pt',map_location='cpu',weights_only=False);net=Ranker(cp['config']['dimensions']);net.load_state_dict(cp['state']);net.eval()
    portable=PortableRanker();raw=np.load(HERE/'pairs.npz')['x'].reshape(-1,197)
    x=np.clip((raw-cp['mean'])/cp['std']*cp['active'],-12,12).astype(np.float32)
    with torch.inference_mode():expected=net(torch.from_numpy(x)).numpy()
    actual=portable(raw);error=float(abs(expected-actual).max());assert error<1e-4
    rng=np.random.default_rng(20261101);a=rng.normal(size=(60,80,8)).astype(np.float32);q=rng.uniform([1,1],[78,58],size=(100,2))
    old=np.c_[[sample(a[:,:,i],q) for i in range(8)]].T
    b=np.ascontiguousarray(np.moveaxis(a,2,0));new=np.c_[[sample(b[i],q) for i in range(8)]].T
    assert np.array_equal(old,new)
    compared=[]
    for oldpath in (HERE/'batch_001/R1/gallery').glob('*/*.json.gz'):
        newpath=HERE/'batch_002/R1/gallery'/oldpath.parent.name/oldpath.name
        oldr=load_gz(oldpath)['result'];newr=load_gz(newpath)['result']
        assert oldr['selected_candidate_id']==newr['selected_candidate_id']
        assert oldr['candidates']==newr['candidates']
        compared.append(str(newpath))
    assert compared,'No actual old/new repair replay comparison'
    flat=np.full((3000,4096),.35,np.float32);heat=np.zeros((975,2048),np.float32);feature=np.zeros((8,248,512),np.float16)
    ctx=Context(flat,heat,feature,[[3800,1800],[3680,1820]])
    refused=run(ctx,portable,edge=[],trace=True);assert refused['selected_candidate_id'] is None
    assert not refused['candidates'] or all(c['length_px']<=130 for c in refused['candidates'])
    # Same fixed828 pairs: all/no feature block interventions, no new training.
    manifest=load(HERE/'pair_manifest.json');pair=np.load(HERE/'pairs.npz')['x']
    initial=PortableRanker(HERE/'model/initial_portable.npz')
    ablations={}
    for name,lo,hi in [('trained',None,None),('A_spatial_feature_block_neutral',0,48),('native_profile_block_neutral',54,153),('geometry_block_neutral',168,189)]:
        xx=pair.copy()
        if lo is not None:xx[:,:,lo:hi]=cp['mean'][lo:hi]
        v=portable(xx.reshape(-1,197)).reshape(-1,2);ablations[name]=dict(pair_order_fraction=float((v[:,0]>v[:,1]).mean()),mean_margin=float((v[:,0]-v[:,1]).mean()))
    v=initial(pair.reshape(-1,197)).reshape(-1,2);ablations['untrained']=dict(pair_order_fraction=float((v[:,0]>v[:,1]).mean()))
    dump(HERE/'checks.json',dict(exit_code=0,portable_matches_torch_max_abs=error,strided_contiguous_track_query_exact=True,
        actual_partial_batch_repair_replays=len(compared),actual_repair_candidates_identical=True,
        flat_zero_heat_refused=True,flat_result=refused,model_A_sha256=sha(load(HERE/'protocol.json')['models']['A']['path']),
        fixed_pair_interventions=ablations,scope='numerical/data contracts and training-pair intervention only; no physical identity assertion',seconds=time.time()-tick))
    print('Checks passed. Replay pairs',len(compared),'portable error',error,'training ablations',ablations)
if __name__=='__main__':main()
