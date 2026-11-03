"""Reproducible alignment checks and image-only scene overlays (no eval targets)."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2
import numpy as np

def main(run):
    p=Path(run).resolve()/'data';idx=json.loads((p/'index.json').read_text(encoding='utf-8'));checks=[]
    sources={s['sample_id']:s for s in json.loads(Path(idx['manifest']).read_text(encoding='utf-8'))['samples']}
    for s in idx['items']:
        z=np.load(s['cache']);seed=np.array(s['known_seed_xy']);length=np.linalg.norm(np.diff(seed,axis=0),axis=1).sum()
        assert length<=120.000001
        assert hashlib.sha256(Path(s['cache']).read_bytes()).hexdigest()==s['cache_sha256']
        # Byte-level verification binds entry hints to the frozen source without
        # parsing any evaluation centerline or rope-side geometry.
        source=sources[s['id']]
        assert hashlib.sha256(Path(source['annotation_path']).read_bytes()).hexdigest()==source['annotation_sha256']
        scene=json.loads(Path(s['scene_path']).read_text(encoding='utf-8'))
        shapes={k:list(z[k].shape) for k in z.files};assert len(set(tuple(x) for x in shapes.values()))==1
        assert np.isfinite(z['image']).all() and 0<=z['image'].min()<=z['image'].max()<=1
        assert not np.any(z['valid']&~z['roi']) and z['seed'].sum()>0
        if s['split']=='eval':assert not z['target'].any() and not z['valid'].any()
        checks.append(dict(id=s['id'],cache_hash_pass=True,annotation_byte_hash_pass=True,aligned=True,seed_length_original_px=float(length),
          conditional_contact_status=scene['contact_proxy']['status'],
          eval_target_all_zero=bool(not z['target'].any()) if s['split']=='eval' else None,
          eval_valid_all_zero=bool(not z['valid'].any()) if s['split']=='eval' else None))
        im=cv2.cvtColor((z['image']*255).astype('uint8'),cv2.COLOR_GRAY2BGR)
        def xy(points):return np.rint((np.array(points).reshape(-1,2)-np.array(s['bbox'][:2]))*s['scale']).astype('int32')
        for seg in scene['edge_segments_xy']:cv2.polylines(im,[xy(seg)],False,(255,100,0),2)
        for q in xy(scene['target_edge_points_xy']):cv2.circle(im,tuple(q),2,(255,0,255),-1)
        cv2.polylines(im,[xy(seed)],False,(0,255,255),4)
        c=scene['contact_proxy']['xy']
        if c is not None:cv2.circle(im,tuple(xy([c])[0]),12,(0,0,255),2)
        cv2.putText(im,'blue: observed interface | magenta: distal edge candidate | red: contact visual proxy',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(im,'yellow: entry hint only; no target identity or physical contact confirmed',(15,60),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
        cv2.imencode('.png',im)[1].tofile(str(p/'checks'/f'{s["id"]}__scene.png'))
    (p/'alignment_validation.json').write_text(json.dumps(dict(exit_code=0,count=len(checks),items=checks,
      command=sys.argv,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      scope='alignment, frozen hashes, 120px seed cap, eval zero-supervision; not semantic path accuracy'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('18 caches verified; 18 observed scene overlays saved; no evaluation target geometry read')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);main(ap.parse_args().run)
