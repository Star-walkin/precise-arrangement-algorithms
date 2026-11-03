"""Independent read-only frozen dataset checks, no label/cache modifications."""
from pathlib import Path
import json,hashlib,collections,time
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cv2.setNumThreads(1);tick=time.time();pr=load(HERE/'protocol.json');ix=load(HERE/'dataset/index.json');old=load(BASE/'runs/run_20261030_002/data/index.json')
    new=[i for i in ix['items'] if i['origin']=='new'];prior=[i for i in ix['items'] if i['origin']=='old'];assert len(new)==24 and len(prior)==10
    tg={i['source_group'] for i in ix['items']};eg={i['source_group'] for i in old['items'] if i['split']=='eval'}
    assert len(tg)==7 and len(eg)==3 and not(tg&eg)
    assert len({i['image_sha256'] for i in new})==24 and not({i['image_sha256'] for i in new}&{i['image_sha256'] for i in old['items']})
    rows=[];signatures={};timegroups=collections.defaultdict(list)
    lookup={j['id']:j for j in pr['candidate_jobs']}
    for item in new:
        a=load(item['annotation_path']);assert sha(item['annotation_path'])==item['annotation_sha256']==ix['label_hashes'][item['annotation_path']]
        assert a['review']['original_viewed'] and a['review']['native_overlay_viewed']
        assert sha(item['image_path'])==item['image_sha256'];assert sha(item['cache'])==item['cache_sha256']
        with np.load(item['cache']) as z:
            image,seed,target,valid,stage,roi=[z[k] for k in ['image','seed','target','valid','stage','roi']]
        assert image.dtype==target.dtype==seed.dtype==np.float32
        assert valid.dtype==roi.dtype==bool and stage.dtype==np.uint8
        assert all(v.shape==(975,2048) for v in [image,seed,target,valid,stage,roi])
        assert np.isfinite(image).all() and image.min()>=0 and image.max()<=1 and roi.all()
        assert np.isfinite(target).all() and 0<=target.min()<=target.max()<=1
        assert np.isfinite(seed).all() and seed.sum()>0
        assert not np.any(valid&((stage<1)|(stage>4))) and not np.any(target[~valid])
        assert np.any(valid&(target>=.5)) and np.any(valid&(target<.5))
        # Reconstruct geometry and exact permitted tube, independently from
        # cached target. Endpoint caps omitted from each individual fragment.
        expected=np.zeros(valid.shape,bool);end_exclusion_count=0;accepted=0
        for seg in a['segments']:
            if seg['status']!='accepted_weak':continue
            accepted+=1;line=np.asarray(seg['points_xy']);q=np.rint((line-[0,1050])*.5).astype(np.int32)
            lm=np.zeros(valid.shape,np.uint8);cv2.polylines(lm,[q],False,1,1,cv2.LINE_8)
            distance=cv2.distanceTransform(1-lm,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            local=distance<=seg['valid_radius_px']*.5
            yy,xx=np.indices(valid.shape)
            for p in [line[0],line[-1]]:
                center=(p-[0,1050])*.5;cap=(xx-center[0])**2+(yy-center[1])**2<=9.
                end_exclusion_count+=int((local&cap).sum());local&=~cap
            expected|=local
        assert np.array_equal(expected,valid),'local tubes or endpoint ignore mismatch '+item['id']
        raw=cv2.imdecode(np.fromfile(item['image_path'],np.uint8),cv2.IMREAD_COLOR)
        gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        resized=cv2.resize(gray[1050:],(2048,975),interpolation=cv2.INTER_AREA)
        assert np.array_equal(image,resized),'gray ordering/crop mismatch'
        small=cv2.resize(gray[1050:],(256,128),interpolation=cv2.INTER_AREA);s=cv2.resize(small,(32,32),interpolation=cv2.INTER_AREA);dct=cv2.dct(s)[:8,:8];signatures[item['id']]=(dct>np.median(dct[1:])).ravel()
        j=lookup[item['id']];timegroups[item['source_group']].append((float(j['source_time_s']),item['id']))
        rows.append(dict(id=item['id'],valid_pixels=int(valid.sum()),ignore_fraction=float((~valid).mean()),
            center_high_pixels=int((valid&(target>=.5)).sum()),accepted_fragments=accepted,
            local_valid_tubes_exact_match=True,endpoint_cap_removed_pixels_sum=end_exclusion_count,
            original_gray_crop_exact=True,label_hash_frozen=True,new_input_has_actual_zero_gray=bool((image==0).any())))
    intervals={};close_pairs=[]
    for g,rr in timegroups.items():
        rr=sorted(rr);dt=np.diff([t for t,i in rr]);intervals[g]=dict(images=len(rr),minimum_selected_time_gap_s=float(dt.min()) if len(dt) else None,median_selected_time_gap_s=float(np.median(dt)) if len(dt) else None)
        for (ta,a),(tb,b) in zip(rr,rr[1:]):
            hamm=int((signatures[a]!=signatures[b]).sum())
            close_pairs.append(dict(group=g,id1=a,id2=b,seconds=tb-ta,perceptual_hash_hamming=hamm,interpretation='coarse appearance diagnostic only; small distance does not establish exact duplicate'))
    out=dict(date='2026-11-02',type='independent read-only artifact/schema/geometry review, not independent visual identity truth',
        new_images=24,old_images=10,train_groups=sorted(tg),eval_groups=sorted(eg),train_eval_group_intersection=[],
        exact_new_image_duplicate_count=0,new_image_hash_overlap_with_old18=0,all24_caches_and_label_hashes_verified=True,
        per_image_checks=rows,selected_temporal_intervals=intervals,adjacent_selected_pair_appearance_diagnostics=close_pairs,
        limitations=['New left stage2 and right stage3 are visual projection proxies, not independently verified physical contact or material identity stages.',
          'The existing stage-balanced loss weights occupied visual buckets; differences in stage conventions can affect weighting, so this is not a pure sample-count experiment.',
          'Adjacent images within a source group remain correlated;24 images add3 source groups, not24 independent conditions.',
          'Native center refinement and same-model visual review are weak-label consistency evidence, not calibrated center error or independent identity accuracy.',
          'No confirmed stage4 new target endpoint, unresolved post-turn branches remain ignore.',
          'No explicit wrong-neighbor identity negatives were fabricated from unlabelled pixels.'],
        critical_data_mapping_bug_found=False,seconds=time.time()-tick,no_labels_or_caches_modified=True)
    p=HERE/'dataset_quality_review.json';assert not p.exists();p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print('Reviewed24 frozen caches; no mapping/source/endpoint bug. Seconds',out['seconds']);print(intervals)
if __name__=='__main__':main()
