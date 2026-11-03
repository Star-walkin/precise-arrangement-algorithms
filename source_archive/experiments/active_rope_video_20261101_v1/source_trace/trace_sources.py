"""Trace the explicit open400 image manifest to16 original videos and10 stills.

No device imports. Video decoding is limited to the preselected earliest
dataset frame from each whitelisted source, only for exact pixel replay.
"""
from __future__ import annotations
import csv,hashlib,json,platform,sys,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT/'数据资产/05_派生与实验结果/camera_condition_images_20261009_v2'
VIDEO_ROOT=ROOT/'数据资产/01_原始采集/现场视频/训练采集'
STILL_ROOT=ROOT/'现场图片/钢尺测量/测试'
H6=ROOT/'experiments/entry_conditioned_rope_path/postprocess_b_20261031_v6/inputs.json'

def sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):d.update(block)
    return d.hexdigest()

def dump(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    sys.stdout.reconfigure(encoding='utf8');cv2.setNumThreads(1);tick=time.time()
    manifest=DATA/'manifest.csv';rows=read_csv(manifest)
    old_hashes=json.loads((DATA/'source_hashes.json').read_text(encoding='utf8'))
    h6=json.loads(H6.read_text(encoding='utf8'));h6jobs={r['id']:r for r in h6['jobs']}
    assert len(rows)==400 and len(h6jobs)==292
    grouped=defaultdict(list)
    for r in rows:grouped[r['source_file']].append(r)
    assert len(grouped)==26
    videos=[];stills=[];image_map=[];errors=[]
    for source,items in grouped.items():
        path=Path(source).resolve(strict=True)
        assert not any(x in str(path) for x in ['20260925','dual_20260923_113029','dual_20260923_113745'])
        isvideo=path.suffix.lower()=='.avi'
        assert path.is_relative_to((VIDEO_ROOT if isvideo else STILL_ROOT).resolve())
        expected=old_hashes[source];current=sha(path)
        assert current==expected['sha256'] and path.stat().st_size==expected['bytes']
        assert all(x['source_sha256']==current for x in items)
        basic=dict(source_group=items[0]['source_group'],camera_role=items[0]['camera_role'],
            observed_edge=items[0]['observed_edge'],camera_serial=items[0]['camera_serial'] or None,
            source_file=source,canonical_path=str(path),bytes=path.stat().st_size,sha256=current,
            hash_matches_frozen_400_manifest=True,dataset_images=len(items),
            winding_layer=items[0]['winding_layer'],winding_layer_evidence=items[0]['winding_layer_evidence'],
            historical_use=items[0]['historical_use'],dataset_use_status=items[0]['dataset_use_status'],
            already_in_original_asset_library=isvideo,copied=False)
        if not isvideo:
            ident=Path(items[0]['output_file']).stem
            stills.append(basic|dict(image_id=ident,has_source_video=False,reason='Independent ruler-test BMP; no source video claimed.',
                has_h6_automatic_entry=ident in h6jobs))
            image_map.append(dict(id=ident,source_file=str(path),kind='still',dataset_image=str(DATA/items[0]['output_file']),
                dataset_image_sha256=items[0]['output_sha256'],source_sha256=current,source_frame=None,source_time_s=None,
                has_h6_automatic_entry=ident in h6jobs))
            continue
        items.sort(key=lambda r:int(r['source_video_frame_zero_based']))
        video_id=basic['source_group']+'__'+basic['camera_role']
        cap=cv2.VideoCapture(str(path));assert cap.isOpened(),str(path)
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        code=int(cap.get(cv2.CAP_PROP_FOURCC));codec=''.join(chr((code>>(8*i))&255) for i in range(4))
        assert n>0 and fps>0 and (w,h)==(4096,3000)
        sidecar=path.with_name(basic['camera_role']+'_frames.csv')
        if basic['source_group'].startswith('dual_'):
            ts=read_csv(sidecar);assert len(ts)==n
            assert [int(x['video_frame']) for x in ts]==list(range(n))
            times=[float(x['host_time_s']) for x in ts];time_basis='pc_host_receipt_monotonic_s_not_exposure'
            assert all(b>a for a,b in zip(times,times[1:]))
            sidecars={str(p):sha(p) for p in [sidecar,path.with_name('paired_frames.csv'),path.with_name('session.json')]}
        else:
            times=[i/fps for i in range(n)];time_basis='video_frame_index_divided_by_nominal_fps_approximate';sidecars={}
        tpath=HERE/'frame_times'/(video_id+'.json')
        dump(tpath,dict(video_id=video_id,source_file=str(path),frame_indices_zero_based=list(range(n)),
            observation_times_s=times,time_basis=time_basis,nominal_fps=fps,
            media_playback_time_formula='zero_based_frame/nominal_fps',exposure_time_known=False))
        maps=[]
        for r in items:
            ident=Path(r['output_file']).stem;f=int(r['source_video_frame_zero_based']);observed=float(r['source_time_s'])
            assert 0<=f<n
            err=abs(times[f]-observed);assert err<1e-5,(ident,err)
            m=dict(id=ident,source_file=str(path),video_id=video_id,kind='video_frame',source_frame=f,
                source_time_s=observed,source_time_basis=r['source_time_basis'],media_playback_time_s=f/fps,
                mapped_observation_time_s=times[f],mapping_absolute_error_s=err,
                dataset_image=str(DATA/r['output_file']),dataset_image_sha256=r['output_sha256'],source_sha256=current,
                has_h6_automatic_entry=ident in h6jobs,h6_condition_count=len(h6jobs[ident]['seeds']) if ident in h6jobs else 0)
            maps.append(m);image_map.append(m)
        first=maps[0];f=first['source_frame'];assert cap.set(cv2.CAP_PROP_POS_FRAMES,f)
        ok,frame=cap.read();position_after=cap.get(cv2.CAP_PROP_POS_FRAMES);cap.release();assert ok
        png=cv2.imdecode(np.fromfile(first['dataset_image'],dtype=np.uint8),cv2.IMREAD_COLOR)
        assert png is not None and png.shape==frame.shape
        exact=bool(np.array_equal(png,frame));maxdiff=int(np.abs(png.astype(np.int16)-frame.astype(np.int16)).max())
        if not exact:errors.append(dict(video_id=video_id,kind='earliest_frame_pixel_mismatch',maximum_abs_difference=maxdiff))
        h6maps=[m for m in maps if m['has_h6_automatic_entry']]
        video=basic|dict(video_id=video_id,frame_count_metadata=n,nominal_fps=fps,width=w,height=h,codec=codec,
            metadata_duration_s=n/fps,last_frame_media_time_s=(n-1)/fps,
            observation_time_basis=time_basis,observation_first_s=times[0],observation_last_s=times[-1],
            frame_times_path=str(tpath),frame_times_sha256=sha(tpath),sidecar_sha256=sidecars,
            earliest_dataset_anchor=first,last_dataset_anchor=maps[-1],
            earliest_h6_anchor=h6maps[0] if h6maps else None,h6_dataset_images=len(h6maps),
            no_h6_entry_is_not_absence_of_rope=True,
            earliest_frame_replay=dict(decoded_frame=f,cap_position_after=position_after,dataset_image=first['dataset_image'],
                pixel_equal=exact,maximum_abs_pixel_difference=maxdiff,decoded_shape=list(frame.shape),
                no_resize_crop_mirror=True,frame_decodes=1))
        videos.append(video);print(json.dumps(dict(video=video_id,pixel_equal=exact,n=n,fps=fps,seconds=n/fps,h6_images=len(h6maps)),ensure_ascii=False),flush=True)
    assert len(videos)==16 and len(stills)==10 and len(image_map)==400
    result=dict(schema_version=1,status='passed' if not errors else 'pixel_replay_discrepancy',created_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv,python=sys.version,opencv=cv2.__version__,script_sha256=sha(__file__),manifest=str(manifest),manifest_sha256=sha(manifest),
        historical_source_hashes=str(DATA/'source_hashes.json'),historical_source_hashes_sha256=sha(DATA/'source_hashes.json'),
        h6_inputs_sha256=sha(H6),source_images=400,video_derived_images=390,standalone_still_images=10,
        videos_count=16,video_source_groups=11,total_source_groups=21,h6_eligible_images=292,h6_video_images=282,h6_still_images=10,
        video_bytes=sum(v['bytes'] for v in videos),video_metadata_duration_sum_s=sum(v['metadata_duration_s'] for v in videos),
        total_video_frames_metadata=sum(v['frame_count_metadata'] for v in videos),decoded_source_frames=16,
        no_h6_entry_video_ids=[v['video_id'] for v in videos if v['earliest_h6_anchor'] is None],
        canonical_video_library=str(VIDEO_ROOT.resolve()),source_copy_performed=False,
        original_files_changed=False,reserved_or_excluded_sources_opened=False,devices=False,
        source_time_warning='Dual-camera host receive times are not exposure times; frame/fps is encoded playback time. Never treat them as interchangeable calibrated clocks.',
        videos=videos,stills=stills,errors=errors,elapsed_seconds=time.time()-tick)
    dump(HERE/'source_trace.json',result);dump(HERE/'image_source_map.json',dict(rows=image_map,manifest_sha256=sha(manifest)))
    dump(HERE/'open_source_video_whitelist.json',dict(paths=[v['canonical_path'] for v in videos],sha256={v['canonical_path']:v['sha256'] for v in videos},
        explicit_open_400_derived_sources_only=True,reserved_allowed=False,trace_sha256=sha(HERE/'source_trace.json')))
    summary=['开放400图的原始视频溯源','',f'原始视频库：{VIDEO_ROOT.resolve()}',
        '400图对应16段视频中的390帧，以及10张独立钢尺BMP；不是400段独立视频。',
        f'16段视频总时长（容器标称）{result["video_metadata_duration_sum_s"]:.3f}秒，{result["total_video_frames_metadata"]}帧，{result["video_bytes"]}字节。',
        '所有16段已在数据资产原始采集库内，因此不重复复制。原素材未修改。',
        '已逐文件核验26份原始来源SHA-256，逐一回放16段最早400图对应帧；像素不一致列表见JSON。',
        'source_trace.json记录逐源帧数/FPS/编解码器/最早锚点；image_source_map.json记录全部400图的原视频帧号与时间；frame_times/保存每帧时间口径。',
        '双机会话采用主机接收时刻，不是曝光时刻；frame/FPS是播放器时间。单机仅有近似frame/FPS时间。',
        '5段视频在旧H6清单中未获得自动入口，应显示未初始化或未识别，不能称为无绳。',
        '钢尺BMP权威源仍为根目录现场图片；没有凭文件目录补造对应视频。',
        '未读取9月25日、113029、113745或任何其他排除视频；未读取标签、连接设备或改变权限。','']
    for v in videos:summary.append(f'{v["video_id"]}: {v["metadata_duration_s"]:.3f}s / {v["frame_count_metadata"]}帧 / {v["nominal_fps"]:.6f}fps / H6抽图入口{v["h6_dataset_images"]}/{v["dataset_images"]}\n{v["canonical_path"]}')
    (HERE/'README.txt').write_text('\n'.join(summary)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','videos_count','video_metadata_duration_sum_s','total_video_frames_metadata','video_bytes','decoded_source_frames','elapsed_seconds']},ensure_ascii=False))
    return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
