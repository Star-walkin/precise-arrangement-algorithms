"""Collect finished offline outputs without changing inference or historical data."""
import json,hashlib,os,sys,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    run=HERE/'all_001';s=read(run/'summary.json');trace=read(HERE/'source_trace/source_trace.json')
    if s['completed_videos']!=16 or s['failures']:raise RuntimeError('All16 sources must finish before delivery')
    deliver=HERE/'四子图视频';deliver.mkdir(exist_ok=True)
    counts=collections.Counter();engine_counts=collections.Counter();reasons=collections.Counter();rows=[]
    guard_rejected=changed_rank=temporal_rejected=0;analyses=forwards=rendered=0
    for v in s['results']:
        ident=v['video_id'];folder=run/ident;src=Path(v['output_path']);dst=deliver/src.name
        if not dst.exists():os.link(src,dst)
        if sha(dst)!=v['output_sha256']:raise ValueError('Delivery link changed')
        counts.update(v['counts']);analyses+=v['analysis_frames'];forwards+=v['forward_count'];rendered+=v['output_frames']
        for line in (folder/'detections.jsonl').read_text(encoding='utf8').splitlines():
            r=json.loads(line);engine_counts[r['engine_status']]+=1;t=r['temporal']
            candidates=r['gate_candidates'];guard=next((i for i,c in enumerate(candidates) if c['image_supported']),None)
            if candidates and guard is None:guard_rejected+=1
            if guard is not None and not t['accepted']:temporal_rejected+=1
            if t['accepted'] and t['accepted_candidate_index']!=guard:changed_rank+=1
            for c in t.get('candidate_diagnostics',[]):
                for reason in c.get('reasons',[]):reasons[reason]+=1
        rows.append(dict(video_id=ident,video=str(dst),sha256=v['output_sha256'],
            analysis_frames=v['analysis_frames'],forward_count=v['forward_count'],states=v['counts'],
            source_duration_s=v['source_duration_s'],output_duration_s=v['output_duration_s'],wall_seconds=v['wall_seconds']))
    report=dict(version='IR-V1.0.0',image_baseline='活动绳整链基准 P1 / IR-P1.0.0',status='completed_offline',
        videos=16,source_images400=400,video_derived_gallery_images=390,standalone_stills=10,
        independent_capture_source_groups=trace['video_source_groups'],total_source_seconds=trace['video_metadata_duration_sum_s'],
        analysis_frames=analyses,new_B_forwards=forwards,output_video_frames=rendered,states=dict(counts),
        engine_states=dict(engine_counts),current_image_guard_rejections=guard_rejected,
        temporal_rejections_with_current_image_candidate=temporal_rejected,temporal_choice_differs_from_image_guard=changed_rank,
        wall_seconds=s['wall_seconds'],analysis_throughput_all_workers_fps=analyses/s['wall_seconds'],
        online_realtime_verified=False,full_identity_accuracy_evaluated=False,model_training=False,
        baseline_B_changed=False,H7_used=False,reserved_sources_opened=False,devices_connected=False,
        source_trace_sha256=sha(HERE/'source_trace/source_trace.json'),protocol_sha256=sha(run/'protocol.json'),
        source_video_library=trace['canonical_video_library'],delivery_folder=str(deliver),outputs=rows,
        limitations=['earliest allowed image scene prior; not arbitrary raw-frame cold start',
            'SAM2/F3 geometry propagated from first anchor and checked against current image; not recomputed full F3 each frame',
            'automatic entry can be absent or incorrect; missing response is explicitly uncomputed',
            'temporal stability and rejection cannot establish same-rope identity or full attached-tail correctness',
            '3seconds per projected spacing is an uncalibrated heuristic; host receipt timestamps are not exposures'])
    write(HERE/'experiment_report.json',report)
    table='\n'.join(f"{r['video_id']} | {r['analysis_frames']}次 | {r['states']} | {r['source_duration_s']:.2f}s" for r in rows)
    text=f'''活动绳视频时序 V1 / IR-V1.0.0 — 实际运行记录（2026-11-01）

完成16/16段来源视频的原图、冻结方案B热图、候选、最终位置四子图，文件位于：
{deliver}
视频为1600×1280、5fps、按源视频原速播放；算法每0.5秒源观测时间分析一次。
源素材共{trace['video_metadata_duration_sum_s']:.3f}秒，处理总墙钟{s['wall_seconds']:.1f}秒。
当前CPU离线吞吐为{analyses/s['wall_seconds']:.3f}分析帧/秒（全部进程合计），不声称在线实时达标。

基准与溯源
原方案B750＋H6整链封存为活动绳整链基准P1 / IR-P1.0.0，338份源码/配置/模型副本核验通过。
该名称区别于2026-10-30的局部短缺口连接器P1；本次指整链档案。未采用H7 C/D，未改动B权重。
历史400图=16段视频中的390帧＋10张独立钢尺照片。视频已有规范原始资产地址，无需重复复制：
{trace['canonical_video_library']}
26份来源文件哈希匹配，16个首锚视频帧与图库PNG像素完全相同。10张照片没有对应视频。
历史5段没有H6入口也保留完整视频，并用当帧A7尝试入口；未知不是无绳。

当前帧与历史如何分工
首个开放图库锚点只提供SAM2/F3场景几何；此前画面显示未初始化，不读未来锚点补回。
此后只沿静态机械背景做有界LK配准，并重新检查当前原图/热图支持。不是每帧重新跑完整F3/SAM2。
首锚旧A6/A7记录因来源加载被读取后立即清空，不用旧目标路径选入口或回填。视频B热图均当前帧新前向。
A7当前图像自动发现入口，每3秒刷新；间隔内入口LK还需原图局部结构支持。
B保持原裁切、灰度、归一化、种子和权重；两个旧样例数值复放误差均小于1e-5。
至少80原图像素当前原图支持后，原H6最多3候选进入因果门控。未知尾段不会因此自动被确认。
门控只参考过去最多5次接受/2秒历史。3秒每投影绳距是用户速度描述的启发式，并非物理标定。
跳动拒绝不写回接受历史；1秒内仅显示橙虚线旧参考，过期UNKNOWN；持续3次新候选可建立新状态段，
明确标REACQUIRED，不声称跨缺口同根身份恢复。没有用未来帧、没有平均路径后补线。

实际结果与边界
分析{analyses}帧，B新前向{forwards}次，视频输出{rendered}帧。
状态计数：{dict(counts)}。
有当前图像候选但时序未接受：{temporal_rejected}次；时序选择不同于相同图像守卫首选：{changed_rank}次。
这些是程序状态/产物数，不是正确率；UNKNOWN、拒识或稳定均不能证明识别正确。
首轮短片已看到右侧清晰例持续接受、左侧弱例拒绝及重捕获、无入口例保持未知。
整段贴面身份、真实目标末端及真实遮挡仍无独立完整参考，不能由视频平稳替代验证。
后续优先检查自动入口和长时场景几何是否仍正确，再决定周期重测F3；本轮不优化热图B。

逐视频
{table}

复查材料
all_001/protocol.json：运行前冻结的参数/输入清单/代码哈希。
每源detections.jsonl：当前完整候选、未经时序的首选、同图像守卫首选、时序状态与咨询历史时刻。
每源responses/*.npz：原始float32热图，按entry_0等分开保存，含原坐标bbox；未从路径重画热图。
每源render_frames.jsonl：播放帧→源帧→分析帧映射、结果年龄、当前/旧参考显示状态。
source_trace/：400图来源映射、视频白名单、每帧时间口径、来源哈希。
temporal/test_receipt.json及rendering/render_check_receipt.json：机制和显示检查，均不是身份准确率。
audit/：独立因果/坐标/媒体完整性审计。完整输出固定抽样目视记录另见visual_review.json。
无训练、标签或保留集访问，无设备连接、上线或权限变更。原始素材和历史结果保持不变。
'''
    (HERE/'REPORT.txt').write_text(text,encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['outputs','limitations']},ensure_ascii=True))
if __name__=='__main__':main()
