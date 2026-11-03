"""Deliver completed, audited per-source-frame videos and register evidence."""
import json,os,hashlib,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    run=HERE/'all_001';s=read(run/'summary.json');a=read(run/'audit_receipt.json')
    assert s['status']=='complete' and s['videos']==16 and s['frames']==18017
    assert a['status']=='passed' and a['frames']==18017
    delivery=HERE/'四子图视频';delivery.mkdir(exist_ok=True);states=collections.Counter();files=[]
    for v in s['results']:
        src=Path(v['output_path']);dst=delivery/src.name
        if not dst.exists():os.link(src,dst)
        assert sha(dst)==v['output_sha256']
        files.append(dict(path=str(dst),sha256=v['output_sha256'],frames=v['output_frames'],fps=v['output_fps']))
        states.update(v['states'])
    report=dict(version='IR-V2.0.0',status='completed_offline',videos=16,source_frames=18017,
        fresh_gray_detections=18017,output_frames=18017,
        new_B_forwards_core_frames=s['new_B_forwards'],actual_B_forwards_including_warmup=a['actual_B_forwards_including_chunk_warmup'],
        actual_gray_detection_calls_including_warmup=a['actual_gray_detection_calls_including_warmup'],
        states=dict(states),wall_seconds=s['wall_seconds'],delivery_folder=str(delivery),files=files,
        gray_coordinate_changes=sum(v['consecutive_gray_geometry_changes'] for v in a['videos']),
        gray_detection_rule='fresh original-gray observations every source frame; current adjacent-boundary midpoints used by H6',
        old_gray_coordinate_fallback=False,held_heatmaps=False,held_gray_centers=False,
        original_full_F3_fusion_rerun_per_frame=False,static_roi_calibration_used=True,
        B750_weights_changed=False,training=False,devices=False,reserved_sources=False,
        compute_realtime_verified=False,target_identity_accuracy_evaluated=False,
        protocol_sha256=sha(run/'protocol.json'),audit_sha256=sha(run/'audit_receipt.json'),
        audit_checks=a['checks'],source_library='D:/precise_arrangement/数据资产/01_原始采集/现场视频/训练采集')
    write(HERE/'experiment_report.json',report)
    text=f'''活动绳视频逐帧V2 / IR-V2.0.0（2026-11-01）

本轮实际完成16段、18,017个源帧的灰线新检测、当前B响应及候选接续，并按源帧率输出18,017帧四子图。
结果：{delivery}
完整运行墙钟：{s['wall_seconds']:.1f}秒。当前CPU离线运算；源帧率回放不代表计算实时。

修正的原因
V1的_scene只对首锚中心线做共同位移，_measured_tracks只更新图像支持分，不改变线坐标。
V2的_scene每次从当前图像重新提取宽沟槽边界，生成相邻边界中点；H6连接用这批当前中心线。
没有当帧证据时留空，不取旧灰线回填；STALE只允许独立标明的旧活动绳参考，不是当前识别。

方法与边界
保持原B750模型、灰度输入、裁切和标准化，网络在存在自动入口的每个源帧重新前向。
卷面新观测采用ROI加权LCN、沿宽纹理走向积分、当前原灰双侧亮肩、有限行序沟槽和相邻边界中点。
它是轻量的视频新观测实现，不是逐帧运行原F3的全部历史融合链。
已有C4/SAM2卷面掩模和C3粗方向/未标定投影间距只作离线场景配置；未使用保存的灰线坐标作输出。
因此场景ROI本身还不是逐帧SAM2重测，相机大幅位移/ROI形变与完整目标身份仍需另外验证。
旧scene数据作为离线校准在第0帧即加载，其来源明确保存；这不属于依据未来目标路径做时序判断。
细直绳LK孔径失败时，当前原图法向±4px剖面检查才允许继续使用短入口；失败仍保留未知。

并行与历史
按20秒块并行测量，块前至多1秒历史画面用于自动入口预热。最终每段视频全程按源帧序作统一历史筛选。
N适配各视频fps，约半秒历史；重捕获帧数同步适配，避免高帧率时队列跨度不足0.4秒。
每个接受结果仍是当帧候选原始点，未用时序平滑补线；所有引用的目标位置历史早于当前时刻。
整体状态数：{dict(states)}。这些是产物状态，不是真实正确率。

实际验证
左右两图合成平移4原像素，检测中心线中位移动均为4像素；全黑输入均无灰线。
pilot_001保留了LK孔径失效反例；pilot_002修复后45/45源帧逐帧检测、694项结构检查通过。
全量独立审计{a['checks']}项，16视频完整解码，18,017源帧/检测帧/输出帧逐一对应。
所有贴面候选尾段逐点核验位于同帧最新中心线上。灰线几何摘要相邻变化累计{report['gray_coordinate_changes']}次。
真实静止画面允许相同坐标，不按时间人为制造位移；结构检查和坐标变化不能替代逐根真值。
弱光断线、错误入口、活动绳覆盖不完整、未知与竞争接续仍可能存在，未据此声称完整身份准确。

数据与缓存
仍为原开放400图对应的16段视频；10张独立照片无视频。原素材、P1和V1均保留。
当帧检测使用原始完整float32响应；视频显示缓存为800×381的固定0..1 uint16 PNG，不参与检测。
每5秒float_keyframes/保存完整原始浮点图，逐帧原响应摘要保存在measurements.jsonl.gz。
每帧灰线、当前候选、源帧号、灰线检测帧号、引用历史和最终选择均保存，可核对无旧线回填。
协议：all_001/protocol.json；审计：all_001/audit_receipt.json；代码、命令、旧失败见README.txt。
无训练、设备连接、保留集或S1标签访问。
'''
    (HERE/'REPORT.txt').write_text(text,encoding='utf8')
    tag='视频逐帧灰线V2 / IR-V2.0.0'
    entry=(f'2026-11-01新增[{tag}](D:/precise_arrangement/experiments/active_rope_video_20261101_v2/experiment_report.json)：'
        '修正V1灰线几何沿用首锚的问题，同16开放视频18,017源帧均从当前原灰重新提取边界并转相邻中心，'
        'H6贴面候选使用当帧中心；原B750权重不改，有入口帧重新前向，输出原帧率四子图。'
        f'18,017帧逐一对齐及{a["checks"]}项结构审计通过；这是轻量新观测，非原F3全部融合链重跑。'
        '静态场景ROI/粗方向配置仍沿用，弱区断线和同根身份未验收，不以坐标变化或审计通过解释为真实准确率。'
        'P1/V1保留，无训练、设备或保留集访问。')
    changes=[]
    for p in [ROOT/'数据资产/README.md',ROOT/'工程导航/05_实验版本档案/README.md',ROOT/'experiments/entry_conditioned_rope_path/README.md']:
        before=sha(p);txt=p.read_text(encoding='utf8')
        if tag not in txt:p.write_text(txt+'\n\n## '+tag+'\n\n'+entry+'\n\n[16段逐帧四子图视频](D:/precise_arrangement/experiments/active_rope_video_20261101_v2/四子图视频)\n',encoding='utf8')
        changes.append(dict(path=str(p),before=before,after=sha(p)))
    p=ROOT/'AGENTS.md';before=sha(p);txt=p.read_text(encoding='utf8')
    if tag not in txt:
        i=txt.index('- V14训练拟合');p.write_text(txt[:i]+'- '+entry+'\n'+txt[i:],encoding='utf8')
    changes.append(dict(path=str(p),before=before,after=sha(p)));write(HERE/'registration_receipt.json',changes)
    write(HERE/'delivery_receipt.json',dict(status='complete',files=files,report_sha256=sha(HERE/'experiment_report.json'),audit_sha256=sha(run/'audit_receipt.json')))
    print(json.dumps({k:v for k,v in report.items() if k!='files'},ensure_ascii=True))
if __name__=='__main__':main()
