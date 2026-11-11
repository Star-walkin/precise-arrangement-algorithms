"""Summarize completed measurements without equating coverage with accuracy."""
from pathlib import Path
import sys,json,gzip,hashlib,subprocess
from collections import Counter
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'RopeEdgeSuite_R11_20261108'))
from run_bg5 import e,record,dump

def intervals(rows,predicate):
    groups=[];current=[]
    for r in rows:
        if predicate(r):current.append(r['frame'])
        elif current:groups.append(current);current=[]
    if current:groups.append(current)
    return [dict(first=g[0],last=g[-1],frames=len(g),duration_s=len(g)/10) for g in groups]

def main():
    run=ROOT/'runs/dual_20260923_113745';front=ROOT/'frontend/dual_20260923_113745_v2'
    receipt=e.read(run/'receipt.json');assert receipt['status']=='complete' and receipt['frames']==600
    audit=e.read(run/'audit_summary.json');assert audit['complete'] and audit['checks_passed'] and audit['frames']==600
    video=e.read(run/'delivery/video_receipt.json');assert video['decoded_frames']==600 and video['fps']==10
    assert e.read(ROOT/'test_receipt.json')['passed'] and e.read(ROOT/'actual_case_test_receipt.json')['passed'] and e.read(ROOT/'resume_parity_receipt.json')['passed']
    assert e.read(ROOT/'actual_memory_recovery_receipt.json')['passed']
    old=e.read(ROOT.parent/'RopeEdgeSuite_R11_20261108/runs/dual_20260923_113745_refined/compact.json')
    new=e.read(run/'compact.json');unknown=Counter();gaps=[];attempts=Counter();valid_after=[]
    for i,r in enumerate(new):
        full=record(run,i);s=full['s1'];reg=full['region'];y=full['intersection_y_px']
        gaps.append(None if y is None or reg['edge_y'] is None else y-reg['edge_y'])
        if r['count_state']=='unknown':
            why=('活动绳当前证据不足' if not s['valid'] else '卷面当前证据不足' if not reg['valid'] else
                 '固定d处活动绳投影在边界上方' if y is not None and y<reg['edge_y'] else 'F3局部相位或连续计数证据不足')
            unknown[why]+=1
        for a in full['S1_tracker_full']['diagnostics']['attempts']:
            for why in a.get('reasons',[]):attempts[why]+=1
        if i>=316:valid_after.append(r['s1_valid'])
    def metric(rows):return dict(frames=len(rows),S1_gate_passed=sum(r['s1_valid'] for r in rows),
        region_gate_passed=sum(r['region_valid'] for r in rows),count_unknown=sum(r['count_state']=='unknown' for r in rows),
        count_numeric=sum(r['count_value'] is not None for r in rows),within_five_displayed=sum(r['display_N'] is not None for r in rows))
    runs=intervals(new,lambda r:not r['s1_valid'])
    results=dict(old=metric(old),new=metric(new),new_after31p6_s=dict(frames=len(valid_after),S1_gate_passed=sum(valid_after)),
        S1_invalid_intervals=runs,count_unknown_first_blocker=dict(unknown),candidate_rejections=dict(attempts),
        display_step_diagnostics=audit['display_step_diagnostics'],
        interpretation='gate pass coverage, not ground truth accuracy; first-blocker categories overlap other stage failures',
        d_original_px=2238.7,d_recalibrated=False,ground_truth_accuracy_available=False)
    dump(ROOT/'comparison.json',results)
    # A compact result timeline, also distinguishing projection invalidity.
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    t=np.arange(600)/10;fig,ax=plt.subplots(3,1,figsize=(12,7),sharex=True)
    ax[0].step(t,[r['s1_valid'] for r in old],where='post',label='RES-1.1',color='#b76a54')
    ax[0].step(t,np.asarray([r['s1_valid'] for r in new])+.04,where='post',label='RES-1.2 / SG2',color='#207c83')
    ax[0].set_yticks([0,1],['未通过','通过']);ax[0].set_ylabel('活动绳检查');ax[0].legend(loc='lower left')
    for rows,label,color in [(old,'原版','#b76a54'),(new,'新版','#207c83')]:
        ax[1].plot(t,[np.nan if r['display_N'] is None else r['display_N'] for r in rows],label=label,color=color,lw=1.2)
    ax[1].set_ylabel('剩余根数（0～5）');ax[1].set_ylim(-.2,5.3)
    ax[2].plot(t,[np.nan if v is None else v for v in gaps],color='#67508e',lw=1)
    ax[2].axhline(0,color='black',lw=.8);ax[2].set_ylabel('d处投影－边界 / px');ax[2].set_xlabel('视频时间 / s')
    for a in ax:a.axvline(31.6,color='#777777',ls='--',lw=.8);a.grid(alpha=.2)
    fig.suptitle('113745：完整600帧对照；通过率不等于准确率，负投影差不强行计为零')
    fig.tight_layout();fig.savefig(ROOT/'comparison_timeline.png',dpi=150);plt.close(fig)
    # Re-encode an exact source-time highlight; never replace the complete video.
    clip=run/'delivery/RES12_SG2_113745_30s-42s.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-ss','30','-i',video['path'],'-t','12','-an',
        '-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(clip)],check=True)
    c=cv2.VideoCapture(str(clip));n=0
    while True:
        ok,im=c.read()
        if not ok:break
        assert im.shape==(1080,1920,3);n+=1
    fps=c.get(5);c.release();assert n==120 and fps==10
    dump(run/'delivery/highlight_receipt.json',dict(path=str(clip),source_start_s=30,source_end_s=42,decoded_frames=n,fps=fps,sha256=e.sha(clip)))
    source=e.read(run/'binding.json');finalfront=e.read(front/'binding.json')
    source.update(final_frontend_binding_sha256=e.sha(front/'binding.json'),frontend_resumptions=finalfront.get('resumptions',[]),
        final_tracker_code_sha256=e.sha(ROOT/'tracker_sg2.py'),baseline_manifest_reverified_sha256=e.verify_baseline())
    dump(run/'binding.json',source)
    assert e.sha(source['source'])==source['source_sha256']
    longest=max((x['frames'] for x in intervals(new[316:],lambda r:not r['s1_valid'])),default=0)
    report=f'''RES-1.2 / SG2 活动绳连续跟踪优化结果

输入：113745左相机，4096×3000，10fps，完整600帧、60秒。输出1920×1080、10fps。原基准未改动。
四项修改：
1. 分段实测支持拟合：允许有限缺口，长度限制为当前宽度4倍、上限120px，同时检查缺口两侧宽度、极性、全局支持比例和残差。只使用当前实测点，不将插值点用于拟合。
2. 短时搜索记忆：保留最近可信轨迹0.5秒用于定位；失败帧不输出历史ABC作为当前测量。下一帧重新检测，超时重新捕获。
3. 候选身份与事务提交：校验宽度、角度、位置连续性；从零或失锁重建需要连续2帧确认，确认后更新锚点。
4. 当前BG5.1区域输入A7/B750/H6；在当前入口线上检验多个位置，用当前双边证据验证引导器遮挡前后的线一致性。允许活动绳在投影中覆盖卷面，不简单删除所有卷面内候选。

整段输出覆盖情况（不是准确率）：
活动绳通过检查：原{results['old']['S1_gate_passed']}/600 → 新{results['new']['S1_gate_passed']}/600。
卷面通过检查：原{results['old']['region_gate_passed']}/600 → 新{results['new']['region_gate_passed']}/600。
计数未知：原{results['old']['count_unknown']}/600 → 新{results['new']['count_unknown']}/600。
五根以内显示帧数：原{results['old']['within_five_displayed']} → 新{results['new']['within_five_displayed']}。
31.6秒及之后284帧，新活动绳通过{sum(valid_after)}帧；该段最长连续未通过{longest}帧（{longest/10:.1f}秒）。
计数未知的首个阻断原因：{json.dumps(dict(unknown),ensure_ascii=False)}。这些是按优先级分类，其他原因可能同时存在。
数值稳定性另行记录：{json.dumps(audit['display_step_diagnostics'],ensure_ascii=False)}。尚未用独立运动/根数真值判定这些上升是测量误差还是运动变化，不宣称计数严格单调。

验证：7项回归测试、3个实帧正负例检查、状态恢复的数值一致性检查通过；完整600帧审计通过。旧版33.6/52.6秒的卷面绳纹候选在新身份检查中被拒绝。关键帧人工视检记录见visual_review.json；不是逐帧人工真值标注。
完整运行41.2秒无当前测量，未输出历史ABC；41.3秒通过短时记忆搜索重新获得当前像素支持，恢复检查见actual_memory_recovery_receipt.json。
第111帧暴露原S1采样窗口无法容纳异常宽度候选的问题，已新增精确异常拒绝。成功计算路径未改变。从110帧的完整可信状态恢复，复算110帧的ABC、种子、支持点和分段结果完全一致；版本哈希与恢复边界记录在binding.json中。

剩余限制：d仍按用户约束2238.7像素，没有在本视频重新标定。活动绳跟踪恢复不等于固定d下的几何关系始终成立。部分后续帧的直线投影落到边界上方，此时拒绝计数，不强行输出0；应进一步核查该片的弯曲转折点、换向/换层及投影模型。F3和计数也保留证据不足状态。没有独立剩余根数或实体接触真值，不能报告准确率，也不能把ECW视觉候选当作实体贴边确认。本次是离线逐帧处理，不声称达到10fps实时计算速度。

主交付：{video['path']}
30～42秒重点片段：{clip}
对照曲线：comparison_timeline.png
完整记录：runs/dual_20260923_113745/frames；算法：tracker_sg2.py；逐帧审计：audit_summary.json。
'''
    (ROOT/'优化报告.txt').write_text(report,encoding='utf-8')
    deliver=[ROOT/'优化报告.txt',ROOT/'comparison.json',ROOT/'comparison_timeline.png',ROOT/'comparison_33p6.png',ROOT/'visual_review.json',ROOT/'test_receipt.json',
        ROOT/'actual_case_test_receipt.json',ROOT/'actual_memory_recovery_receipt.json',ROOT/'resume_parity_receipt.json',run/'binding.json',run/'receipt.json',run/'audit_summary.json',
        Path(video['path']),clip,run/'delivery/video_receipt.json',run/'delivery/highlight_receipt.json']+list(ROOT.glob('*.py'))+list((run/'delivery').glob('preview_*.jpg'))
    dump(ROOT/'delivery_manifest.json',dict(status='complete',frames=600,files=[dict(path=str(p),size=p.stat().st_size,sha256=e.sha(p)) for p in deliver]))
    (ROOT/'WORK_IN_PROGRESS.txt').write_text('本轮已完成。完整视频、重点片段与验证结果见 优化报告.txt 和 delivery_manifest.json。\n',encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
