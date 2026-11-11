"""Assemble results only after both complete runs and decoded videos exist."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent
SESSIONS=['dual_20260923_113745','dual_20260924_111143']
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frac(n,d):return f'{n}/{d}（{100*n/d:.1f}%）'
def windows(ev):
    return '无候选窗口' if not ev else '；'.join(f"{x['first_near']['frame']/10:.1f}–{x['last_near']['frame']/10:.1f} 秒，在线确认 {x['detection']['frame']/10:.1f} 秒" for x in ev)
def main():
    reports=[];series=[];manifest=read(ROOT/'baseline/manifest.json')
    for r in manifest['files']:assert sha(ROOT/'baseline'/r['path'])==r['sha256']
    ref=read(ROOT.parent/'LC1_boundary_BG4_local_temporal_20261107_v1/full_003/compact_validation.json')
    ecwref=read(ROOT.parent/'LC1_ECW_20261107_v1/delivery/experiment_results.json')
    reference=dict(session='dual_20260924_111500',frames=len(ref),S1_valid=sum(x['s1']['valid'] for x in ref),
        region_valid=sum(x['region']['valid'] for x in ref),count_states=dict(Counter(x['temporal']['state'] for x in ref)),
        displayed_count=sum(x['temporal']['display_N'] is not None for x in ref),ECW_valid=ecwref['v3_local_geometry_valid'],
        role='Previously completed development/reference video, not a new held-out run',
        visual_near_window_nominal_s=[24.5,30.2],physical_contact_confirmed=False)
    for s in SESSIONS:
        run=ROOT/'transfer_runs'/s;receipt=read(run/'receipt.json');audit=read(run/'audit_summary.json');v=read(run/'delivery/video_receipt.json')
        assert receipt['status']=='complete' and receipt['frame_skips']==0 and audit['complete'] and audit['checks_passed']
        assert v['decoded_frames']==audit['frames'] and v['fps']==10 and sha(v['path'])==v['sha256']
        rows=read(run/'audit_rows.json');assert len(rows)==audit['frames']
        recovery=read(run/'recovery.json') if (run/'recovery.json').exists() else None
        if recovery:assert recovery['replay_verified'] and receipt['committed_prefix_unchanged']
        reports.append(dict(**audit,events=read(run/'windows.json'),receipt=receipt,video=v,binding=read(run/'binding.json'),recovery=recovery));series.append(rows)
    result=dict(algorithm='RES-1.0',baseline_manifest_sha256=sha(ROOT/'baseline/manifest.json'),
        unchanged_fixed_parameters=True,reference=reference,new_videos=reports,
        cross_session_geometry=read(ROOT/'cross_session_geometry.json'),
        absolute_count_accuracy_validated=False,physical_contact_accuracy_validated=False,
        note='Internal validity/coverage is not accuracy; reference calibration was not re-estimated on the new sessions.')
    dump(ROOT/'experiment_results.json',result)
    lines=['RES-1.0 绳缘识别系统：固定基准与新视频迁移实验','日期：2026-11-07','',
        '本轮已保存整套基准，并完成两段新视频的全部 1201 帧检测和叠加视频输出。所有原帧顺序保留，不跳帧，不使用旧视频检测缓存，不针对新视频调整算法阈值。',
        '主结果应解释为固定参数迁移能力与失败模式；下列通过率是算法内部证据可用率，不是准确率。','',
        '五根以内显示帧数以全部视频帧为统计背景，包含本来就远离边缘的时段；它不是五根内召回率。没有检测到贴边窗口也不等于视频中没有实体接触。','',
        '一、基准',
        '名称：绳缘识别系统 RES-1.0（Rope Edge Suite），版本 1.0.0。',
        '组成：P1/B750+A7/H6 → S1 当前直线跟踪 → BG1 卷面区域 → 01/F3 黄点 → BG3 连续边界 → BG4 根数时序估计 → ECW-3 视觉贴边窗口。S2 保存但未启用。',
        '固定 d=2238.7 原图像素，4096×3000 左相机，不镜像；只显示五根范围内的数值。',
        f"源码、模型和配置冻结文件 {len(manifest['files'])} 个；manifest SHA-256：{result['baseline_manifest_sha256']}。",
        '基准压缩包：RES-1.0_baseline.zip。源码与小型配置已打包；原视频及 Python 环境不在该压缩包中。','',
        '二、原参考视频（此前完成，开发视频，不能作为本轮独立验证）',
        f"111500：{len(ref)} 帧；S1 内部门槛通过 {frac(reference['S1_valid'],len(ref))}；区域通过 {frac(reference['region_valid'],len(ref))}；五根内显示 {reference['displayed_count']} 帧。",
        'ECW-3 视觉贴边候选 24.5–30.2 秒，在线首次确认 24.9 秒。尚未证明实体接触。','',
        '三、新视频逐片结果']
    for q in reports:
        n=q['frames'];elapsed=q['receipt']['elapsed_s'];known=n-q['count_states'].get('unknown',0)
        lines += ['',q['session'],f"原视频：{q['binding']['source']}",f'帧数 {n}，原始帧率 10 fps，时长 {n/10:.1f} 秒。',
            f"S1 支撑有效：{frac(q['S1_valid'],n)}；卷面区域有效：{frac(q['region_valid'],n)}。",
            f"根数内部状态可用：{frac(known,n)}；未知：{frac(n-known,n)}；五根以内显示：{q['displayed_count']} 帧。",
            f"根数状态分布：{json.dumps(q['count_states'],ensure_ascii=False)}。",
            '相邻显示值跳动统计：'+json.dumps(q['display_step_diagnostics'],ensure_ascii=False)+'。',
            f"ECW 当前局部几何有效：{frac(q['ecw_valid'],n)}；近边证据 {q['near_frames']} 帧；分离证据 {q['far_frames']} 帧。",
            '视觉贴边候选：'+windows(q['events'])+'。',
            '区域拒绝原因（同帧可有多个）：'+json.dumps(q['region_reasons'],ensure_ascii=False)+'。',
            'ECW 活动绳局部测量拒绝原因：'+json.dumps(q['ECW_rope_reasons'],ensure_ascii=False)+'。',
            'S1 有效支撑至 d 外推长度（px）：'+json.dumps(q['S1_extrapolation_px'],ensure_ascii=False)+'。',
            'ECW 局部支撑至 d 外推长度（px）：'+json.dumps(q['ECW_extrapolation_px'],ensure_ascii=False)+'。',
            f"耗时 {elapsed/60:.1f} 分钟，包含并行实验资源竞争，整体吞吐 {n/elapsed:.3f} 帧/秒。这是离线实现，尚未达到 10 fps 实时运行。",
            f"逐帧数据审计：通过；边界 {q['curves']} 条次，交点 {q['crossings']} 个次，局部恢复 {q['recovered']} 条次。",
            f"完整叠加视频：{q['video']['path']}"]
        if q['recovery']:
            lines += [f"运行恢复：进度文件发生 Windows 占用写入中断；已逐帧重放并核对前 {q['recovery']['replay_frames']} 帧 S1、相机、根数与窗口状态后继续，前缀检测文件哈希不变。耗时包含中断与重放。数值算法未修改，只增加文件替换重试。"]
    lines += ['','四、结果判读',
        '113745 可见活动绳，首帧受左侧导向装置遮挡，S1 支撑约 88 像素，低于 400 像素门槛；随后活动绳可被跟踪。',
        '该视频早段主要区域失败来自黄色外轮缘局部证据不足。即使卷面黄灰交界可见，BG1 全部条件未通过时仍停止 F3。抽看通过帧时，左侧顶部局部也可能受入绳干扰，因此“通过”不能视作真实边界正确。',
        'ECW 在活动绳背后是灰色卷面时，常因两侧黄色背景不足而不提供局部直线测量；靠近边缘后的表现见完整视频及时间线。',
        '重要失败帧：113745 的 33.6 秒。S1 重捕获后将卷面纹理当作活动绳，绿色支撑线位于真实自由入绳下方，斜率方向也不同；光学支撑通过不等于同绳身份正确。31.9 秒还存在原图活动绳可见但入口未通过的漏检例。参见 visual_review_notes.json 与对应 preview_336.jpg、preview_319.jpg。',
        '掩膜诊断：33.6 秒重捕获种子在当前顶部边界提议下方约 59–60 像素，但旧参考搜索掩膜在这些点的值仍为 0。结合原图可见它位于卷面，提示跨视频旧掩膜失配可能使卷面纹理进入自由绳候选；未通过重新标定对照实验来确定其单独因果贡献。见 reacquisition_mask_diagnostic.json。',
        '111143 的首、中、尾等抽看画面未见明确活动绳，用于检查无明显入绳画面是否产生输出。尚未逐帧人工标注全部活动绳真值，不能把零输出直接换算成零误报率。',
        '跨视频背景配准：113745 的前三帧未能可靠确认与原参考视频的背景对应；111143 的前三帧配准位移约 1.2–1.4 像素。两者都没有独立重新验证 d。',
        '因此本轮五根内显示值均为固定旧标定条件下的算法输出，尚不能确认真实剩余根数正确。',
        '视觉贴边只表示投影接近可见黄灰交界，实体法兰内壁接触与绝对间隙未确认。','',
        '五、验证与后续基准使用',
        '1. 新视频接口与原实现通过数值对照，当前帧 F3 输入数组完全一致，BG4 与 ECW 结果在指定对照帧一致。',
        '2. 逐帧核验当前图像哈希绑定、F3 区域前置门槛、原始点归属唯一、每条边界单次交点、显示曲线与计数曲线一致、预测与相位有效期、无凭空实体接触值。',
        '3. 两段输出均重新解码验证帧数、1920×1080 尺寸和 10 fps；视频 SHA-256 写入各自 video_receipt.json。',
        '4. 原视频与冻结基准运行前后哈希一致。运行回执记录完整帧数和零跳帧。',
        '5. 下一版应优先重新建立新视频的场景/入口搜索区域，加强 P1/S1 重捕获的同绳身份、宽度和运动位置约束，验证跨视频 d 标定，并处理外轮缘遮挡/高光导致的区域失效；保持当前 RES-1.0 不变，在同一组视频上比较可用率、误报和有标注时的准确率。','',
        '附件：experiment_results.json（结构化结果）、transfer_timeline.png（完整时间线）、transfer_runs/<视频>/audit_summary.json（审计）、基准说明.txt。']
    (ROOT/'迁移实验报告.txt').write_text('\n'.join(lines),encoding='utf8')
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(3,2,figsize=(16,9),sharex='col',layout='constrained')
    for col,(q,rs) in enumerate(zip(reports,series)):
        t=np.arange(len(rs))/10;ax=axes[0,col]
        for j,(key,label,color) in enumerate([('s1_valid','S1 支撑有效','#258c70'),('region_valid','卷面区域通过','#2685b0'),('ecw_valid','ECW 局部有效','#c58b25')]):
            vals=np.array([r[key] for r in rs]);ax.scatter(t[vals],np.full(vals.sum(),2-j),s=5,c=color)
        ax.set_yticks([0,1,2],['ECW','卷面','S1']);ax.set_ylim(-.6,2.6);ax.set_title(q['session']+' / left',fontsize=12)
        ax=axes[1,col];values=np.array([np.nan if r['count_value'] is None else r['count_value'] for r in rs]);shown=np.array([r['display_N'] is not None for r in rs])
        ax.plot(t,values,color='#bac1c8',lw=1,label='内部估计，含五根以外');ax.scatter(t[shown],values[shown],s=8,c='#287ca3',label='五根内实际显示')
        ax.axhline(5,color='#bc7152',ls='--',lw=1);ax.set_ylabel('剩余根数输出（未标定真值）');ax.set_ylim(-.3,12);ax.legend(loc='upper right',fontsize=8)
        ax=axes[2,col];mapping={'unknown':0,'separated':1,'near_pending':2,'near_candidate':3,'candidate_pending_evidence':2}
        vals=np.array([mapping[r['ecw_state']] for r in rs]);ax.step(t,vals,where='post',color='#997536',lw=1.4)
        ax.set_yticks([0,1,2,3],['未知','分离证据','待确认','视觉贴边候选']);ax.set_ylim(-.4,3.4);ax.set_xlabel('视频时间（秒）')
        for j in range(3):axes[j,col].set_xlim(0,len(rs)/10);axes[j,col].grid(alpha=.15)
    fig.suptitle('RES-1.0 固定参数迁移：d=2238.7 px，未在新视频重新标定；通过率不是准确率',fontsize=15)
    fig.savefig(ROOT/'transfer_timeline.png',dpi=160);plt.close(fig)
    files=['baseline/manifest.json','baseline_archive_receipt.json','RES-1.0_baseline.zip','基准说明.txt','transfer_engine.py','run_transfer.py',
        'review_transfer.py','audit_transfer.py','finalize_transfer.py','resume_transfer.py','io_retry_test_receipt.json','visual_review_notes.json','reacquisition_mask_diagnostic.json','parity/receipt.json','experiment_results.json','迁移实验报告.txt','transfer_timeline.png']
    for s in SESSIONS:
        files += [f'transfer_runs/{s}/{p}' for p in ['binding.json','receipt.json','windows.json','audit_summary.json','delivery/video_receipt.json',f'delivery/RES10_{s}_full.mp4']]
        if (ROOT/'transfer_runs'/s/'recovery.json').exists():files.append(f'transfer_runs/{s}/recovery.json')
        files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'transfer_runs'/s/'delivery').glob('preview_*.jpg'))]
    dump(ROOT/'delivery_manifest.json',dict(algorithm='RES-1.0',files=[dict(path=p,bytes=(ROOT/p).stat().st_size,sha256=sha(ROOT/p)) for p in files]))
    print('COMPLETE: report, full timeline, result JSON and delivery manifest')
if __name__=='__main__':main()
