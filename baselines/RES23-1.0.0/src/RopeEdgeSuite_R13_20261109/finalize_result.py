"""Publish completed experiment, including its negative counting result."""
import sys,subprocess,json
from collections import Counter
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from common import *

def main():
    run=ROOT/'runs/dual_20260923_113745';out=run/'delivery'
    receipt=read(run/'receipt.json');audit=read(run/'audit_summary.json');video=read(out/'video_receipt.json')
    assert receipt['status']=='complete' and receipt['frames']==600 and audit['passed'] and audit['frames']==600 and video['decoded_frames']==600
    assert read(ROOT/'test_receipt.json')['passed']
    new=read(run/'compact.json');old=read(RUN12/'compact.json');cal=read(ROOT/'d_calibration/receipt.json');hold=read(ROOT/'holdout_receipt.json')
    oldgeo=read(PREV/'geometry_review_20261109/geometry_rows.json')
    negatives=[r['frame'] for r in oldgeo if r['s1_valid'] and r['region_valid'] and r['gap_px']<0]
    modes=Counter(r['F3_mode'] for r in new);reasons=Counter(why for r in new for why in r['geometry_reasons'])
    summary=dict(frames=600,old=dict(tracking_pass=sum(r['s1_valid'] for r in old),region_pass=sum(r['region_valid'] for r in old),
        count_numeric=sum(r['count_value'] is not None for r in old),within_five=sum(r['display_N'] is not None for r in old),unknown=sum(r['count_state']=='unknown' for r in old)),
        new=dict(tracking_pass=sum(r['tracking_valid'] for r in new),region_pass=sum(r['region_valid'] for r in new),
            foreground_rejected=sum(r['foreground_rejected'] for r in new),conditional_projection_pass=sum(r['geometry_valid'] for r in new),
            projection_pass_frames=[r['frame'] for r in new if r['geometry_valid']],
            count_numeric=sum(r['value'] is not None for r in new),within_five=sum(r['display_N'] is not None for r in new),
            unknown=sum(r['count_state']=='unknown' for r in new),certified_count_frames=0),
        original_negative_projection_frames=len(negatives),negative_projection_newly_accepted=sum(new[i]['geometry_valid'] for i in negatives),
        original_negative_foreground_rejected=sum(new[i]['foreground_rejected'] for i in negatives),
        F3_modes=dict(modes),projection_reasons=dict(reasons),holdout_scored=hold['scored'],holdout_matched=hold['matched'],
        d_calibration_valid=cal['calibration_valid'],d_eligible_frames=cal['eligible_frames'],
        interpretation='rejection coverage is not root-count accuracy; continuous root counting has NOT been restored')
    dump(ROOT/'comparison.json',summary)
    clip=out/'RES13_BG6_GV1_113745_30s-42s.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-ss','30','-i',video['path'],'-t','12',
        '-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(clip)],check=True)
    cap=cv2.VideoCapture(str(clip));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(1080,1920,3);n+=1
    fps=cap.get(5);cap.release();assert n==120 and fps==10
    dump(out/'highlight_receipt.json',dict(path=str(clip),frames=n,fps=fps,source_time_s=[30,42],sha256=sha(clip)))
    # Keep before/after evidence in original image coordinates.
    sys.path.insert(0,str(PREV));import render_sg2
    from render_result import annotate
    bind=read(run/'binding.json');cap=cv2.VideoCapture(bind['source'])
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    sheet=Image.new('RGB',(1800,1250),'#111b25');dr=ImageDraw.Draw(sheet)
    dr.text((20,10),'RES-1.2：原检测',font=font,fill='white');dr.text((920,10),'RES-1.3：边界与交点分别检查',font=font,fill='white')
    for j,i in enumerate([336,369,396]):
        cap.set(1,i);ok,im=cap.read();assert ok;oldr=record(i);r=record(i,run)
        for k,a in enumerate([render_sg2.annotate(im,oldr),annotate(im,r)]):
            tile=Image.fromarray(cv2.cvtColor(a[1950:2450,900:2550],cv2.COLOR_BGR2RGB)).resize((880,300));sheet.paste(tile,(10+k*900,60+j*390))
        reason='导杆/宽亮带遮挡：边界拒绝，F3暂停' if new[i]['foreground_rejected'] else '实测段不足以支持固定 d 交点，暂停计数'
        dr.text((20,367+j*390),f'{i/10:.1f}s：{reason}',font=font,fill='#e6bb6b')
    cap.release();sheet.save(ROOT/'before_after_review.jpg',quality=95)
    report=f'''RES-1.3 / BG6-GV1 实验与验证结果

本轮结论：导杆被误当卷面边界的已核验案例得到拒绝，但可靠的连续剩余根数没有恢复。
这是边界与测量资格验证版，不应将“少报错”或“暂停输出”解释为测距精度已经提高。

输入：113745左相机原视频，600帧，4096×3000，10fps。输出600帧、60秒、1920×1080。
旧版与基准源码保留。SG2沿用同一源视频、同一像素帧的已有测量，便于单独评估新增BG6和GV1。

具体修改：
1. BG6在当前黄灰边界候选下方，按绳距检查亮带宽度及横向连续性。单列宽亮带超过0.85个先验绳距，连续跨度至少1.5个绳距时视作前景干扰；计数列及其邻域被覆盖则拒绝目标边界，不提供替代边界，不绘制为已接受青线。
2. 远离计数列的干扰列从卷面ROI中剔除；ROI变化帧重新执行原01/F3和分组曲线。只有原图像素、ROI和SG2均相同的帧才复用同帧F3。边界拒绝帧停止F3。
3. GV1将“活动绳跟踪有效”和“在d处可计数”分开。限制超出实测区域的外推距离为12个先验绳距且不超过实测跨度0.75倍；将点分为空间块、分成三段独立拟合，检查远端预测敏感度，同时检查边界误差和交点相容性。阈值为明确的开发约束，尚未由独立真值验证最优性。
4. 无效交点停止计数并清空连续计数状态；不能把负投影截为0。有效观测的权重计入直线和边界敏感度。d未标定，所有可能数值只能作为参考d条件估计，不能认证为绝对剩余根数。

完整600帧统计：
活动绳跟踪通过：{summary['old']['tracking_pass']} → {summary['new']['tracking_pass']}（SG2保持一致）。
卷面/计数边界通过：{summary['old']['region_pass']} → {summary['new']['region_pass']}。
新前景遮挡检查拒绝：{summary['new']['foreground_rejected']}帧；与其他拒绝原因可能重叠。
有条件交点通过：{summary['new']['conditional_projection_pass']}帧，帧号{summary['new']['projection_pass_frames']}。
连续计数数值输出：{summary['old']['count_numeric']} → {summary['new']['count_numeric']}帧。
计数未知/暂停：{summary['old']['unknown']} → {summary['new']['unknown']}帧。
旧67个投影为负的帧，新版接受为有效交点的数量：{summary['negative_projection_newly_accepted']}；其中{summary['original_negative_foreground_rejected']}帧触发前景遮挡。
F3处理方式：{json.dumps(dict(modes),ensure_ascii=False)}。
交点拒绝原因（可重叠）：{json.dumps(dict(reasons),ensure_ascii=False)}。

d重新标定实验：
预先选择20/60/100为开发候选，40/80/120为留出候选，实际执行现有P1/B750/H6，使用同帧卷面搜索区域。
严格的远边筛选还要求投影差减去2个先验绳距后仍至少5根；6个候选中只有部分通过此保守筛选，不声称全是已知远边真值。
没有候选同时通过完整当前绳路、局部折弯质量及远边条件，因此没有开展无依据的联合d拟合，也没有得到通过留出验证的新d。参考值仍2238.7，不是重新标定结果。
另在第20和120帧执行原始S2补证：它在待补的自由段采样中分别提供32/67和79/101个邻近当前支持，但不能证明全部缺口为同一根活动绳；没有把S2固定直轴当成完整曲线或新的d。

验证范围：
8项回归测试通过；600帧来源、ROI、F3复用条件、坐标、显示和门控审计通过。
固定参数后，对11个留出原图样本先作独立画面视检：8个清晰、2个前景干扰、1个模糊。10个可判读样本与新前景检查一致；模糊样本不计入此匹配数。
这属于助手的画面抽查，不是人工像素标签或根数真值，也不是跨视频泛化准确率。

失败和限制：
本轮没有恢复连续计数。短外推合格帧稀疏，当前连续计数器需要足够的连续相位证据；放宽门槛会重新接受尚未验证的长外推，不能因此宣称完成测距。
BG6主要覆盖此次明显宽亮带前景，低对比度、暗色或很细的遮挡仍需进一步验证。可见黄灰交界仍不是经过物理标定的工字轮内侧接触边。
GV1中的±50px d扰动是灵敏度情景，非d标定置信区间。空间分块与误差包络也未经过绝对误差校准。
下一步的实质瓶颈是获得入卷附近的当前活动绳观测，并在该片验证d或局部曲线模型；只放宽阈值、平滑计数或改d以消除负数均不能解决。

完整验证视频：{video['path']}
30～42秒片段：{clip}
前后对照图：{ROOT/'before_after_review.jpg'}
源代码：region_bg6.py、geometry_gv1.py、run_full.py；实验数据：runs/dual_20260923_113745。
'''
    (ROOT/'验证报告.txt').write_text(report,encoding='utf-8')
    # Independently recheck previously delivered files to ensure preservation.
    previous=read(PREV/'delivery_manifest.json')
    assert all(Path(x['path']).stat().st_size==x['size'] and sha(x['path'])==x['sha256'] for x in previous['files'])
    files=list(ROOT.glob('*.py'))+[ROOT/'验证报告.txt',ROOT/'comparison.json',ROOT/'before_after_review.jpg',ROOT/'validation_plan.json',
        ROOT/'visual_holdout_labels.json',ROOT/'holdout_receipt.json',ROOT/'holdout_raw_contact_sheet.jpg',ROOT/'test_receipt.json',
        ROOT/'d_calibration/receipt.json',ROOT/'d_calibration/s2_gap_probe/receipt.json',run/'binding.json',run/'receipt.json',
        run/'audit_summary.json',run/'compact.json',Path(video['path']),clip,out/'video_receipt.json',out/'highlight_receipt.json']+list(out.glob('preview_*.jpg'))
    dump(ROOT/'delivery_manifest.json',dict(status='complete',frames=600,continuous_count_restored=False,
        previous_delivery_files_unchanged=True,files=[dict(path=str(f.resolve()),size=f.stat().st_size,sha256=sha(f)) for f in files]))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
