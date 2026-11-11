from common20 import *

def add(id,frame,artifact,findings):
    q=dict(id=id,frame=frame,artifact=artifact,findings=findings,calibration='final_conditional_geometry_estimate',review_type='assistant_visual_inspection_not_independent_truth')
    entries[:] = [x for x in entries if not (x.get('id')==id and x.get('frame')==frame and x.get('artifact')==artifact)]
    entries.append(q)

r=read(ROOT/'visual_review.json');entries=r['entries']
add('dual_20260923_113029__left',306,'delivery/dual_20260923_113029__left/review/000306.jpg','黄色法兰碎片合并和当前宽范围凸曲线恢复后，计数列覆盖真实卷面。可见 19 条区间内绳沟，拟合曲线与图上绳纹相符；未见明显重复类别。模型中心距约 18.77 圈，非独立真值。')
add('dual_20260923_113029__left',41,'delivery/dual_20260923_113029__left/review/000041.jpg','原失败 ROI 仅保留左侧半幅卷面。扩大当前侧面黄色碎片的提议范围后恢复完整计数区域，图中 11 条绳沟与可见排列一致；约 10.81 圈为当前条件估计。')
add('dual_20260923_113745__left',248,'candidate_dual_20260923_113745__left_000248.jpg','新 CE4 当前绳体轴沿可见活动绳，计数区间有约 3 条沟槽；约 2.82 圈与图像几何一致。旧轴只是上游外推，不适合作为否决新身份的真值。采用经当前证据核验的候选。')
add('dual_20260923_113745__left',405,'candidate_dual_20260923_113745__left_000405.jpg','工具遮挡且候选仍需长距离上游直线外推；黄色线在绳转弯区域的投影不能证明真实入卷位置。该候选未进入交付计数，保留当前证据不足状态。')
add('dual_20260923_101517__right',580,'delivery/dual_20260923_101517__right/review/000580.jpg','当前活动绳已近可见卷面界限；首几条沟槽无法独立构成完整相位链。近边不确定状态与图像相符，没有把中心距约 1.14 圈当成实体接触证据。')
add('dual_20260923_101643__right',570,'delivery/dual_20260923_101643__right/review/000570.jpg','活动绳明显远离上侧边界，输出大于五圈合理。图中计数段检测 29 条沟槽，但存在缺口，显示明确为几何估计；未把该结果当作精确整数真值。')
add('dual_20260923_113745__left',310,'delivery/dual_20260923_113745__left/review/000310.jpg','31 秒已恢复当前绳轴和 F3 计数，图中约一条绳沟位于两端点之间，显示约 1.4 圈及模型范围。该处不再因可见边界横向范围限制而整列丢失。')
add('dual_20260923_113745__left',370,'delivery/dual_20260923_113745__left/review/000370.jpg','37 秒工具覆盖接近位置，仍有下方当前 F3 沟槽。被遮挡的边界用蓝虚线，显示约 0.3 圈同时给出跨零且较宽的范围，贴边状态为不确定；不能从此帧确认实体接触。')
add('dual_20260923_113745__left',397,'delivery/dual_20260923_113745__left/review/000397.jpg','换向附近工具遮挡，活动绳投影在边界模型上方，不能把负中心距改成零圈。投影冲突提示与图像一致；该段不是实体接触确认。')
add('dual_20260923_113305__right',580,'delivery/dual_20260923_113305__right/review/000580.jpg','原图上活动绳与卷面边界接近；当前区间内一条沟槽，输出约 1.4 圈及跨较宽范围的误差。工具在更低处，未被计为活动绳。')
add('dual_20260924_111143__right',140,'delivery/dual_20260924_111143__right/review/000140.jpg','该阶段开始时已经近边，缺少满足最终五圈约束的参考帧；固定 d 保持未标定。白色诊断列仅用于当前局部近边观察，画面确实近边，但没有给固定列根数。')
add('dual_20260924_111143__right',180,'delivery/dual_20260924_111143__right/review/000180.jpg','工具和活动绳接近边界，蓝虚线为遮挡后的模型范围。输出跨零的较宽区间及近边不确定，没有确认实体接触。')
add('dual_20260924_111500__left',300,'delivery/dual_20260924_111500__left/review/000300.jpg','活动绳在边界附近，卷面沟槽在下方清楚可见，端点处仍受活动绳自身遮挡及曲线模型偏差影响。约 1.1 圈的范围跨零，保持近边不确定，不能确认实体贴边。')
dump(ROOT/'visual_review.json',r);print('Visual entries',len(entries))
