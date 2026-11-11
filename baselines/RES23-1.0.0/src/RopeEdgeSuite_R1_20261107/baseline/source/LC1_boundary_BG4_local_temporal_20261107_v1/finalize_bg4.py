"""Delivery receipt and concise Chinese report from checked results."""
from datetime import datetime
from bg4_common import *

def main():
    out=ROOT/'delivery_final';m=read(out/'metrics.json');v=read(out/'video_receipt.json');a=read(out/'geometry_temporal_audit.json')
    assert v['status']=='completed' and v['decoded_frames']==601 and a['failures']==0
    q=m['approach_121_270'];f=m['full'];forecast=m['held_out_forecast_deviation']
    text=f'''BG4 v1：局部证据复核与有限时序计数
最终运行：full_002；最终交付目录：delivery_final
生成日期：{datetime.now().isoformat(timespec='seconds')}

输入及适用范围
源视频：{SOURCE}
601 帧，10 fps，60.1 秒，原始尺寸 4096×3000。
固定原图列 d=2238.7 px；本版未重新估计 d。仅在估计位于 0–5 个绳距时显示剩余距离。
使用同一原图像素绑定的 BG1 卷面门控、S1 直线、用户提供的 01 绳边观测及 BG3 黄点类别与曲线。
BG1/BG3 原始数据、权重、算法及上一版结果均未覆盖。

本版方法
1. 保留 BG3 已接收的整条连续曲线。对因远处越界、残差或相邻关系失败的类别，在 d 左右 8 个参考绳距内复核黄点。
   局部找回要求至少 12 个点、d 两侧各至少 3 点、跨度至少 4 个绳距、90% 残差不超过 0.22 个参考绳距。
   拟合只保留包含 d 的连续有效区间；不得越出卷面模型范围、与邻线交叉、超出保留黄点的首末端。
   这不会把整条错误曲线重新认作正确；d 处的遮挡与无证据长缺口仍可导致拒绝。
2. 当前相位可用时使用相位距离；存在局部补全或插值时降为估计。仅有几何距离时，要求当前局部间距支持与近期相位锚点。
   几何换算使用近期“相位距离－几何距离”的中位偏差校正，不能从无锚点状态自行启动，也不能无限维持数值。
3. 使用因果状态 [剩余绳距, 变化速度] 预测与校正，拒绝突跳；纯预测最长 0.3 秒，只有弱几何证据时距相位锚点最长 1.0 秒。
   超时或内部状态标准差超过 0.65 绳距，返回未知。此标准差是模型内部量，不是已标定的真实误差界限。
4. 经近期图像观测确认接近方向后约束输出；连续反向证据触发重新确认。换向、卷面几何/角度显著变化均可要求重新获取。
   相机监测改为直接对齐首帧静止背景，避免逐帧累加漂移；确认相机移动后固定 d 失效。没有声称已能可靠识别所有换层事件。
5. 实测校正／估计校正／短时预测／未知分别显示；预测时不补画当前缺失的绳边。
   “实测校正”是当前图像更新后的滤波值，不是真实根数标注。绿色为图像支持段，橙色为模型补全段。
   每条当前边界在 d 只贡献一个交点，绘制和计数使用同一曲线。类别编号只在本帧有效。

相同视频的结果（没有人工真实根数标注）
全片：旧版未知 {f['old_unknown']}/601 帧，新版未知 {f['new_unknown']}/601 帧。
全片状态：{f['states']}
五圈范围内显示：旧版 {f['old_displayed']} 帧，新版 {f['new_displayed']} 帧；新版状态构成：{f['displayed_states']}。
预先选定的接近段 121–270 帧，共 150 帧（它是审查时间段，不是人工标定的精确五圈区间）：
  未知：{q['old_unknown']} → {q['new_unknown']} 帧。
  五圈内显示：{q['old_displayed']} → {q['new_displayed']} 帧。
  两版本均有数值的相邻帧配对数：{q['paired_adjacent_count']}。
  相邻跳变量中位数：{q['old_paired_abs_delta'].get('median')} → {q['new_paired_abs_delta'].get('median')} 绳距。
  相邻跳变量 95% 分位：{q['old_paired_abs_delta'].get('p95')} → {q['new_paired_abs_delta'].get('p95')} 绳距。
  上升次数：{q['old_paired_increases']} → {q['new_paired_increases']}（同一配对集合）。
实际最长纯预测间隔：{m['maximum_prediction_age_s']:.3f} 秒。
确定性因果回放：{m['deterministic_causal_replay_passed']}。
后段 171–600 帧按固定规则屏蔽连续三帧观测，用过去状态预测、再与被屏蔽的当前图像相位估计比较：
  可比较 {forecast['n']} 个预测点；绝对偏差统计 {forecast}。
  这个检验衡量视频内部一致性，并非真实根数精度或独立视频泛化精度。

验证
16 项针对性自动测试通过，覆盖交点一致性、遮挡、长缺口、短时缺测/超时、异常值、换向、相机移动、抖动与间距误判。
全部 601 帧几何和状态审计通过：{a}。
最终 MP4 已逐帧解码检查：601 帧，1920×1080，10 fps；逐帧匹配源图像 SHA256。
视频 SHA256：{v['sha256']}
处理为离线回放，没有证明当前实现能达到现场 10 fps 实时速度。

仍然需要解决
显示更连续、跳动更小，不等于工字轮边界或真实根数已经更准确。当前边缘仍来源于 BG1，边界偏移会传递到计数。
尚缺人工边界/活动绳位置/真实剩余根数标注与触边零点标定。长遮挡、可靠观测持续缺失和真实几何突变时保留未知。
下一项有价值的验证是对接近边缘的视频帧人工标注，再分别测量边界误差、绳距误差、剩余根数误差与换向延迟。

文件与复现
交付视频：BG4_local_temporal_full.mp4；对比图：comparison_timeline.png。
逐帧记录：../full_002/frames；冻结代码与绑定：../full_002/snapshot、binding.json。
指标、审计、视频校验：metrics.json、geometry_temporal_audit.json、video_receipt.json。
新跑全流程：使用工作区 .venv_lc1/Scripts/python.exe 执行 run_bg4.py --name <新的目录名> --workers 4。
重放此次空间证据：replay_reference.py（拒绝覆盖已有 full_002，复现时应另设新目录）。
完整视频导出与审计：export_video.py、audit_bg4.py，均对应最终 full_002。
full_001 是相机累计漂移误报的审查版本；delivery 是已停止的中间导出，均不作为交付。
'''
    report=ROOT/'BG4_算法与结果.txt';report.write_text(text,encoding='utf8')
    files=[report,out/'BG4_local_temporal_full.mp4',out/'comparison_timeline.png',out/'metrics.json',out/'geometry_temporal_audit.json',out/'video_receipt.json',ROOT/'full_002/binding.json',ROOT/'full_002/receipt.json']
    files+=list(out.glob('preview_*.png'))
    dump(out/'delivery_manifest.json',dict(status='complete',final_run='full_002',source=str(SOURCE),source_sha256=SOURCE_SHA,
        unit_tests=dict(unique_test_count=16,passed=True),files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},
        finalizer_sha256=sha(__file__),absolute_count_accuracy_validated=False))
    print('DELIVERY',report,flush=True)
if __name__=='__main__':main()
