"""Write the experiment report and verify delivery bindings after visual review."""
import hashlib,json
from pathlib import Path
from local_measurement import ROOT,BG4,SOURCE,read,dump
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'delivery';stats=read(out/'experiment_results.json');videos=read(out/'video_receipt.json');tests=read(out/'test_receipt.json')
    assert tests['passed'] and tests['tests']==16
    assert sha(SOURCE)=='ead815ac73239dec6fcbd141ffb2873fb2ac575dfb79347a036ee877b65255f8'
    assert stats['total_frames']==601 and len(stats['events'])==1
    for v in videos['videos']:assert Path(v['path']).exists() and sha(v['path'])==v['sha256']
    rows=read(ROOT/'v3_full/records.json');decisions=read(out/'frame_decisions.json')
    assert len(rows)==len(decisions)==601
    for i,(r,s) in enumerate(zip(rows,decisions)):
        assert r['frame']==s['frame']==i and not r['physical_contact_confirmed']
        if s['evidence']['near']:assert r['valid'] and s['evidence']['local_valid']
    report='''ECW-3 活动绳视觉贴边时间窗口实验
日期：2026-11-07

结论
在 dual_20260924_111500/left_DA9627065.avi 的完整 601 帧中，得到一个视觉贴边候选窗口：
  候选证据首帧：245，视频 24.5 秒，主机时间 24.507436 秒。
  按时间顺序确认候选：249，视频 24.9 秒，主机时间 24.914066 秒。
  最后近边证据：302，视频 30.2 秒，主机时间 30.196647 秒。
  58 个采样帧中，57 帧有当前近边证据；300 帧（30.0 秒）未通过局部边界门槛，明确保留为本帧未确认。
  30.5 秒开始出现新的分离证据；到 30.8 秒，已有三帧分离证据可供事后复核。
  因果状态在 30.6 秒证据过期，不会利用后续帧回填接触状态。

这一窗口表示投影图像中的贴边候选，不等于已验证的钢丝绳与金属法兰实体接触。
缺少实体内边缘/接触零点标定、接触真值和独立测试视频，因此不报告物理接触准确率。
57/58 是候选时段的图像证据覆盖率，不是准确率。

三轮实际试验与改进
v1：尝试亮度脊线动态规划、排除活动绳走廊后的多项式边界。
    原图对照发现：150/220 等帧的亮度路径会跳到邻绳；260 等帧出现背景脊线误跟踪；
    320 帧丢失右侧支撑后，四次曲线向错误方向外推。该方案未作为最终窗口依据。
    v1/review.jpg 与 v1/records.json 保留失败案例，避免只展示成功结果。
v2：只接受两侧均有黄色背景的中性灰活动绳自由段，重新做局部直线拟合；
    不把灰色卷面上相似的亮绳作为自由段支撑。宽度来自当前自由段，不冒充 F3 绳间节距。
    入卷侧重新检测黄色/灰色跃迁，输出拟合残差、支撑范围和外推距离。
    全片回测发现局部跃迁点仍可能选错相邻层，导致边界门槛不通过。
v3：先沿短局部区域关联色彩跃迁点，再做鲁棒二次曲线拟合。
    要求 d 两侧以及 d 附近均有当前图像点，且残差达标。
    不跨大段缺失拟合整条法兰，不将预测补线当成实测。
    结合多帧持续证据、进入/离开迟滞、相机状态和证据过期规则，形成 ECW-3。
    在 20.0–35.0 秒共 151 帧中，局部几何有效帧由 v2 的 99 帧增加至 v3 的 126 帧。
    在最终候选窗口内，有效几何由 46/58 增加至 57/58。这说明覆盖改善，不能单独证明边界准确率改善。

测量定义与实现
固定列 d = 2238.7，原图左相机坐标，4096×3000，不镜像。
自由段：以同帧 S1 轴作搜索条件，在 x=900…2100 的剖面中寻找中性灰绳，要求上下两侧为黄色；
连续支撑中断则停止，最近 500 像素支撑作鲁棒线拟合，300/400/500/650 像素窗口作敏感性检查。
边界：BG1 当前可见轮廓仅用于限定搜索区域；重新从当前图像取 d±160 像素的色彩跃迁点，
将邻近列的候选点关联，并要求中心附近、两侧包围和残差条件通过。
q = y_local_free_axis(d) - y_current_visible_interface(d)。q 是中心线与可见界面的投影差。
q 的有效近边范围为 [-0.60w, 0.60w]，w 为自由段的光学纵向绳宽；敏感度界不大于 0.65w。
与同帧 S1/BG1 粗几何作一致性检查；这些来自同一相机的信号有相关性，不作为独立投票。
自由段不能直接观察到 d 时，橙色虚线明确显示外推。候选窗口内原 S1 外推中位数 781.2 像素，
局部拟合后中位数 594.7 像素；仍有 410.7–798.7 像素外推，因此尚未消除关键系统误差。
敏感度界含拟合窗口变化、残差和外推惩罚，是工程诊断量，不是标定后的统计置信区间。
相机背景运动门槛使用现有 BG4 同帧固定参考检测结果。

为什么没有直接输出实体接触间隙
黄色/灰色可见分界在活动绳与卷面轮廓重合时可能包含活动绳本身。
这时用中心线差减去半根绳宽会得到负的“间隙”，无法解释为真实穿透或真实接触。
v2 保留这个失败诊断，v3 不再将它解释为实体表面间距。
最终输出的 physical_gap_px 始终为 null，physical_contact_confirmed 始终为 false。
要验证实体贴紧，下一步应标定真实内法兰边界，并使用同步另一视角或清晰的已知接触片段复核。

P1/H6 的当前能力核查
对 220、245、260、290、320 帧重新运行现有真实 B750 权重与 H6 当前帧流程。
五帧均未通过完整弯绳支撑门槛：245 帧没有可接受的 A7 入口，其余四帧存在曲线/过渡支撑不足或断口。
因此 P1/H6 未用于宣布已识别完整弯绳或实体接触，也没有把它的诊断连接当作真实活动绳。
p1_probes/ 保存原始记录与可得到的热力图。没有训练新模型。
新增 ECW 局部测量与时序算法为传统算法；复用了既有 S1/BG4 输出，完整既有流程仍含原 P1 神经模型。

阈值、数学拟合与时间不确定性
近边阈值取 0.45、0.60、0.75 个光学绳宽，进入持续要求取 0.30、0.40、0.50 秒，共 9 组检查。
所有组合均只得到一个候选窗口；证据起点为 24.5 秒，最后近边证据为 30.1 或 30.2 秒；
在线确认时间为 24.9 或 25.1 秒。起点的显示时刻与确认时刻是两个不同量。
最近明确非近边证据为 23.9 秒，故进入的证据包围区间为 [23.9,24.5] 秒；
最后近边与首次新分离证据之间为 [30.2,30.5] 秒。这些是采样/门槛证据区间，不是物理接触置信区间。
另以 soft_l1 鲁棒损失拟合“下降—平台—上升”三段模型，平台约为 24.87–29.00 秒。
逐次去掉一秒数据块重拟合，平台起点范围 24.60–25.20 秒、终点范围 28.57–29.20 秒。
这一简单线性模型会把后续阶梯式变化摊成斜坡，平台结束早于直接近边证据约 1.2 秒，
所以拟合结果只作为诊断，没有覆盖当前图像支持的窗口。拟合使用未来帧，不能冒充在线检测。

复核和验证
查看了 v1/v2/v3 原图与叠加对照、预选时刻原图、全片证据时间图，以及交付视频进入、
缺失、离开等关键帧的实际画面。预选帧属于同一开发视频的辅助复核，不是独立测试集。
助手的视觉复核记录不冒充人工专家真值。
19.5、35.0、50.0 秒明显分离的复核帧均未触发候选；25.2、27.5、29.8 秒为近边候选。
23.2、30.7 秒原图的精确接触状态仍有歧义，未用这些帧计算准确率。
16 项测试全部通过，包括：已知斜率/边界的合成图、无绳、灰色背景误跟踪、单帧误报、
缺失证据过期、相机变化、真实时间间隔、缺失 F3 不等于零，以及前缀结果不受未来帧影响。
完整输出视频逐帧解码 601 帧，短片 150 帧，均为 1920×1080、10 fps；哈希记录见 video_receipt.json。
输入视频未修改，既有 BG4 版本未替换。视频中的 BG4 根数是原版辅助估计，本实验不宣称重新验证了根数真值。
v3 全片测量耗时约 70.5 秒，使用缓存的同帧 S1/BG4 输入；不能据此宣称端到端实时性能。

交付内容
ECW3_full_601frames.mp4：原片 0–60.0 秒的完整 601 帧检测回放。
ECW3_near_edge_20-35s.mp4：20.0–34.9 秒的 150 帧重点片段。
timeline.png：局部几何、数学拟合与时序证据对照。
experiment_results.json：统计、窗口、参数敏感性与复核记录。
frame_decisions.json：601 帧当前证据和因果状态。
extrapolation_audit.json：候选窗口原 S1 与局部拟合外推长度审计。
test_receipt.json、video_receipt.json、manifest.json：测试、视频解码与文件校验。

复现（工作目录：E:\\study\\program\\济宁\\precise_arrangement）
使用 .venv_lc1/Scripts/python.exe。
按顺序运行本实验目录的 run_measurements.py、run_v3.py、analyze_experiment.py、
test_windows.py、test_measurements.py、render_delivery.py、finalize.py。
probe_p1.py 为五个关键帧的单独复核；local_measurement.py 和 ecw_v2.py/ecw_v3.py 可生成各轮开发对照。
全部依赖和原始帧绑定使用本实验同目录代码中的明确路径；不读取保留测试视频。
'''
    (out/'实验报告.txt').write_text(report,encoding='utf8')
    # Bind data, code, baseline, test and delivery. Manifest excludes itself.
    files=list(ROOT.glob('*.py'))+list(ROOT.glob('*.json'))+[ROOT/'v1/records.json',ROOT/'v2/records.json',ROOT/'v3/records.json',
        ROOT/'v2_full/records.json',ROOT/'v3_full/records.json',ROOT/'p1_probes/summary.json']
    files+=list(out.glob('*.json'))+list(out.glob('*.txt'))+list(out.glob('*.mp4'))+list(out.glob('*.png'))
    files=[p for p in files if p.name!='manifest.json']
    manifest=dict(source=str(SOURCE),source_sha256=sha(SOURCE),baseline_compact=str(BG4/'full_003/compact_validation.json'),
        baseline_sha256=sha(BG4/'full_003/compact_validation.json'),files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    dump(out/'manifest.json',manifest)
    for row in manifest['files']:assert sha(ROOT/row['path'])==row['sha256']
    print('delivery verified',len(manifest['files']),'files; 601 frame decisions; 16 tests; 2 decoded videos')
if __name__=='__main__':main()
