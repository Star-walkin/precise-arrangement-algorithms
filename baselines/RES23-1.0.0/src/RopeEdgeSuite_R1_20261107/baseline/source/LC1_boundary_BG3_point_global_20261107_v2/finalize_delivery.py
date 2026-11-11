"""Publish a local report only after full analysis, audit and video decode succeed."""
import json,shutil
from inputs import ROOT,sha,dump

def read(p):return json.loads(p.read_text(encoding='utf8'))
def main():
    out=ROOT/'delivery';run=ROOT/'full_001'
    metrics=read(out/'metrics.json');audit=read(out/'geometry_audit.json');video=read(out/'video_receipt.json')
    receipt=read(run/'receipt.json');binding=read(run/'binding.json')
    assert receipt['status']=='completed' and receipt['completed']==601
    assert audit['status']=='passed' and not audit['issues'] and audit['frames']==601
    assert video['status']=='completed' and video['frames']==601
    for p,h in binding['code_sha256'].items():assert sha(p)==h,p
    c=metrics['counts'];m=metrics['matched_jumps']['BG2']
    comparison='共同有效的相邻帧不足，不能比较跳动。'
    if m['pairs']:
        comparison=(f"在第 121–270 帧中两版均能显示数值的 {m['pairs']} 对相邻帧上：\n"
            f"  |相邻帧根数差|中位数：BG2 {m['old']['absolute_delta_median']:.4f} → BG3 v2 {m['new']['absolute_delta_median']:.4f} 个绳距。\n"
            f"  |相邻帧根数差|95% 分位数：BG2 {m['old']['absolute_delta_p95']:.4f} → BG3 v2 {m['new']['absolute_delta_p95']:.4f}。\n"
            f"  仍出现数值增加：BG2 {m['old']['increasing_pairs']} 次，BG3 v2 {m['new']['increasing_pairs']} 次。\n"
            "这只是相同有效样本上的稳定性比较，不能转换成真实根数准确率。")
    text=f'''BG3 v2：原始黄点全局归类与连续边界拟合

本次完成的修改
先使用用户提供的 01 传统算法获取同一帧的原始黄点，再直接在点层面归类。每个通过校验的类别只有一条连续模型；内部点缺失不会把这一类别重新拆成短线段。
本次代码与输出均位于独立目录：{ROOT}

输入与处理范围
视频：{binding['source']}
视频 SHA256：{binding['source_sha256']}
全部 601 帧，10 fps，60.1 秒，原始尺寸 4096×3000。
采用同像素绑定的 BG1 卷面、黄色法兰、S1 记录；本轮改动聚焦于卷面内的绳槽边界归类与拟合，没有重新训练或运行 P1/H6。
d 保持用户指定的原图 x=2238.7。内部处理缩放为 0.5，视频和输出曲线坐标还原至原图。
每个有效输入均核对视频像素、卷面 mask 和原始黄点；已有缓存仅复用相同帧的原始点及图像证据场，不复用任何旧拟合曲线或分类结果。

算法步骤
1. 卷面门控：保留现有逐帧边界可信性条件，未通过的帧不输出绳边或距离。
2. 坐标展平：利用原始点的局部轨迹估计公共弧度与横向绳距变化，减少透视缩放导致的相邻绳槽混类。局部轨迹编号只用于公共形状估计，不作为最终类别。
3. 点归类：在展平坐标中初始化按上下顺序排列的类别，按当前曲线距离反复分配原始点；一个点最多属于一类，同类同一采样列最多保留一个点。归属不清楚的点不强行分配。
4. 每类拟合：采用鲁棒 B 样条残差模型和二阶平滑惩罚；每类至少 16 点、横向跨度至少 6 个参考绳距。抑制离群点后重新拟合。
5. 邻绳约束：必要时进行小幅约束重拟合，保持边界次序与最小间距；超过允许偏移或仍冲突的类别整类拒绝，不通过拆成碎片掩盖冲突。
6. 连续输出：每类从最左有效点延伸到最右有效点，在内部连续采样；不超出这一观测跨度。缺点段也保持同一类别与同一模型。
7. 统一计数：画面中的这条模型与 x=d 相交得到计数交点，每类每帧只贡献一个交点；不使用另一套短线段计数。

绿线和橙线的含义
绿线：当前位置有当前图像证据支持的模型部分。
橙线：这一类别的模型补全部分。可穿过卷面观测 mask 的内部证据缺口，但必须位于卷面几何包络内，且不越过当前顶部边界或已观测侧向范围。
上述补全只扩展模型显示域，不修改原始卷面 mask、遮挡 mask 或计数有效域。
d 处需有左右黄点支撑；当前图像支持时允许的点间跨度不超过 6 个参考绳距，插值状态时不超过 2 个参考绳距。遮挡、卷面无效、长缺点或附近类别冲突都会使距离未知。
该版本保证的是“每个已接受类别在其有效端点之间连续”，并非保证所有真实边界均已识别或可以无限补线。

全视频结果
卷面门控通过 {metrics['region_passed']} 帧，未通过 {metrics['region_blocked']} 帧。
边界类别/曲线累计 {c['curves']:,} 条；这是逐帧累加数量，不是钢丝绳根数。
原始黄点累计 {c['raw_points']:,} 个，其中 {c['classified_points']:,} 个归入已接受类别。
计数通过校验：BG2 {metrics['BG2_count_valid']} 帧 → BG3 v2 {metrics['new_count_valid']} 帧。
位于 0–5 绳距并显示数值：BG2 {metrics['BG2_displayed']} 帧 → BG3 v2 {metrics['new_displayed']} 帧。
{comparison}

验证
8 项合成测试通过：类别唯一、已知类别归属、长缺口连续但拒绝计数、已知真值的插值误差、非交叉、缺失整根不凭空生成、显示包络不改计数 mask、遮挡拒绝交点。
另有 2 项缓存恢复回归测试通过：损坏缓存保留后重算重试，以及重算校验失败时停止并恢复查找路径。
全视频结构审计：{c['curve_samples']:,} 个曲线采样点、{c['ordered_pair_checks']:,} 组曲线顺序检查、{c['unique_count_intersections']:,} 个唯一计数交点，失败数 0。
同时核对了保存曲线与解析模型一致、每类首末黄点跨度、无重复点归属、相邻像素之间 0.5 个处理像素采样的解析模型顺序，以及绿线证据状态。
视频完成后完整解码核验 601 帧、1920×1080、10 fps；没有抽帧、加速或删除未知帧。
本次分析耗时 {receipt['elapsed_s']:.1f} 秒（包含原始证据缓存复用及恢复扫描；不含中断等待，渲染另计），这是离线开发结果，尚未达到或验证现场实时速度。
第 325 帧的旧实验原始缓存不完整，本次从该帧原图重新提取并确认与原始黄点一致后续跑；旧缓存没有修改，已完成帧文件没有重写。见 full_001/cache_recovery.json 和 receipt.json。

仍需验证
类别编号是每帧的视觉绳槽标签，不是跨帧的物理绳编号，也不能直接当成剩余根数。未延伸到 d 的局部类别，其编号不保证与整幅图上下次序一致；非交叉检查使用实际重叠范围内的几何位置。
没有人工逐帧根数真值和接触零点标定；平滑、连续或有效输出更多都不能单独证明物理距离正确。
剩余根数仍受卷面上界、S1 交点、d 假设与局部绳距影响。本轮未强制数值单调下降。

主要文件
global_points.py：点归类、鲁棒拟合、约束与交点。
inputs.py：01 原始点/场提取与缓存核验。
pipeline.py：卷面门控、连续模型与计数集成。
run_checked.py：推荐入口；缓存损坏时从同帧原图重算并核验，不继续使用损坏缓存。
run_bg3.py：冻结的完整视频运行器（默认防止覆盖既有输出目录）。
render_bg3.py、render_stream.py：与计数使用相同曲线的视频输出。
test_global_points.py、audit_results.py、audit_parallel.py：合成测试与实际输出审计。
full_001/frames：逐帧模型、原始点归属、计数与拒绝原因。
full_001/binding.json、snapshot：本次分析代码版本与来源绑定。
delivery/BG3_point_classes_continuous_full.mp4：完整检测视频。
delivery/comparison_240.png：原图、黄点类别与连续曲线对照。
delivery/metrics.json、geometry_audit.json、video_receipt.json：机器可读结果。

复运行检测（请使用未存在的 --name，避免覆盖）
.venv_lc1/Scripts/python.exe "{ROOT / 'run_checked.py'}" --name full_002 --workers 6 --full
查看某次运行的代表帧
.venv_lc1/Scripts/python.exe "{ROOT / 'render_bg3.py'}" --run full_002 --frames 127,163,200,240,270
完整视频渲染入口 render_stream.py 读取本目录 full_001；若制作新的完整版本，建议复制代码到新的实验目录后使用 full_001。
'''
    report=ROOT/'BG3_v2_算法与结果.txt';report.write_text(text,encoding='utf8')
    # Preserve post-run audit/report code separately from the analysis snapshot.
    for p in (ROOT/'audit_results.py',ROOT/'audit_parallel.py',ROOT/'run_checked.py',ROOT/'finalize_delivery.py'):shutil.copy2(p,run/'snapshot'/p.name)
    artifacts=[report,*ROOT.glob('*.py'),out/'BG3_point_classes_continuous_full.mp4',
        out/'metrics.json',out/'geometry_audit.json',out/'video_receipt.json',out/'comparison_240.png',
        run/'binding.json',run/'receipt.json']
    dump(out/'delivery_manifest.json',dict(status='completed',source_sha256=binding['source_sha256'],
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in artifacts},physical_accuracy=None))
    print(report,flush=True)
if __name__=='__main__':main()
