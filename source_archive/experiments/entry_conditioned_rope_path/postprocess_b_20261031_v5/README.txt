热图接续 H5 / IR-H5.0.0：有支持的长直段优先，再比较方向与灰色中心轨迹

四栏图库：D:\precise_arrangement\experiments\entry_conditioned_rope_path\postprocess_b_20261031_v5\all_001\四栏检测结果
实际292/292、301入口、退出0，273.9秒；{'straight_free_to_F3_center_track': 80, 'free_fragment_no_resolved_connection': 212}。这些是产物类别，不是识别准确率。
原400图中的108张旧入口未解决，仍留总分母，不能称无绳。

同一冻结B模型、301份浮点响应与自动入口；仅离线后处理，无训练/新前向/人工标签/保留集/设备访问。
自由段按实际支持区间长度排序，入口提示和未知缺口不计长度。48原px长度分档先于次级评分。
同档：0.60长度、0.25方向、0.10受支持F3轨迹、0.05图像，再扣桥接弱证据/宽度变化等；次级分固定小于1，不得反转长度主序。
第一次几何偏轴及SAM2远侧终域均为约束/诊断，不是真实接触或身份真值。未观察后段不能获得虚构零转向奖励。
热图弱但原图双边清楚时，仅利用已匹配、已接受且有股纹证据的A7有限观测进行保守补段；原图复测通过才计支持，单列raw_a7_supported，绝不改写B支持指标。
同一A7的相反方向条件仅在实际路径有明确外→内支持差异时筛任务方向；B图不合并，不能据此宣称物理接触或完整同根。
主轴之外备选轴留作诊断，尚未全局解决跨轴身份竞争。未证明“几乎全部正确”。

绿色=原图+B联合，黄色=仅热形态支持，浅蓝=A7+原图补段，青/紫=接续/贴面候选，橙虚线=弱证据或推断。
热图固定0～1，称响应强度。记录xy映射原图=2*xy+[0,1050]，浮点B在每条记录的response_file中保持原hash。
14张固定失败/对照开发；实际开发轮次：['pilot_001', 'pilot_002', 'pilot_003', 'pilot_004']。额外8图是冻结后开放开发核查，不是独立保留集，也不据此回流调参。
源/参数/运行命令：inputs.json、freeze_final.json、各run/run_start.json、source_snapshot、progress.jsonl。
查看comparison_all_001和review_receipt.json中的改善、退化与未解决案例；不能只按总长度判断提升。

运行环境：D:/precise_arrangement/vision_prototype/.venv/Scripts/python.exe，CPU，既有依赖不改。
实际全量命令：vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v5/run_h5.py --run all_001 --subset all --workers 4
复跑请用新run目录，不覆盖历史结果。
