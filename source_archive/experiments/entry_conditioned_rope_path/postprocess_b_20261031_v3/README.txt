热图接续 H3 / IR-H3.0.0 — A7直线修正与卷面中心轨迹上的方向变化评分

图库：D:\precise_arrangement\experiments\entry_conditioned_rope_path\postprocess_b_20261031_v3\all_001\四栏检测结果
292张四栏图：原图 / 冻结B热图 / 连接候选 / 最高评分候选，8192×1655，热图固定0–1。
输入沿用H2全部292图、301入口条件和浮点热图；原400来源中108张入口仍未解决，不称无绳。没有新模型前向或训练。

1. 未卷入段用匹配A7实测轴修正为直线；无可靠A7时用标记清楚的稳健直线回退。原H2可见范围和有界缺口保留，重新检查原图与B，仅降级支持，不把未知变观测。
2. 贴面点严格取一条F3灰色中心轨迹源点。H1离轨曲线不能作为主结果；无法接入则保留自由段，不用任意平滑线替代灰轨道。
3. 固定评分由图像0.50、方向0.40、共同尾部视野覆盖0.10及弱证据/丢失直段/宽度惩罚构成。方向项测量24半尺寸px弦的角幅、总变差、局部曲率、两端切向差，统一看接续后最多120半尺寸px；短于该视野会扣覆盖分。
4. 桥若偏离后续仍有图像支持的直段超过0.65局部宽的点数比例大于10%，直接排除。不会仅为少转弯而丢掉已观察直段。

实际292/292，退出0，237.3秒；产物类别{'free_fragment_no_resolved_connection': 74, 'straight_free_to_F3_center_track': 218}。这些计数不是完整目标身份准确率。
绿=自由段、青=过渡、紫=贴面候选，橙虚线=弱证据/推断。A7轴/F3中心的精确符合仅说明约束已实现，不证明同根、真实接触或末端正确。
pilot_001提前截断反例保留；唯一修订后pilot_002再冻结，全量all_001。旧H2、B模型、原图、旧标签不变。
记录all_001/记录/*.json.gz，points_xy为半尺寸crop坐标，原图=2×xy+[0,1050]。surface_source_indices指向同entry.candidate_track_geometry的源点。
配置/源码哈希：inputs.json、freeze_final.json、all_001/run_start.json；逐图来源及响应哈希在记录。原环境vision_prototype/.venv，无依赖安装、标签/保留集读取、设备连接。

实际命令（D:/precise_arrangement）：
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v3/run_h3.py --run pilot_001 --subset pilot --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v3/run_h3.py --run pilot_002 --subset pilot --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v3/run_h3.py --run all_001 --subset all --workers 4
复跑必须用新run名，不能覆盖旧结果。

与H2同条件对比见comparison_all_001/summary.json：自由线约束已实现；总路径长度-17.76%，局部图像支持长度-9.22%，仅自由10→74。189图有足够共同下游，方向幅度差中位约+0.0034°、总变差+3.15°，没有普遍变平滑或完整识别改善的证据。必须连同路径/支持长度和共同范围不足数一起读，不能用截短造成的小角度宣称更好。固定12例旧新末栏并排在该目录。
独立产物核验见all_001/independent_geometry_audit.json，不是身份准确率评价。
