热图接续 H2 / IR-H2.0.0：冻结B热图 + A7自由段 + SAM2/F3卷面中心轨迹

交付图片：D:\precise_arrangement\experiments\entry_conditioned_rope_path\postprocess_b_20261030_v2\all_001\四栏检测结果
292张JPEG，每张8192×1655，依次为原图、B热图、多候选、最高评分候选。
已扫描的来源分母400；292张有旧自动入口，108张入口仍未解决，不认定其无绳。
复用299份H1数值B响应；Rrun1274新增A7自动轴的两个方向分别前向，共301条件。
新条件不是人工提示；所有不同入口响应始终分开。显示采用所选入口的原响应，固定0–1。

最优指固定启发式评分最高，不代表完整同根身份正确。绿=自由、青=过渡、紫=贴面候选；橙虚线=弱证据或推断，白圈=当前末端。
F3是边界。先在纯SAM2卷面内配对真实相邻边界，再取局部中点并有限细化（≤间距15%、≤2半尺寸px）。缺边不按固定绳距补全。
A7确认同轴且两端各有30原px原图支持时，允许≤min(400原px,20实测宽)的缺口候选，累计≤600原px；缺口不是观测/遮挡真值。
连接项包括原图双边、B中心/双侧低带、局部曲率/方向/宽度兼容、F3双边、放弃已观察自由段惩罚。没有3D标定，不宣称物理规律已验证。

实际：292/292，退出码0；整批216.4秒；新增B前向1.32秒。
路径产物类别：{'direct_H1_response_path': 118, 'free_bridge_adjacent_F3_lane': 164, 'free_fragment_no_resolved_connection': 10}。这些是输出计数，不是准确率。
固定12开发例先检查、必要修订一次，随后冻结整批；没有读取完整身份标签或保留集，没有训练或设备连接。
pilot_001及integration_smoke_001的历史失败保留；正式为pilot_002和all_001。
模型、原图、旧实验保持原内容；新记录在all_001/记录/*.json.gz，points_xy是半尺寸crop坐标，原图坐标=2×xy+[0,1050]。
所有来源/模型/参数/核心源码哈希见inputs.json、freeze_final.json、all_001/run_start.json；逐图来源哈希在压缩记录。

环境沿用vision_prototype/.venv（cv2/numpy/Pillow）与vision_training/.venv（PyTorch CPU），未安装依赖。
实际命令（工作目录D:/precise_arrangement）：
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261030_v2/prepare_h2.py prepare
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261030_v2/prepare_h2.py infer
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261030_v2/run_h2.py --run pilot_001 --subset pilot --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261030_v2/run_h2.py --run pilot_002 --subset pilot --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261030_v2/run_h2.py --run all_001 --subset all --workers 4
复跑须使用新的run名，不能覆盖已有结果。

最终独立结构核验：all_001/audit_002/report.json，292图/301条件全部通过，退出0；752候选及弱证据虚线、坐标/种子映射、模型/源码/原图/响应哈希与JPEG解码均核验。结构检查不证明身份准确。
首轮audit_001因单线程OpenCV距离变换重建seed出现约1e-8差异而失败，原记录保留；审计改回并行分支后逐bit一致，没有修改输入/产物/容差。
实际核验命令：vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261030_v2/audit_h2.py --run all_001 --out audit_002 --workers 2
