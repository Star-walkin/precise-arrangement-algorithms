# 当前生产模块与支撑源码

`src/` 保持原实验相对目录，所有文件字节与 [source_inventory.json](../source_inventory.json) 对应。复制进来的旧版目录中既有当前依赖，也有探针/对照脚本。**当前有效算法是下面的组合，不是任意一个旧目录单独运行的结果。**

| 层 | 当前模块位置 | 作用 |
|---|---|---|
| 原始前端 | [R19/engine.py](../src/RopeEdgeSuite_R19_20261110/engine.py) | 当前 BG5、SG2/SG3/SG5、BG11、CG2 调度 |
| 热图与路径 | [LC1 支撑目录](../src/RopeEdgeSuite_R1_20261107/baseline/source/LC1_locked_column_20261105_v1/) | B750/P1/H6、S1/S2 适配与原始路径几何 |
| 活动绳融合 | [R19/tracker_sg5.py](../src/RopeEdgeSuite_R19_20261110/tracker_sg5.py) | 当前身份核验与候选融合 |
| 分段与记忆 | [R12/tracker_sg2.py](../src/RopeEdgeSuite_R12_20261109/tracker_sg2.py)、[R15/tracker_sg3.py](../src/RopeEdgeSuite_R15_20261109/tracker_sg3.py) | 有证据分段拟合、短时搜索记忆、重捕获 |
| 边界 | [R19/boundary_bg11.py](../src/RopeEdgeSuite_R19_20261110/boundary_bg11.py) 及 BG9/BG10 | 活动绳/工具排除、鲁棒形状、有限历史先验 |
| F3 基础与分组 | [R1/transfer_engine.py](../src/RopeEdgeSuite_R1_20261107/transfer_engine.py)、[R20/groups20.py](../src/RopeEdgeSuite_R20_20261110/groups20.py) | 复用用户 `01` 核心和冻结黄点归类曲线算法 |
| ROI 与碎片 | [R20/roi22.py](../src/RopeEdgeSuite_R20_20261110/roi22.py)、[R20/region_fragments24.py](../src/RopeEdgeSuite_R20_20261110/region_fragments24.py) | 当前计数带与法兰碎片修复；按记录绑定的修复支路执行 |
| 卷面侧轴线 | [R20/coil_edge_axis_v4.py](../src/RopeEdgeSuite_R20_20261110/coil_edge_axis_v4.py) | CE4 当前卷面侧测量 |
| 切线与 d | [R20/tangent_column.py](../src/RopeEdgeSuite_R20_20261110/tangent_column.py)、[R20/calibrate_guard.py](../src/RopeEdgeSuite_R20_20261110/calibrate_guard.py) | TC1、逐段 d 与至少 5 根的最终审核 |
| 距离计数 | [R20/count20.py](../src/RopeEdgeSuite_R20_20261110/count20.py) | 单类单交点、完整相位/几何估计、敏感性 |
| 动态列 | [R21/measurement21b.py](../src/RopeEdgeSuite_R21_20261110/measurement21b.py) | 无可靠固定 d 时的当前列与短时不确定性检查 |
| 近边事件 | [R21/events21b.py](../src/RopeEdgeSuite_R21_20261110/events21b.py) | 当前证据进入、持续离开与失效原因 |
| MTD1 | [R22/events22.py](../src/RopeEdgeSuite_R22_20261111/events22.py) | 分离运动方向/距离等级，异常值拒绝、短暂保持 |
| 图像周期 | [R22/rotation22.py](../src/RopeEdgeSuite_R22_20261111/rotation22.py) | 多纹理区域周期及一致性 |
| CPL1 | [R23/period23.py](../src/RopeEdgeSuite_R23_20261111/period23.py) | 根数变化稳健周期与两周期时间模型 |
| MFT2 | [R23/fusion23.py](../src/RopeEdgeSuite_R23_20261111/fusion23.py) | 周期融合、第二次回归与当前离边证据 |
| 最终输出 | [R23/run23.py](../src/RopeEdgeSuite_R23_20261111/run23.py)、[R23/render23.py](../src/RopeEdgeSuite_R23_20261111/render23.py) | 保留原字段并增加融合结果、生成视频 |
| 审核 | [R23/audit23.py](../src/RopeEdgeSuite_R23_20261111/audit23.py) | 完整解码、45 测试、因果前缀、旧字段和基准哈希 |

R22 `motor22.py` 的 MFT1 是历史对照，R23 有效结果应读取 MFT2。旧的固定 `d=2238.7` 仍存在于基础前端代码中，作为相应模块的历史参考；最终测量使用 R20/R21 的逐段标定/动态列，不能据此认定全部视频统一采用 2238.7。

`probe*`、`inspect*`、`trial*`、`review*`、`report*` 是研究、查看或报告工具；其中含选例视频名和人工评价时间不表示算法在检测时使用这些时间。生产 CPL1/MFT2 不读取 `human_reference.json`。具体冻结生产源哈希也保留在 [R20](../evidence/R20/release_manifest.json)、[R21](../evidence/R21/release_manifest.json)、[R22](../evidence/R22/release_manifest.json)、[R23](../evidence/R23/release_manifest.json) 的原发布清单中。
