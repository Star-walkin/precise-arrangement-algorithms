# 自动精密排线视觉算法版本目录

截至 2026-11-03，按独立编号且有可追溯记录的离线研究方法计，共 **61 版**：V 系列 28、RB 7、DR 4、IR 22。

当前离线基准为 RB-F3、RB-C3、DR-D4 和 IR-P1；版本数不代表整根身份、t0/t1/t2 或上线验收。
本仓库保存筛选后的源码与小型报告/协议。早期 V1–V4 主要共用 `vision_geometry_lab` 源码，并无逐版独立冻结代码快照；V15、V21 仅有脚本形态的主证据。目录中的路径保留原工程层级供追溯，原始数据、标签、权重、设备控制和保留集不在本仓库。

| 版本 | 中文名称 | 系列 | 状态 | 源码 | 证据 |
| --- | --- | --- | --- | --- | --- |
| `V1` | 首版独立几何研究 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/首版实验记录.md) |
| `V2` | 屏蔽带与椭圆弧 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/第二版屏蔽带与椭圆弧实验记录.md) |
| `V3` | 局部边界与新入卷轨迹 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/第三版局部边界与新入卷轨迹实验记录.md) |
| `V4` | 遮挡补全与学习模型对照 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/第四轮遮挡补全与学习模型对照实验.md) |
| `V5` | 中心线模型与精修 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V5中心线模型与精修_实施及标注指南.md) |
| `V6` | 小样本训练对照 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V6小样本训练对照与下一步计划.md) |
| `V7` | 冻结特征训练与断线诊断 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V7冻结特征训练与断线诊断_结果.md) |
| `V8` | 补充标注与到边判据 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V8补充标注接收与到边判据说明.md) |
| `V9` | 全局上下文与局部细节 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V9全局上下文与局部细节_实验记录.md) |
| `V10` | 可学性与传统脊线 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V10_可学性与传统脊线对照.md) |
| `V11` | 指定绳路径与连接 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V11_指定绳路径与连接实验.md) |
| `V12` | 光学中心精修与偏差诊断 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V12_光学中心精修与偏差诊断.md) |
| `V13` | 暗面边界与纹理连续性 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V13_暗面边界与纹理连续性检验.md) |
| `V14` | 中心与双边联合监督 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V14_中心与双边联合监督对照.md) |
| `V15` | 双边辅助权重 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/v15_aux_weight_experiment.py) |
| `V16` | 三节点原始视频可行性 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V16_三节点原始视频可行性评估.md) |
| `V17` | 时序边界守卫 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V17_时序边界守卫与三节点对照.md) |
| `V18` | 局部增厚前沿可观测性 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V18_局部增厚前沿可观测性验证.md) |
| `V19` | 持续侧缘扩张 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V19_持续侧缘扩张候选与三节点验证.md) |
| `V20` | 材料变化与旋转外观 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V20_材料变化与旋转外观对照.md) |
| `V21` | 红框与凸起复核 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/v21_box_audit.py) |
| `V22` | 局部双边与法兰连续追踪 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V22_局部双边与法兰连续追踪验证.md) |
| `V23` | 活动绳能力重审 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V23_活动绳能力重审与研究路线收敛.md) |
| `V24` | 锚段一致性训练 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V24_锚段一致性训练与关键转弯瓶颈验证.md) |
| `V25` | 短段细节与V14补测 | V | 研究记录 | [源码](source_archive/vision_geometry_lab) | [记录](source_archive/vision_geometry_lab/V25_短段细节对照与V14独立补测.md) |
| `V26` | 因果信号与材料点跟踪 | V | 研究记录 | [源码](source_archive/vision_geometry_lab/experiments/v26_causal_material) | [记录](source_archive/vision_geometry_lab/V26_因果信号与材料点跟踪首轮实验.md) |
| `V27` | 材料运动可辨识性 | V | 研究记录 | [源码](source_archive/vision_geometry_lab/experiments/v27_material_observability) | [记录](source_archive/vision_geometry_lab/V27_材料运动可辨识性与最小真值闭环.md) |
| `V28` | 局部卷面结构与多候选接续 | V | 研究记录 | [源码](source_archive/vision_geometry_lab/experiments/v28_local_spool_connection) | [记录](source_archive/vision_geometry_lab/V28_局部卷面结构与多候选接续_实验记录.md) |
| `RB-B1.0.0` | 亮带边界基准B1 | RB | 历史基准 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_bright_rope_boundaries_20261025_v3) | [记录](source_archive/工程导航/05_实验版本档案/README.md) |
| `RB-C2.0.0` | 局部缺口曲线C2 | RB | 研究记录 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_local_conic_20261025_v5) | [记录](source_archive/工程导航/05_实验版本档案/README.md) |
| `RB-C3.0.0` | 整组圆锥与整链基准C3 | RB | 现行离线基准 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_grouped_conic_20261026_v6) | [记录](source_archive/工程导航/05_实验版本档案/README.md) |
| `RB-C4.0.0` | 全卷面试验C4 | RB | 研究记录 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_full_surface_20261026_v7) | [记录](source_archive/工程导航/05_实验版本档案/README.md) |
| `RB-F1.0.0` | 亮暗融合F1 | RB | 研究记录 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_bright_dark_fusion_20261028_v1) | [记录](source_archive/数据资产/05_派生与实验结果/camera_condition_bright_dark_fusion_20261028_v1/experiment_report.json) |
| `RB-F2.0.0` | 重叠接续F2 | RB | 研究记录 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_seam_joint_20261028_v2) | [记录](source_archive/数据资产/05_派生与实验结果/camera_condition_seam_joint_20261028_v2/experiment_report.json) |
| `RB-F3.0.0` | 全卷面协同F3 | RB | 现行离线基准 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_full_surface_joint_20261028_v3) | [记录](source_archive/数据资产/05_派生与实验结果/camera_condition_full_surface_joint_20261028_v3/experiment_report.json) |
| `DR-C3.1.0` | 欠曝光卷面三步实验 | DR | 研究记录 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_dark_roi_20261026_v1) | [记录](source_archive/工程导航/05_实验版本档案/README.md) |
| `DR-D2.1.1` | 暗纹锚定D2 | DR | 研究记录 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_dark_boundaries_20261027_v2) | [记录](source_archive/数据资产/05_派生与实验结果/camera_condition_dark_boundaries_20261027_v2/experiment_report.json) |
| `DR-D3.0.1` | 暗沟约束D3 | DR | 研究记录 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_dark_groove_20261027_v3_1) | [记录](source_archive/数据资产/05_派生与实验结果/camera_condition_dark_groove_20261027_v3_1/experiment_report.json) |
| `DR-D4.0.0` | 沟宽筛选D4 | DR | 现行离线基准 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_dark_width_20261027_v4) | [记录](source_archive/数据资产/05_派生与实验结果/camera_condition_dark_width_20261027_v4/experiment_report.json) |
| `IR-A1.0.0` | 活动绳直段与转弯候选A1 | IR | 研究记录 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_incoming_rope_20261026_v1) | [记录](source_archive/工程导航/05_实验版本档案/README.md) |
| `IR-A2.0.0` | 活动绳增强A2 | IR | 研究记录 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_incoming_rope_20261026_v2) | [记录](source_archive/工程导航/05_实验版本档案/README.md) |
| `IR-A3.0.1` | 断界寻绳A3 | IR | 研究记录 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_whole_incoming_20261028_v3) | [记录](source_archive/数据资产/05_派生与实验结果/camera_condition_whole_incoming_20261028_v3/experiment_report.json) |
| `IR-A4.0.1` | 断界寻绳修订A4 | IR | 研究记录 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_whole_incoming_20261028_v4) | [记录](source_archive/数据资产/05_派生与实验结果/camera_condition_whole_incoming_20261028_v4/experiment_report.json) |
| `IR-A5.0.0` | 差分轨道A5 | IR | 研究记录 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_incoming_photometric_20261029_v5) | [记录](source_archive/数据资产/05_派生与实验结果/camera_condition_incoming_photometric_20261029_v5/experiment_report.json) |
| `IR-A6.0.0` | 自模板多候选A6 | IR | 研究记录 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_incoming_template_20261029_v6) | [记录](source_archive/数据资产/05_派生与实验结果/camera_condition_incoming_template_20261029_v6/experiment_report.json) |
| `IR-A7.0.0` | 直绳双边A7 | IR | 研究记录 | [源码](source_archive/数据资产/05_派生与实验结果/camera_condition_free_straight_20261030_v7) | [记录](source_archive/数据资产/05_派生与实验结果/camera_condition_free_straight_20261030_v7/experiment_report.json) |
| `IR-H1.0.0` | 冻结B热图接续H1 | IR | 研究记录 | [源码](source_archive/experiments/entry_conditioned_rope_path/postprocess_b_20261030_v1) | [记录](source_archive/experiments/entry_conditioned_rope_path/postprocess_b_20261030_v1/experiment_report.json) |
| `IR-H2.0.0` | 多模块候选热图接续H2 | IR | 研究记录 | [源码](source_archive/experiments/entry_conditioned_rope_path/postprocess_b_20261030_v2) | [记录](source_archive/experiments/entry_conditioned_rope_path/postprocess_b_20261030_v2/experiment_report.json) |
| `IR-H3.0.0` | F3中心轨迹约束接续H3 | IR | 研究记录 | [源码](source_archive/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v3) | [记录](source_archive/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v3/experiment_report.json) |
| `IR-H4.0.0` | 热图形态与持续直脊接续H4 | IR | 研究记录 | [源码](source_archive/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v4) | [记录](source_archive/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v4/experiment_report.json) |
| `IR-H5.0.0` | 实测直段分档接续H5 | IR | 研究记录 | [源码](source_archive/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v5) | [记录](source_archive/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v5/experiment_report.json) |
| `IR-H6.0.0` | F3来源谱系单轨接续H6 | IR | 研究记录 | [源码](source_archive/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v6) | [记录](source_archive/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v6/experiment_report.json) |
| `IR-H7.0.0` | 热力图输入与监督权重对照H7 | IR | 研究记录 | [源码](source_archive/experiments/entry_conditioned_rope_path/heatmap_20261101_v1) | [记录](source_archive/experiments/entry_conditioned_rope_path/heatmap_20261101_v1/experiment_report.json) |
| `IR-P1.0.0` | 活动绳整链基准P1 | IR | 现行离线基准 | [源码](source_archive/工程导航/05_实验版本档案/活动绳整链基准_P1_20261101) | [记录](source_archive/工程导航/05_实验版本档案/活动绳整链基准_P1_20261101/README.txt) |
| `IR-V1.0.0` | 视频时序约束V1 | IR | 研究记录 | [源码](source_archive/experiments/active_rope_video_20261101_v1) | [记录](source_archive/experiments/active_rope_video_20261101_v1/experiment_report.json) |
| `IR-V2.0.0` | 视频逐帧灰线V2 | IR | 研究记录 | [源码](source_archive/experiments/active_rope_video_20261101_v2) | [记录](source_archive/experiments/active_rope_video_20261101_v2/experiment_report.json) |
| `IR-R1.0.0` | 空间关系匹配R1 | IR | 研究记录 | [源码](source_archive/experiments/entry_conditioned_rope_path/relation_r1_20261101) | [记录](source_archive/experiments/entry_conditioned_rope_path/relation_r1_20261101/REPORT.txt) |
| `IR-DATA-D1.0.0` | 自动弱监督数据扩充D1 | IR | 研究记录 | [源码](source_archive/experiments/entry_conditioned_rope_path/data_expansion_d1_20261102_v1) | [记录](source_archive/experiments/entry_conditioned_rope_path/data_expansion_d1_20261102_v1/REPORT.txt) |
| `IR-GEOM-G1.0.0` | 原图几何追踪与相对卷面方向G1 | IR | 研究记录 | [源码](source_archive/experiments/entry_conditioned_rope_path/geometry_direction_20261103_v1) | [记录](source_archive/experiments/entry_conditioned_rope_path/geometry_direction_20261103_v1/REPORT.md) |
| `IR-M1-R1.0.0` | 人工中心监督U-Net与匹配R1 | IR | 研究记录 | [源码](source_archive/experiments/entry_conditioned_rope_path/manual_r1_20261103_v1) | [记录](source_archive/experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/REPORT.txt) |
| `IR-U1-P1-POST.0.0` | 人工监督U1热图＋P1/H6冻结后处理 | IR | 研究记录 | [源码](source_archive/experiments/entry_conditioned_rope_path/manual_u1_p1_20261103_v1) | [记录](source_archive/experiments/entry_conditioned_rope_path/manual_u1_p1_20261103_v1/REPORT.txt) |

## 口径

- 比较臂、训练步数、运行号不另计版本。C1 失败分支和 S1 标注准备未计入。
- `IR-M1-R1.0.0` 是第 60 个完成版本，`IR-U1-P1-POST.0.0` 在四栏292图和独立审计完成后成为第 61 个。旧59版盘点是历史快照。
- 完成版本状态按各自实验报告和交付审计记录；算法输出数量不等于同根身份准确率。
- V 系列与 IR-V1/V2 是两套不同编号；源码与报告路径说明可追溯性，不保证离线版本可在新机器直接独立运行。
- 不含原始媒体、标签、模型权重、推理结果、虚拟环境或设备控制文件；需要在原工程受控环境中复现。
