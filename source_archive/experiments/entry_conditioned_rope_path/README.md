# 入口条件化目标绳段离线实验

## 2026-10-31：热图接续 H3 / IR-H3.0.0（自由直轴与中心轨迹约束）

同条件全量对比见 [H2/H3对比](postprocess_b_20261031_v3/comparison_all_001/summary.json)：总路径长度减少17.76%、局部图像支持长度减少9.22%；共同范围足够的189图未显示普遍方向变化下降。已实现直轴/中心轨迹约束，但不据此认定接续改善，不继续靠扩大预算调评分。

[292张四栏结果](postprocess_b_20261031_v3/all_001/四栏检测结果/) · [实际报告](postprocess_b_20261031_v3/experiment_report.json) · [说明](postprocess_b_20261031_v3/README.txt)。按用户要求，A7只提供自由段直轴，保留H2原有界可见范围并重新检查支持；贴面点严格取同一F3中心轨迹，加入方向幅度/总变差/曲率/两端切向差评分。首版A7末端截短引起提前接圈的反例保留，修订一次后冻结。沿用同292图/301个B浮点条件，实际292/292退出0、237.3秒，无训练或新前向；218图接灰色中心轨迹、74图仅自由段（H2为10），是约束后产物计数，不是身份准确率。直线与轨道符合不证明正确同根，整体接续改善未证；弱段、拒接及缩短例保留。旧模型/实验不变，无标签/保留集读取或设备连接。

## 2026-10-31：热图接续 H2 / IR-H2.0.0（B + A7 + F3中心轨迹）

[292张四栏检测图](postprocess_b_20261030_v2/all_001/四栏检测结果/) · [运行报告](postprocess_b_20261030_v2/experiment_report.json) · [算法与复跑说明](postprocess_b_20261030_v2/README.txt)。按用户指定流程，复用299份冻结B浮点响应，为Rrun1274不匹配旧入口的A7轴新增两个方向前向，共301自动条件。纯SAM2卷面内将实际相邻F3边界转成中心轨迹，结合原图双边、热图中心/两侧低带、A7同轴重现与有限曲线接续进行固定评分。12例检查后冻结，整批292/292、退出0、216.4秒；164个F3接续候选、118个直接H1路径、10个仅自由段，均是产物类别而非身份正确率。400来源中108张仍无可用入口，不认定无绳；候选竞争、弱尾、方向未解决及未抵真实末端保留。未证明完整同根身份或净增益，不替换旧基线；无训练、标签/保留集读取或设备连接。目录日期沿用开始时的20261030。

## 2026-10-30：热图接续 H1 / IR-H1.0.0（方案B后处理候选）

[292张六步长条图](D:/precise_arrangement/experiments/entry_conditioned_rope_path/postprocess_b_20261030_v1/all_001/长条对照图) · [分步图](D:/precise_arrangement/experiments/entry_conditioned_rope_path/postprocess_b_20261030_v1/all_001/分步图像) · [算法说明](D:/precise_arrangement/experiments/entry_conditioned_rope_path/postprocess_b_20261030_v1/H1算法与结果说明.txt) · [实际报告](D:/precise_arrangement/experiments/entry_conditioned_rope_path/postprocess_b_20261030_v1/experiment_report.json)。按用户指定方案B图库的292图、299个自动入口复推原B第750步模型；原400图中108个旧入口失败不改称无绳。固定12图开发及明确留档的可见范围/记录隔离修复后冻结，再跑预定8图和全量。实际292/292、退出0，原图与热图亮暗剖面、同轴重现、方向连续候选及F3弱提示；六步长条按多入口分行，另存分步图和估计绳带。独立结构审计通过，299份数值响应范围与旧B记录一致；结构计数不是身份准确率。导杆和早转旧圈部分改善，但暗尾多解、局部偏位、短尾和工具错入口仍保留；绿仅局部匹配，橙待确认，F3橙虚线不入主掩膜。未证明整根同根或相对基线普遍净增益，不替换F3/A6/A7；未训练、未读标签/保留集、未接设备，无HTML。


2026-10-30新增监督方案A/B实际对照：[run_20261030_002报告](runs/run_20261030_002/REPORT.md) · [图片](runs/run_20261030_002/comparison/visualizations) · [复现说明](runs/run_20261030_002/README.md)。两组均从旧final续训，A781步、B774步，主比较共同750步；新增明确背景和4个短邻绳工程弱负例，新正例长度0。B降低固定背景误响应，但正确后段响应/主路径覆盖退化，未验证完整目标身份，不替换旧基线。一键入口为`run_supervision_experiment.ps1`，旧实现与历史运行保留。

本目录实现从短自由入口出发，经过过渡及卷面侧区域，向观察到的远侧卷面边缘搜索当前目标弧段。单一中心线响应头，无物理绳宽监督的区域不输出“真实主体mask”。已执行运行见 [run_20261029_001/REPORT.md](runs/run_20261029_001/REPORT.md)。原图、旧实验及设备代码不修改。

源数据：历史开放V11开发清单，10训练图/4来源组、8评价图/3来源组。保持第01/02轮已有评价来源不回流。S1尚未训练的独立规划与这些历史用途存在冲突，详见运行的split_conflict_note.json；不得把本模型在S1规划评价组上的输出当独立评价。

## 文件入口

- `audit_sources.py`：来源、历史分组、可信范围与最少缺失参考审计。
- `scene.py`：从原图观察弧及短入口构造卷面边缘候选和阶段位置代理。
- `prepare.py`、`prepare_auto_seeds.py`：只在可信几何内生成中心热图、局部正负监督与ignore，保存完整搜索区域。
- `model.py`、`train.py`：灰度+入口条件U-Net，全局入口特征到达远端；阶段/高低响应平衡MSE；小样本门与单次有界训练。
- `infer.py`：同一完整ROI推理，恢复到原图坐标；原始float32 `.npy` 框外NaN，processed版框外0。
- `connect.py`：网络响应与原图两侧/剖面证据的有限beam搜索，最多3条路径；输入段、图像支持、歧义分开。
- `evaluate.py`：先冻结全部预测再读取质量裁切的8图参考；已给入口不计预测识别；分阶段报告，拒识保留。
- `visualize.py`：全ROI、固定局部和未训练响应对照，热图固定0–1。
- `verify_preparation.py`、`verify_inference.py`：实际数据/输出契约核验。

## 环境及运行

从 `D:\precise_arrangement` 的PowerShell运行，复用已有环境，没有安装依赖。图像/评价环境为 `vision_prototype\.venv\Scripts\python.exe`（OpenCV、NumPy、Pillow）；网络环境为 `vision_training\.venv\Scripts\python.exe`（PyTorch CPU）。训练预算与参数在运行`model/config.json`，路径参数在`path_config.json`，实际命令/退出码在各receipt和summary中。

以下展示本次实际命令的核心入口；已生成的运行目录应保留，复跑用新名称，并复制冻结的`path_config.json`和协议。

```powershell
$run = 'experiments/entry_conditioned_rope_path/runs/run_20261029_001'
& .\vision_prototype\.venv\Scripts\python.exe -B experiments/entry_conditioned_rope_path/audit_sources.py
& .\vision_prototype\.venv\Scripts\python.exe -B experiments/entry_conditioned_rope_path/prepare.py --run $run
& .\vision_training\.venv\Scripts\python.exe -B experiments/entry_conditioned_rope_path/train.py train --index "$run/data/index.json" --out "$run/model"
& .\vision_prototype\.venv\Scripts\python.exe -B experiments/entry_conditioned_rope_path/prepare_auto_seeds.py --run $run
& .\vision_training\.venv\Scripts\python.exe -B experiments/entry_conditioned_rope_path/infer.py --run $run --model "$run/model/initial.pt" --phase initial --threads 4
& .\vision_training\.venv\Scripts\python.exe -B experiments/entry_conditioned_rope_path/infer.py --run $run --model "$run/model/final.pt" --phase final --threads 4
& .\vision_prototype\.venv\Scripts\python.exe -B experiments/entry_conditioned_rope_path/connect.py --run $run
& .\vision_prototype\.venv\Scripts\python.exe -B experiments/entry_conditioned_rope_path/evaluate.py --run $run
& .\vision_prototype\.venv\Scripts\python.exe -B experiments/entry_conditioned_rope_path/visualize.py --run $run
```

`predictions/CONDITIONAL/<图名>`是已有已知入口条件结果。`AUTO/<图名>`沿用同图冻结V28自动入口；3张入口失败保持全NaN，不回退人工提示。每张保存`response.npy`、初始未训练响应、`entry.json`、`inference.json`及`paths.json`。路径点按入口到目标边缘方向排序；`selected_candidate_id=null`是拒识，不代表没有竞争路径可查看。`selected_centerline_schematic.png`仅1像素中心线示意，不是绳体mask；拒识时为空图，竞争候选从JSON与叠加图查看。

“visible_supported”只表示局部原图及网络响应支持；网络响应强度不是真实概率或身份可信度。“attached”阶段是冻结的空间位置代理，不是物理接触标签；碰到黄法兰/灰绳观察界面也不等于抵达正确目标末端。没有确认的遮挡时，弱图像证据保留歧义，不虚构遮挡推断。独立完整目标末端参考仍缺失。

## 2026-10-30：冻结A/B中心线段与短缺口连接P1

[本轮简明报告](postprocess_20261030_001/REPORT.md) · [292张原图/A/B/P1四联图](postprocess_20261030_001/四联图_原图_A_B_中心线示意/) · [108张缺入口诊断](postprocess_20261030_001/无入口诊断_108/) · [实际命令](postprocess_20261030_001/COMMANDS.md)。共同750步A/B冻结，P0旧逐步连接器完全重放；P1用4训练来源校准后冻结，完成既有8评价图/3来源、CONDITIONAL与AUTO的64组合。A条件总体部分参考覆盖38.37%→29.41%，B条件11.06%→12.81%，AUTO均退化；P1四组近边缘覆盖为0，短桥贡献有限，尚不能完整保持目标身份。保留P0，P1不替换基线。

400开放图中292张有299个已存入口，补598次冻结前向后运行两种连接器并保存float与派生片段；108缺入口保留失败，不称无绳。pair0075两入口分别展示，训练来源不纳入评价。第四面板为中心线显示覆盖，不是真实绳体宽mask。598份float、1196路径及模型哈希等核验通过；无训练/微调、无保留集访问、无设备连接。新源码入口为`p1_segments.py`及`postprocess_*.py`，旧模型/连接器/历史输出保留。


## 2026-10-31：热图接续 H4 / IR-H4.0.0

2026-10-31新增[热图接续H4 / IR-H4.0.0](D:/precise_arrangement/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v4/experiment_report.json)：同292图/301条件复用B浮点响应，有限角度/偏移重估直段，以中心高、双侧低热形态确定范围，A7仅初值；保护持续直脊，弱热图接轨不能自动胜过暂不接轨。实际292/292退出0，130图接轨、162图仅自由段、0图无候选；108旧入口未解决仍留400分母。用户R_static095反例长斜段恢复，首版短亮段评分偏置及统一视野积分修订分别保留；热图形态与覆盖改善不等于完整同根身份验证。无训练/新前向/标签/保留集/设备访问，不替换既有基线。

[292张四栏图](D:/precise_arrangement/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v4/all_001/四栏检测结果) · [固定13图原图/H3/H4对照](D:/precise_arrangement/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v4/comparison_all_001/固定13图_原图_H3_H4)


## 2026-10-31：热图接续 H5 / IR-H5.0.0

2026-10-31新增[热图接续H5 / IR-H5.0.0](D:/precise_arrangement/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v5/experiment_report.json)：同292图/301条件复用B响应；14张固定失败/对照开发、8张冻结后开放素材核查，既有保留集不读。按扣除入口/未知的实测直段长度分48px档，再比较转向、受支持F3中心轨迹与图像；取消2000px评分截止，限制穿过卷面后的远侧域外延长。热图漏段仅在A7有限观测和原图双边同时通过时单独补回；同A7相反条件按已实测外→内支持判任务方向。实际292/292退出0，80图接轨、212图仅自由段；108旧入口未解决仍留400分母。0195反例提前转向有局部改善，残余漏段、方向/末端歧义和弱尾保留，未建立“几乎全部正确”或完整同根准确率。无训练/新前向/标签/保留集/设备访问，不替换既有基线。

[292张四栏图](D:/precise_arrangement/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v5/all_001/四栏检测结果) · [固定案例对照](D:/precise_arrangement/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v5/comparison_all_001)


## 2026-10-31：热图接续H6 / IR-H6.0.0

2026-10-31新增[热图接续H6 / IR-H6.0.0](D:/precise_arrangement/experiments/entry_conditioned_rope_path/postprocess_b_20261031_v6/experiment_report.json)：同292图/301条件复用B；16固定图开发、旧8开放回归后全量。直接复用F3接受曲线及来源谱系、完整C4范围，以固定相邻边界对构建中心，源内有界推断另标。桥与贴面硬检非目标中心穿越/切触/重叠，贴面锁单轨；完整H5直轴保留，弱证据跨轨尾从任务路径中分离，真实自由投影交点须连续原图证据覆盖。实际292/292退出0，103图接轨候选、189图仅前缀；108旧入口未解决仍留400分母。0345有单轨接续候选，0255非法桥被拒但完整接续未解决。无交叉/精确来源只证明几何约束，不能证明同根身份或几乎全对；漏段和源缺口保留。无训练/新前向/标签/保留集/设备访问，不替换F3/A7等既有基线。


## 2026-11-01：热力图输入与监督权重对照H7

2026-11-01新增[热力图输入与监督权重对照H7 / IR-H7.0.0](D:/precise_arrangement/experiments/entry_conditioned_rope_path/heatmap_20261101_v1/experiment_report.json)：从同B750续训C400步、D366步，共同350步比较；10训练图/4组、8历史开放评价图/3组隔离，未新增目标正标注。C降低工程弱负权重并温和续训，D同条件增加保留原灰的LCN辅助通道；训练侧选择C后冻结，完成48主评价记录（含9条缺失自动入口）及入口消融、同292图/301条件新前向和冻结H6后处理。左L_run反例来源未参加旧训练，旧监督97.03%仍ignore；400张图库不等于400张完整监督。新旧固定色标/浮点结果、改善与退化均保留，未建立几乎所有图清晰识别或完整同根身份准确率。原B/H6及其他基线保持不变；无设备/保留集或S1新标签访问。

[292张四栏图](D:/precise_arrangement/experiments/entry_conditioned_rope_path/heatmap_20261101_v1/postprocess/all_001/四栏检测结果) · [原图/旧B/新热图](D:/precise_arrangement/experiments/entry_conditioned_rope_path/heatmap_20261101_v1/postprocess/all_001/原图_旧B_新热力图) · [说明](D:/precise_arrangement/experiments/entry_conditioned_rope_path/heatmap_20261101_v1/REPORT.txt)


## 2026-11-01：活动绳整链P1与视频时序V1

2026-11-01新增[活动绳整链基准P1 / IR-P1.0.0](D:/precise_arrangement/工程导航/05_实验版本档案/活动绳整链基准_P1_20261101/README.txt)与[视频时序V1 / IR-V1.0.0](D:/precise_arrangement/experiments/active_rope_video_20261101_v1/experiment_report.json)：原B750＋H6整链338份快照封存，H7不纳入；400图溯源为16段视频390帧及10张独立照片。16/16完整来源视频离线运行，2061次分析、1528次原B新前向，最近5次/2秒已接受历史约束当前最多3候选，异常拒绝不回填；1秒旧参考过期后UNKNOWN，连续新候选另立重捕获状态。输出2Hz分析/5fps原速四子图。场景沿用最早开放锚点SAM2/F3与因果配准，并非逐帧完整F3重测或任意视频冷启动。时序接受与平稳不等于同根身份，未证明在线实时、完整贴面接续或真实准确率；无训练、保留集或设备访问。

[16段四子图视频](D:/precise_arrangement/experiments/active_rope_video_20261101_v1/四子图视频) · [来源映射](D:/precise_arrangement/experiments/active_rope_video_20261101_v1/source_trace/source_trace.json)


## 视频逐帧灰线V2 / IR-V2.0.0

2026-11-01新增[视频逐帧灰线V2 / IR-V2.0.0](D:/precise_arrangement/experiments/active_rope_video_20261101_v2/experiment_report.json)：修正V1灰线几何沿用首锚的问题，同16开放视频18,017源帧均从当前原灰重新提取边界并转相邻中心，H6贴面候选使用当帧中心；原B750权重不改，有入口帧重新前向，输出原帧率四子图。18,017帧逐一对齐及277888项结构审计通过；这是轻量新观测，非原F3全部融合链重跑。静态场景ROI/粗方向配置仍沿用，弱区断线和同根身份未验收，不以坐标变化或审计通过解释为真实准确率。P1/V1保留，无训练、设备或保留集访问。

[16段逐帧四子图视频](D:/precise_arrangement/experiments/active_rope_video_20261101_v2/四子图视频)


## 2026-11-01 B750 方法报告

[B750 活动绳热力图生成方法与失效分析 DOCX](../b750_method_report_20261101_v1/B750活动绳热力图生成方法与失效分析报告.docx)：15页代码溯源，含输入、全搜索框标准化、入口条件网络、浮点回映与H6后处理；四组原像素PNG案例及断线诊断。仅复核既有开放结果，无新训练或模型前向，不替换B750/P1/V2；像素与来源核验见报告目录。


## 2026-11-02 接续关系 R1 离线实验

[R1报告](relation_r1_20261101/REPORT.txt) · [292张原像素两栏PNG](relation_r1_20261101/交付_292张原图与热力图/)。冻结A750，MLP 600步，在pool/beam前评分；同搜索R0/R1固定8图AUTO后段5px覆盖12.30%→19.42%，CONDITIONAL16.11%→17.98%，仅部分几何参考。实际看到L_run585、0255等跨多圈/错误延伸，完整身份未通过，不替换P1/H6。10训练图/4组，828局部弱配对，非独立身份真值；292图/301入口全产物及像素核验通过，原400另108旧入口未解决。无设备、保留集或S1访问。


## 2026-11-03 自动弱监督数据扩充 D1

[292张D1原始热图](data_expansion_d1_20261102_v1/final_292/) · [实际报告](data_expansion_d1_20261102_v1/REPORT.txt)。新增24张自动弱标注图、3来源组，旧10图保留；D0/D1从A750全参数各600/600步，共同第600步。AUTO部分后段5px参考覆盖25.88%→35.99%，但邻圈选择性及旧图拟合退化；不是完整身份准确率。292原像素双栏不叠路径，301浮点独立保存。未知尾段忽略，完整接续/边缘未有独立真值；不替换旧基线，无设备或保留集访问。


## 2026-11-03 原图几何追踪＋相对卷面方向 G0/G1

[实际实验报告](geometry_direction_20261103_v1/REPORT.md) · [292张G1双栏PNG](<C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/final_292>) · [全量审计](<C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/audit_receipt.json>)。沿 D1 相同 292 唯一源图和固定 E1；原图几何追踪 G0 与仅增加软方向加分的 G1，不用网络热值、不训练。292/292 运行退出0，PNG/原图左栏/预测/float32 全量审计通过；G1 为265部分、27歧义。固定8张历史开放评价图两臂已选轨迹和可信部分5px覆盖完全相同，AUTO后段5.17%、CONDITIONAL后段10.17%、近边均0；方向项无可证净增益，真实边缘与整根身份未验收。不替换旧基线，无设备/保留集访问。


## 2026-11-03 人工监督M1＋匹配R1

[实际报告](manual_r1_20261103_v1/REPORT.txt) · [292三栏原像素PNG](C:/precise_arrangement_results/manual_r1_20261103_v1/final_292/) · [来源/模型/入口索引](manual_r1_20261103_v1/final_292_index.json)。24人工目标中心/51红邻绳，旧10保留；U-Net800步、新R1 369对/600步，301同次新响应/特征与路径。有限邻圈可分性改善，但新R1未全面优于旧R1，错误入口、跨圈和拒识仍保留；不报告完整身份准确率，不替换旧基线。292/301独立像素/来源审计通过，无设备或保留集访问。
