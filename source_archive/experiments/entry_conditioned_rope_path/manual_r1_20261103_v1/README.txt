人工监督M1：新U-Net＋匹配R1（2026-11-03）

主交付292张三栏无损PNG：
C:\precise_arrangement_results\manual_r1_20261103_v1\final_292
左原图 / 中新U1原始热图 / 右新U1+R1路径候选，每栏4096×3000。
中间热图未叠路径或标签；固定0~1。第三栏青绿色是算法声称的支持、橙虚线是推断。
这些状态不是身份可信概率。详细收益、退化和明确错误见REPORT.txt。

本轮实际：24人工目标中心+51红邻绳负线，旧10保留；U-Net从A750全参数800步，
再从新网络提取响应与dec1特征，同结构R1训练369配对600步。
已完成301真实新前向/路径、固定三臂48项评价、全292独立像素审计。
原图、用户JSON及旧模型不改；保留未知、拒识和错误，不替换历史基线。

定位文件
protocol.json                  来源、入口、预算、储存映射
label_audit/annotation_snapshot 人工原样JSON快照
dataset/index.json             34训练样本与冻结cache（junction到C）
dataset/statistics.json        目标/负线/冲突与有效域
model/checkpoints/step_0800.pt 新U-Net权重；model/train_log.jsonl等实际日志
selected_checkpoints.json      固定选择最后保存步，未按评价挑模型
relation/model/                新R1权重/portable版/训练日志与receipt
relation/pair_manifest.json    人工正确候选-红邻绳配对及未知排除
inference/gallery/U1/          301份float32 response+float16新dec1 features（junction到C）
paths/gallery/                 301份selected/alternatives/分段状态（junction到C）
evaluation/                    固定8图AUTO/CONDITIONAL的R0/R1_old/R1_new评价
visual_review/                 原生局部与旧D1对比、明确失败记录
final_292_index.json           292源图、模型、入口、用途、PNG与float/path/hash索引
audit_delivery_receipt.json    独立全292/301核验，通过不等于身份准确率

实际命令（从D:\precise_arrangement执行）
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/prepare_manual.py
experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/run_unet.ps1
experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/run_r1.ps1
experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/run_render.ps1
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/evaluate_results.py
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/audit_delivery.py --workers 4

训练/前向/路径命令会拒绝覆盖，重现需先改到独立新版本目录和新的bulk根，不在本目录重复写入。
执行耗时与真实退出保留在各execution_receipts、日志以及repairs/gallery_role_001。
R1首轮10个钢尺照片角色解析失败，仅补10项，291旧JSON不变；评价格式及mode修正前结果也保留。

由于D盘当时仅余2.37GB，新的大量数据放C盘，代码与小模型留D；未移动任何原始素材或历史目录。
没有连接设备、读取保留集或S1标签，也没有改变线上权限。
