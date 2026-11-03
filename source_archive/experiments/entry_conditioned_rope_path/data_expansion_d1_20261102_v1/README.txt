自动弱监督数据扩充 D1 / IR-DATA-D1.0.0
启动2026-11-02，完成2026-11-03。离线研究，不替换旧基线。

先看 REPORT.txt；主图在 final_292（恰好292张无损双栏PNG）。
final_292_index.json 区分实际训练图、同训练来源未用帧、历史评价来源与其他来源。
原图各4096x3000不变；右侧是D1原始响应，固定0~1，无路径或补线。
网络实际计算2048x975的固定搜索框，映回4096x3000；这不表示网络原生4096分辨率。
301个入口分别保存在 inference/gallery/D1，不合并或按效果择入口。

数据：dataset/index.json、statistics.json；标签和原生复核图：labels_left/labels_right。
24新增图自动弱中心/198未知段；没有完整弯后或边缘身份真值。
模型与日志：branches/D0、branches/D1；共同step_0600由预定规则选择。
旧8图固定评价：evaluation/；先保存预测再读取参考：evaluation_paths/。
A750/D0/D1固定诊断：diagnostics_comparison/。

运行环境沿用 vision_training/.venv（Torch CPU）与 vision_prototype/.venv（图像）。
实际完整命令和逐项退出记录：run_pipeline.ps1、execution_receipts.json。
完整流程实际运行；最末原始审计exit1由OpenCV线程末位差造成，历史记录保留。
严格修复后78.07秒全量审计exit0：audit_repair_execution_002.json，
audit_repair_receipt_002.json；定位证据audit_thread_diagnostic.json。

命令从 D:\precise_arrangement 执行，具体训练和导出命令见ENGINE_README.txt。
不要在本运行目录重跑写入命令覆盖产物；训练、推理和渲染会拒绝已有目的路径。
重现实验应复制代码到新运行目录并显式保留同协议来源路径，先生成数据再训练。
只重导出时复用冻结D1浮点、模型元数据和render_export.py，在独立新目录导出；
不要用重新训练模型的响应替代本轮冻结响应。

可安全重新审计，另给新输出路径：
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/data_expansion_d1_20261102_v1/audit.py --workers 4 --output experiments/entry_conditioned_rope_path/data_expansion_d1_20261102_v1/audit_user_recheck.json

来源隔离、像素审计、训练拟合均不能替代完整活动绳身份准确率。
原图/历史模型/保留集/S1/设备均未改变；其余未知和失败保留。
