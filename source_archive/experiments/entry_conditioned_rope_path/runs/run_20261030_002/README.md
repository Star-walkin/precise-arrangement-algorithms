# 实际结果入口

先读 [REPORT.md](REPORT.md)。主比较为A/B第750步：B压低明确背景，但目标后段覆盖和拒识退化，未验证完整目标弧段身份。

查看图片：`comparison/visualizations/CONDITIONAL/`、`AUTO/`（每图全ROI＋固定4局部），`comparison/visualizations_training_v2/`为修正标题图例排版的实际监督及法向剖面，初版`training_checks/`保留。原始浮点：`branches/A/predictions/`、`branches/B/predictions/`。

从工程根PowerShell运行一个**尚不存在**的运行名：

```powershell
& .\experiments\entry_conditioned_rope_path\run_supervision_experiment.ps1 -RunName run_20261030_003
```

复用既有两个venv及旧`run_20261029_001`的冻结图像缓存、入口/scene、标签和权重；不下载、不访问保留集。逐阶段入口：

```powershell
& .\vision_prototype\.venv\Scripts\python.exe -B experiments/entry_conditioned_rope_path/run_supervision_experiment.py --run experiments/entry_conditioned_rope_path/runs/run_20261030_002 --stage infer --resume
```

已完成的gate跳过，不覆盖部分失败输出。训练预算按时间可能令复跑终点步数略有差异；仍取双方实际保存的最大共同步数。实际本轮训练调用由training receipts记录；infer/evaluate两个阶段已通过一键driver实际执行，完整all的新目录串联入口未为验收而额外重训第二遍。

源码快照和产物manifest在本目录；工程维护代码在上两级。弱标签不是独立真值；完整末端仍未知。所有响应固定0～1显示，未计算域NaN不是低置信概率。
