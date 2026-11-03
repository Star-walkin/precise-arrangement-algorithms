# 实际运行命令与环境

工作目录：D:\precise_arrangement。图像/后处理使用 vision_prototype\.venv\Scripts\python.exe；仅冻结前向使用 vision_training\.venv\Scripts\python.exe。无训练命令。

新运行应另建目录；当前输出是冻结证据，不覆盖重跑。先完成数据准备/训练侧校准并冻结p1_config.json、p1_freeze.json及evaluation_protocol.json，再正式运行。

```powershell
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_batch.py prepare --run experiments/entry_conditioned_rope_path/postprocess_20261030_001
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_batch.py infer --run experiments/entry_conditioned_rope_path/postprocess_20261030_001
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_run.py formal --run experiments/entry_conditioned_rope_path/postprocess_20261030_001
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_run.py batch --run experiments/entry_conditioned_rope_path/postprocess_20261030_001 --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_render.py --run experiments/entry_conditioned_rope_path/postprocess_20261030_001
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_evaluate.py --run experiments/entry_conditioned_rope_path/postprocess_20261030_001
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_visualize.py --run experiments/entry_conditioned_rope_path/postprocess_20261030_001
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_missing_entries.py --run experiments/entry_conditioned_rope_path/postprocess_20261030_001
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/example_postprocess.py --run experiments/entry_conditioned_rope_path/postprocess_20261030_001
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_verify.py --run experiments/entry_conditioned_rope_path/postprocess_20261030_001
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_report.py --run experiments/entry_conditioned_rope_path/postprocess_20261030_001
```

准备、校准和原图审计的细节见data_audit.json、calibration_report.json与calibration/frozen/receipt.json。正式分组、参考口径见evaluation_protocol.json；路径和逐边拒绝理由在formal/及batch/conditions/；提取缓存p1_cache/只含派生数据。

源代码：p1_segments.py为新算法；postprocess_batch.py为冻结数据/前向；postprocess_run.py重放P0与运行P1；postprocess_evaluate.py独立评分；postprocess_visualize.py和postprocess_render.py绘图。旧connect.py、model.py、训练实现未修改。