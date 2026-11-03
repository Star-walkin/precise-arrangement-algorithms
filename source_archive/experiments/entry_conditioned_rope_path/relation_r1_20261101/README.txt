R1 离线实验说明
主交付：交付_292张原图与热力图/  恰好292个原像素两栏PNG。
原始浮点响应/<源图>/entry_00.npz 的response键为未叠加975×2048 float32；features是冻结A的8通道空间缓存，非标签。通过NTFS硬链接复用本轮不可变cache，避免重复磁盘占用。不要分别改写链接副本。
原图=2×模型裁框坐标+[0,1050]；图像回映采用双线性align_corners=False。
模型与训练日志：model/。冻结输入：protocol.json、cache_jobs.json、pair_manifest.json。运行与评价：h6_control/、development_001/、fixed_001/、batch_002/、evaluation/。
失败与性能修复保存在batch_001/、batch_001_interruption.json、cache_all_initial_failed.log、checks.json，不覆盖为首次成功。
全部源码为纯离线。训练用vision_training/.venv，图像和路径用vision_prototype/.venv，不安装依赖。
实际命令入口及顺序如下，均在D:\precise_arrangement运行。复现请先把本目录py文件复制到同级新实验目录并更换相应目录名，勿覆盖现有冻结运行：
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/freeze.py 
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/cache_a.py --subset all --workers 4
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/cache_a.py --subset fixed --workers 2
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/prepare_pairs.py 
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/train_relation.py 
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/control_h6.py 
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/run_paths.py --subset train --branches R0 R1 --out development_001 --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/audit_training.py 
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/freeze_selection.py 
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/run_paths.py --subset fixed --branches R0 R1 --out fixed_001 --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/run_paths.py --subset gallery --branches R1 --out batch_002 --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/render_batch.py 
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/evaluate_fixed.py 
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/check_r1.py 
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/make_review.py 
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/relation_r1_20261101/audit_delivery.py 
结论：局部几何覆盖有变化，完整同根识别未通过；尤其保留明显跨圈错误，详见REPORT.txt。
