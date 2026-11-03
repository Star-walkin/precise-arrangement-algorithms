热图接续 H4 / IR-H4.0.0 — B热图重新估计直段，避免提前跑偏接旧圈

图库：D:\precise_arrangement\experiments\entry_conditioned_rope_path\postprocess_b_20261031_v4\all_001\四栏检测结果
实际292/292张、301入口、退出0、260.6秒。原400张中108张旧入口未解决，不称无活动绳。
四栏：原图 / 固定0～1的B热图 / 竞争路径 / 最高决策效用候选。原模型和301份浮点响应未改，无训练、新前向。
产物类别：{'straight_free_to_F3_center_track': 130, 'free_fragment_no_resolved_connection': 162}。类别数量不是完整识别率。

先用有限角度和法向位置搜索重估B直脊，再决定有限可见延伸；A7只提供初轴/宽度，不锁死中心与旧端点。
绿色为原图+B局部支持；黄色为B双侧热形态单独支持；橙虚线为弱证据/推断。均不等于物理同根真值。
F3相邻边界中心轨迹保持源点；接轨不能舍弃持续的B直绳。近端缺乏B形态时保留自由段，不强制选平滑灰圈。
自由段不接轨是显式候选，分数为固定决策效用，不能当真实概率。
已知失败：左视角L_run f000015和f000142图中可见绳仍继续，但输出短段未覆盖；不是标成未知就算正确。更长/更强响应不能直接解释为同根接续成功。
疑似过延伸：左dual_20260924_111500 pair0015直轴尾伸到卷面右侧之外，黄/橙支持限定保留，未当成已确认完整路径。

记录：all_001/记录/*.json.gz，半尺寸crop坐标xy映射原图=2*xy+[0,1050]。
配置：inputs.json、freeze_final.json、all_001/run_start.json；source_snapshot为实际运行代码；progress.jsonl逐图运行日志。
环境：原vision_prototype/.venv，CPU后处理；原始文件、模型、冻结H1/H2/H3不改；无标签/保留集/设备操作。
固定旧12例加用户R_static_20260922_01__f000095__right反例共13例开发，之后一次全量。
pilot_001短亮段均值偏置反例保留；唯一修订将评分改为共同2000原px视野的支持积分，细化保留一条最长合格轴，阈值/搜索预算不变；pilot_002后冻结。

实际运行命令从D:/precise_arrangement：
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v4/run_h4.py --run pilot_001 --subset pilot --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v4/run_h4.py --run pilot_002 --subset pilot --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v4/run_h4.py --run all_001 --subset all --workers 4
复跑使用新run名，不能覆盖旧运行。

同条件H3/H4：自由总长度-1.50%，自由热形态支持长度+18.46%，自由原图+热联合支持长度+14.27%；所选整路径联合支持长度-7.82%。按同一诊断门槛比较，不能换算真实定位准确率。完整贴面身份与末端未验证。
