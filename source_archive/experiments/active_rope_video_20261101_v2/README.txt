活动绳视频逐帧 V2 / IR-V2.0.0

修改目的：修复V1灰色中心线来自首锚并仅作整体配准的问题。
每个源视频帧均重新提取当帧沟槽边界，再由相邻边界求中点；当前活动绳接续只使用当帧新中心线。
有入口时逐帧调用原B750模型，使用完整浮点响应做H6路径选择；无入口也继续检测卷面中心线。
输出按源视频自身10/约30fps，不再按2Hz检测/5fps显示。
P1、V1原目录和模型不改，新结果在本目录/四子图视频。

检测实现
fresh_surface.py采用半分辨率当前灰度、ROI加权局部均值/标准差、沿宽纹理方向积分、
原图双侧亮肩检验、有限沟槽行序约束、相邻边界配对中点。没有用上一帧中心坐标填回。
旧C4/SAM2结果只提供离线静态搜索ROI，C3共享二次形状只作粗方向参考，像素绳距是未标定尺度先验。
这是一套为视频实现的轻量新观测器，不声称逐帧原封不动重跑完整F3历史融合链。
ROI不是每帧重新运行SAM2；大范围相机运动或ROI真实变化仍需后续场景更新。
原图内暗零灰保留，ROI不使用I>0重建。无当帧证据处不输出旧灰线。

时间处理与并行
按20秒块并行当前图像测量，块前最多1秒历史画面仅作自动入口预热；预热不重复写入最终视频。
随后每段视频严格按原帧序统一进行时序门控，历史跨块连续，拒绝不进入接受历史。
没有按未来候选决定当前路径。离线ROI/方向配置的来源另存engine_receipt，不能混同目标位置历史。
N按每段fps取max(5,ceil(0.5*fps))，覆盖约半秒历史；重捕获帧数同步适配，仍要求持续时间。
此适配避免原低频门控在30fps下因待确认队列过短而无法满足0.4秒条件。
细直入口LK孔径不确定时，在当前原图沿法向±4px重新检查短入口；通过当前双边对比后才使用。
模型权重、灰度裁切、标准化、H6评分及不跨当前中心线约束沿用。

缓存
检测全程使用975×2048原始float32响应；每帧原响应SHA256保存到测量记录。
每5秒保存完整float32 npz留样。逐帧显示缓存是800×381、0～65535固定量化PNG，
只供成片显示，不回流检测；无逐图拉伸。每帧的灰线、原候选和当帧对应关系保存于gzip JSONL。

实际命令（D:\precise_arrangement，既有环境，不安装依赖）
vision_training/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v2/probe.py
vision_training/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v2/run.py --run pilot_001 --pilot --workers 3 --threads 2
vision_training/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v2/run.py --run pilot_002 --pilot --workers 3 --threads 2
vision_training/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v2/audit.py --run pilot_002
vision_training/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v2/run.py --run all_001 --workers 8 --threads 2
vision_training/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v2/audit.py --run all_001
vision_prototype/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v2/check_seek.py
vision_training/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v2/finish_when_ready.py

finish_when_ready.py仅等待本次正在执行的固定all_001完成，随后顺序执行审计与finalize.py，
实际子命令退出码写入postflight_execution_receipt.json；不建立定时任务，不连接设备。
seek_verification.json实际核验了16视频全部65个块起点：随机定位读帧与顺序解码像素完全相同。
make_review_sheets.py只读取已保存的结果预览，固定每段3秒/中点/末尾。16段48位置实际查看后，
attach_review.py将visual_review/review.json中的真实复核记录和哈希附到报告；不自动伪造目视结论。

probe_001的显示字典重复键错误、pilot_001的细绳LK失效与pilot_002当前原图检查修复均保留。
平移与黑图检查只证明新观测响应当前图像，不证明逐根定位准确。完整接续、弱热图及入口错误仍可能失败。
CPU离线运算；源帧率输出不代表在线计算实时。无训练、设备连接或保留集访问。
