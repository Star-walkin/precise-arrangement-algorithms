活动绳视频时序 V1 / IR-V1.0.0

最终四子图视频：本目录/四子图视频（全部完成并finalize后生成）。
原始视频：D:\precise_arrangement\数据资产\01_原始采集\现场视频\训练采集
精确16文件白名单与400图映射：source_trace/。
图像基准：D:\precise_arrangement\工程导航\05_实验版本档案\活动绳整链基准_P1_20261101
P1为原B750＋H6整链档案，区别于历史局部短缺口连接器P1。H7 C/D不纳入。

无需安装依赖。沿用vision_training/.venv Python3.14.6、PyTorch2.14.0+cpu、NumPy2.5.2，
从既有vision_prototype/.venv/Lib/site-packages追加OpenCV5.0/Pillow；实际导入及数值复放均已验证。
CPU 4进程×每进程4个Torch线程，OpenCV线程1。没有CUDA、设备或保留集访问。

实际命令（工程根目录PowerShell，运行目录不得复用或覆盖）：
vision_prototype/.venv/Scripts/python.exe -B 工程导航/05_实验版本档案/活动绳整链基准_P1_20261101/verify_baseline.py
vision_prototype/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v1/temporal/test_causal_gate.py
vision_prototype/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v1/rendering/check_render_panels.py
vision_training/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v1/run_videos.py --run integrated_pilot_001 --pilot --workers 3 --threads 4
vision_training/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v1/run_videos.py --run integrated_pilot_002 --pilot --workers 3 --threads 4
vision_training/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v1/run_videos.py --run all_001 --workers 4 --threads 4
vision_training/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v1/finalize.py

*.log保存实际控制台，运行子目录protocol.json在执行前记录源码哈希，summary.json只在运行结束写入。
engine是新帧适配器；temporal是因果门控；rendering只负责显示，不能改变候选或接受结果。
先前失败和修订保存在frame_engine/pilot_001及rendering/revision_*，没有覆盖后冒充首次成功。

显示约定
第一栏：原视频。第二栏：原始B响应固定0～1；E编号指定当前条件入口，不混合不同入口热图。
第三栏：历史F3场景参考及当前彩色候选，候选标C#/E#。灰色场景线不是当帧重新跑F3的结果。
第四栏：时序接受的原始候选；绿=原图支持，黄=热图支持，蓝=A7，橙=弱证据。
STALE橙虚线仅旧参考，最多1秒；UNKNOWN不冒充当前识别。结果年龄和检测媒体时刻分别标注。
无入口则热图未计算，灰底显示，不伪造全零热图。预先锚点之前为未初始化。

5fps是视频原速展示采样，不是在线算法帧率；2Hz是源时间分析采样，实际CPU耗时另报。
时序门控可拒绝异常跳动，但稳定错误也可能通过，重捕获不保证跨缺口同根身份。
首锚场景先验仍是依赖：本次不声称任意新视频无先验冷启动，也没有逐帧完整重跑SAM2/F3。
