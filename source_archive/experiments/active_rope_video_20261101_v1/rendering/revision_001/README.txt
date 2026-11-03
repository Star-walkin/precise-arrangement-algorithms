四子图纯渲染模块

from rendering.render_panels import render_frame
canvas = render_frame(bgr, heat, engine_result, temporal_result, meta)

返回1600×1280 uint8 BGR：2×2，每格800×640，其中54px标题、586px原图视野。
输入原图保持比例居中，路径按同一原图坐标变换。不改变候选、门控结果或浮点响应。

engine_result.raw_candidates：最多3条，points_xy为原图坐标；可带point_status。
engine_result.surface_tracks：原图points_xy，可带逐点inferred；灰色历史卷面参考，推断用虚线。
兼容surface_coordinate_system='crop_half_xy'时需crop_bbox、crop_scale，默认[0,1050,4096,3000]和.5。

temporal_result：causal_gate.update的原返回值。
ACCEPTED/REACQUIRED必须有accepted=True及有效accepted_candidate_index，直接画该原候选，检查与points_xy一致。
绿色图像支持、黄色仅热图、蓝色A7、橙虚线弱/推断，与旧证据属性对应；任何颜色均不保证同根身份。
STALE只画reference_points_xy橙虚线，明确年龄和“非当前测量”；UNKNOWN不画已识别位置。
首次接受注明历史待积累；REACQUIRED注明新状态段，不能表示跨缺口同根恢复。

heat：B原始单通道浮点数组，通常975×2048；用meta.heat_bbox_xyxy映射原图ROI，默认[0,1050,4096,3000]。
固定0～1 TURBO色标，不逐图拉伸。ROI外/非有限像素灰色。None显示“无入口，热图未计算”，不伪装成零响应。
meta.heat_missing_reason可说明初始化前等其他未计算原因。

meta字段：
video_id、media_time_s、detection_time_s、source_frame_index、heat_age_s、detection_age_s、analysis_hz(默认2)、playback_fps(默认5)。
媒体时刻和检测时刻应使用同一播放器时基；物理/接收时间另存日志，不混作此处相减。
display_reference_age_s可直接给当前显示帧的旧参考年龄；否则使用门控参考年龄+检测年龄。
两检测帧间旧参考若已超过门控过期时间，由调用侧传display_status='UNKNOWN'；渲染模块不作新的接受/时序判断。

布局字体：Windows微软雅黑msyh.ttc。仅从字体文件读取排版信息，不读写媒体、不连接设备。
check_render_panels.py只生成合成示例，检查缺热图/STALE/UNKNOWN等显示契约，不代表真实识别效果。

实际检查：9项契约通过，退出0；两张最终合成预览已实际查看，回执visual_review_receipt.json。
虚线按显示坐标5px实/5px空，稀疏路径不再因采样间距产生长实线。首版保存在revision_000/。
render_check_failure_001.json保留新增虚线检查中的测试索引笔误及修复说明；未修改模型或检测算法。
