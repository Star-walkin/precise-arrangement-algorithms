因果活动绳候选门控 / temporal

用途
只处理当前单帧算法已经提出的最多3条候选，使用此前最多5个已接受结果及2s时间范围估计有界位移参考。
模块不读取图像/标签、不接设备、不生成热图、不改动H6轨道，不以历史轨迹补出当帧识别。
时序稳定、规则接受及重捕获均不等于同根身份正确，也不提供t0/t1/t2证据。

接口
from temporal import TemporalRopeGate, GateConfig
gate = TemporalRopeGate()
out = gate.update(source_time_s, candidates, pitch_px=local_projected_pitch_original_px, scene_cut=False)

candidates为按当前图像基准排序的最多3个字典：
  id: 仅当帧标识，不假设跨帧轨道ID一致；
  points_xy: 原始图像像素、从入口到可见尾端的有序xy，至少80px有效长度；
  image_score: [0,1]内单调归一化的当前图像排序分，不是真实概率；
  image_supported: 上游确认存在当前图像支持的布尔量；历史推断自身不能设置为True。
传入超过3候选、倒序/重复/非有限时间戳会报错。换视频须新建gate或reset()；禁止沿用其他相机历史。

返回
ACCEPTED：当帧几何通过当前图像/时序门控；初始帧 temporal_confirmed=False。
REACQUIRED：旧结果已超时，独立待确认池连续3次、持续至少0.4s的当前图像候选建立新epoch；不宣称跨缺口同根身份。
STALE：当帧被拒；points_xy=None；reference_points_xy仅为最多1s旧参考，应显示虚线、年龄和“非当前测量”。
UNKNOWN：当帧无可用接受结果且旧参考过期，points_xy和reference_points_xy均None，不应继续画为当前识别。
accepted_candidate_index对应输入候选索引；输出采用原候选全部原始点，不做时序平滑、不改变F3中心约束。
reference_age_s、candidate_diagnostics、history_updated、epoch和uses_future可记录以复查。

比较方式
路径入口切向投影仅对齐轴向起点小差异；横向偏移不配准掉。不使用最近轨迹匹配、固定轨道ID或强制平行补线。
在双方共同可比较弧长内取96点，检横向p95与最大偏离、轴向形状变化、入口横移及方向；共同长度至少80px，至少覆盖较短路径55%。
未重叠尾长单独报告为uncompared，不能把缩短的可见范围当成全路径跳动，也不能把该范围当连续验证成功。
历史速度只来自已接受帧，取过去最多5帧/2s的稳健位移中位数并限幅。
异常候选进入独立pending池，不加入接受历史；若原位置恢复，则pending清除。
2.5s以上输入时间空缺或显式scene_cut重置；真实源时间差用于规则，不用CPU处理时间冒充视频时间。

固定默认参数与单位
3s/投影间距来自用户近似描述，速度上限为1.5倍；它是启发式，不是材料速度标定。
横向允许创新=pitch*(0.35 + 1.5*min(dt,0.75)/3)；最大局部偏离不得超过该值2倍。
pitch优先传当前F3局部投影间距（原图像素），本模块夹在8～64px；已有历史时单帧尺度变化限于0.8～1.2倍历史中位数。
缺少pitch则明确标记24px未标定fallback，不能将其称为真实绳径。2Hz、24px时横向创新门限14.4px。
最大方向变化15°；入口轴向变化4 pitch，轴向残差2 pitch；不因拒绝持续变长而无限放宽位移门限。
完整配置可由dataclasses.asdict(GateConfig())导出；实际视频若使用其他参数需另外冻结并记录。

对照与限制
保留同一候选池按单帧image_score选出的未加时序结果，与门控结果逐帧比较。不能用低抖动当识别准确率。
任何稳定错误都有可能通过；连续新候选重捕获也可能是另一根绳，故新epoch不得宣称同根恢复。
短尾未比较区、自由投影跨圈及上游误入口不会被此模块自动变成正确物理接触。

已实际运行
D:/precise_arrangement/vision_prototype/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v1/temporal/test_causal_gate.py
11项机制测试：预期平移、孤立跳变、低分连续候选/邻圈竞争、持续变化新epoch、遮挡过期、场景/时间重置、逆序时间原子拒绝、尾长变化、大尾部偏离、未来因果性、候选上限/无效几何。
test_receipt.json保存实际退出码、参数、源码哈希；这些是合成软件测试，不是视频身份准确率。
