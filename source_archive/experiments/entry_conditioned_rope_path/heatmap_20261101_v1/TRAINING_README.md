# H7 热图训练部分

本目录在冻结 B 的第 750 步权重上做两个继续训练对照。它不增加目标正例，也不把 H6 路径作为标签。目标仍是当前入口指定绳段的稀疏中心响应；输出不是主体 mask，也不是校准的身份概率。

- C：原灰度标准化、原入口种子、零辅助通道。
- D：相同输入，再加浮点 LCN 辅助通道。
- 两组：旧阶段平衡高低响应 1:1 MSE 不变，邻绳弱负权重 0.10、背景弱负权重 0.05；新 AdamW，学习率 0.0002，最多 400 更新或 720 秒，CPU 8 线程、batch 1。
- 两组从同一 B 权重分叉，按相同原图采样、仿射、光度和噪声序列训练。LCN 不调用随机数。逐步保存前两输入通道、标签、有效域和负例域哈希进行精确比较。
- 10 张训练图来自 4 个来源组；8 张既有开放评价图来自另 3 个来源组。训练模块只解码训练缓存；真正保留集和 S1 不参与。

`network.py` 保留原两通道首层卷积，另设零初始化的一通道卷积并相加，种子仍在索引 1。两组初始函数均与 B 逐位一致；总参数从 135897 增至 135969。全局入口描述仍广播到瓶颈，未输入贴面路径提示。

预处理顺序：原图既定搜索矩形 `[0,1050,4096,3000]` → 按旧流程 area 缩小一半为 `975×2048` → 同步几何增强 → 非负灰度光度增强 → 全有效搜索域 LCN / 原灰度标准化 → 底部中性补边至 `992×2048`。实际 LCN 在半尺寸浮点图上计算，sigma 为 16 模型像素（名义对应原图 32 像素），不是先在原图做高斯后缩小。

LCN 使用 `Gaussian(M)`、`Gaussian(I*M)`、`Gaussian(I²*M)`，常量零外延并按权重归一化；sigma16、截断4σ、129点可分离核，标准差下限 `2.5/255`，辅助浮点响应截为 `[-3,3]`。M 是完整搜索域及几何有效域，不是 SAM2 或监督 mask。真实零灰度参加局部统计。灰度原通道保留，不经过显示用的 uint8 图；gamma 只作用于非负原灰度。

推理接口：

```python
from network import load_network
from preprocessing import prepare_input
model = load_network(checkpoint_path)  # 新 checkpoint 或旧 B，CPU eval
x, meta = prepare_input(gray_float_0_to_1, seed_float, checkpoint_config, roi=None)
response = model(x)  # [1,1,Hpad,Wpad]，按 meta['unpadded_shape'] 裁去补边
```

`roi=None` 表示输入搜索矩形全部有效。`checkpoint_config['variant']` 决定 C/D；单独重放旧 B 时使用 C 配置或对旧两通道网络直接使用 `x[:,:2]`。输入灰度与种子必须已经处于同一半尺寸原图坐标映射；接口不隐式缩放。

实际使用环境：`D:/precise_arrangement/vision_training/.venv/Scripts/python.exe`，PyTorch CPU。运行命令均在本目录执行：

```powershell
& D:/precise_arrangement/vision_training/.venv/Scripts/python.exe check_heatmap.py
& D:/precise_arrangement/vision_training/.venv/Scripts/python.exe train_heatmap.py smallcheck
& D:/precise_arrangement/vision_training/.venv/Scripts/python.exe train_heatmap.py train --variant C
& D:/precise_arrangement/vision_training/.venv/Scripts/python.exe train_heatmap.py train --variant D
& D:/precise_arrangement/vision_training/.venv/Scripts/python.exe train_heatmap.py compare
& D:/precise_arrangement/vision_prototype/.venv/Scripts/python.exe render_training.py
```

独立小检查最多 60 步或 120 秒，仅验证链路，权重不用于正式初始化。其冻结门槛是总损失下降、正中心未整体清空、过渡和贴面正监督存在、两负例组存在及数值有限。正式预算包含增强、LCN、哈希、更新、日志和 checkpoint；起止固定训练图诊断前向单独计时，原子更新可能轻微越过时间界限。

主比较取 C/D 最大共同保存步，不能将不同停止点当成配对比较。`common_training_fit.json` 保存同一步 B/C/D 的固定训练 10 图指标，`common_checkpoints.json` 保存路径和 SHA256。训练候选按协议中同一目标与正响应/负响应守卫选择，仍不等于独立识别改善。任何失败或守卫未通过都保留，不增加预算重跑。

旧模型、训练源码、标签及已冻结 H6 产物未修改。`legacy_source_readonly_receipt.json`、`numerical_checks.json`、每组环境与输入审计记录可复查；控制台、退出码、逐步损失、权重在相邻目录保存。未连接设备或改变任何设备权限。
