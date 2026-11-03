# V28 定向复核与必要测试

本文件记录本轮新增模块的独立复核，不代表现场识别成功。只读明确的开发资料和源码，不运行旧全库测试，不解码保留图像或读取保留标签。

## 复用模块的导入安全

`rope_detector.py`、`arc_detector.py`、`geometry.py`、`wound_rope_tracker.py` 顶层仅标准库、OpenCV、NumPy。`v16_video_measurement.py` 顶层仅标准库和 NumPy；复用的 `automatic_seed()` 惰性导入 `wound_rope_tracker._entry()`，不打开视频、模型或设备。该文件中的视频 `freeze/extract` 与模型 `infer` 不由 V28 调用。

特别注意：`geometry.rope_exclusion_band()` 返回整条直线走廊，并非只屏蔽可见自由段；新实验的原图证据评分不得使用被屏蔽或补画的图。`geometry.write_image()` 不自带防覆盖，因此由新运行目录的排他创建保护输出。

## 已运行的旧接口回归

工作目录：`D:\precise_arrangement`。

```powershell
.\vision_prototype\.venv\Scripts\python.exe -m unittest vision_geometry_lab.tests.test_v16_video_measurement.VideoMeasurementTests.test_auto_seed_orients_by_observed_arc vision_geometry_lab.tests.test_v16_video_measurement.VideoMeasurementTests.test_unresolved_width_never_defaulted vision_geometry_lab.tests.test_rope_detector.RopeSyntheticTests.test_roi_returns_original_coordinates vision_geometry_lab.tests.test_rope_detector.RopeSyntheticTests.test_empty_and_invalid_inputs -v
```

实际退出码 0；4 项通过，0 项失败，0 项跳过。输入为合成图或内存数据。没有运行 `test_rope_detector.py` 的 `RopeRepositoryTests`，因为其中存在历史 holdout 图像读取。

## 复核开始时复用源码 SHA-256

| 文件 | SHA-256 |
| --- | --- |
| rope_detector.py | 6d90427900284332f032efd06abd2951d23b8b08df32e6a47a3fc4dd8bcdebf0 |
| arc_detector.py | d0cde99802a3fe6af23736878927ace8fe90adac92a2d09b1407bb6085953bae |
| geometry.py | 1557b81ac00a851785d62427195be46e5b92ff5d6eff1c3e74f80f6603668992 |
| wound_rope_tracker.py | 1b4c9d803bafbfed4433d5a2e96441b8afdae4ba08c9e6fd9e741d5fc578d11a |
| v16_video_measurement.py | dddd0a4c6faeedd1cc58e2fcac163ac5c1d256bb1cea8a5a6c8ffb5ccf7017aa |

## V28 新接口针对性测试

```powershell
.\vision_prototype\.venv\Scripts\python.exe -m unittest vision_geometry_lab.experiments.v28_local_spool_connection.test_v28_contracts -v
```

本轮最终实际执行：退出码 0；17 项通过，0 项失败，0 项跳过。旧接口 4 项与新增 17 项为不同测试；未复制历史测试数。输入均为合成图、内存字典或临时输出文件，合成通过不计入现场成功。

覆盖以下契约：

- 任意旋转条带到原图的坐标往返、带偏移 ROI 的原图范围。
- AUTO 清单仅返回影像元数据；AUTO 检测、候选与评分不访问条件提示或目标标签；评价器在目标标签加载前核验候选已落盘且一致。
- G0/G1/G2 共用同一个候选池，评分不改候选；候选输入顺序反转不改变成本或非并列排名。
- 共同自由直段不进入关键分离区；在抹平的原图上，候选自身声明的支持不能作为 G2 图像证据。
- 全黑、均匀低对比、饱和、单边剖面均不产生测得双边支持；支持明暗两种极性，原像素不被改写。
- 原图确有线结构时，允许与旧行连接和跨过旧行的新段同时存在，不强制接旧行。
- 未知尾段不算正负真值；只有中心线贴合而无可信双边时，身份评价保持未知；合成的可信双边与分离竞争分支能区分正确/错误接法。
- 自由入口成功但空候选与真正入口失败分别统计。
- 导入与 `automatic_seed` 调用时阻断视频捕获、模型及硬件依赖，确认无此访问；复用旧源码哈希保持不变。
- 运行目录和 JSON 文件均排他创建，已有内容不被覆盖。

独立源码复核发现一处统计口径问题：初稿把 `bool(candidates)` 当作入口成功，混淆了入口成功后无候选的情况。已在首次真实评分前由数据模块作者改为优先读取真实 `automatic_entry_success` / `entry_available`，runner 明确填写入口状态，并新增回归测试。未对评分权重、门槛或单图配置做调整。

本轮没有运行会访问历史保留图片的测试类，没有运行全库、设备接口或新网络；没有修改复用旧文件。
