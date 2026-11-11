# 固定 E1 原图模糊＋梯度双边直段 S1：292 图离线实验

日期：2026-11-04。方法 `IR-GRAD-S1.0.0`。本轮属于活动绳自由直段的**条件式传统图像处理研究**，不是 P2/P1/H6 的替换或上线版本。

## 交付

- [292 张原分辨率四子图 PNG](<C:/precise_arrangement_results/straight_gradient_20261104_v1/all_001/final_292>)：每张依次为原始彩色图、σ=2.4 px 轻模糊图、固定 0–12 灰度显示的模糊后梯度幅值图、原图叠加结果。原图及另三个图像区均为完整 4096×3000 原图坐标；整板 8228×6414。青线是输入 E1，黄线为本轮双侧梯度有支持的中心，绿线是估计两侧，红点是最后有支持的截面。缺口分段绘制，不画成已确认的绳。
- [逐图导航 CSV](<C:/precise_arrangement_results/straight_gradient_20261104_v1/all_001/gallery_index.csv>)、[292 份预测 JSON](<C:/precise_arrangement_results/straight_gradient_20261104_v1/all_001/predictions>)、[全量描述统计](<C:/precise_arrangement_results/straight_gradient_20261104_v1/all_001/summary.json>)、[运行回执](<C:/precise_arrangement_results/straight_gradient_20261104_v1/all_001/run_receipt.json>)、[独立审计回执](<C:/precise_arrangement_results/straight_gradient_20261104_v1/all_001/audit_receipt.json>)。292 张 PNG 总计 6,848,654,376 字节（约 6.38 GiB）。
- 代码与复现：[核心](gradient_core.py)、[导出器](run_gradient.py)、[冻结配置](config.json)、[E1 输入清单](inputs.json)、[方法说明](METHOD.md)、[冻结协议](PROTOCOL.md)、[开发修订记录](revision_pilot003.md)、[冻结哈希回执](freeze_receipt.json)。大图放 C 盘独立目录，因 D 盘运行前仅余约 2.1 GiB。

## 输入、算法和实际运行

输入为与 D1/G1 相同的 292 张唯一开放源图，21 个采集来源组，右相机 178、左相机 114；每图仅采用已冻结的 E1 两点、从首点朝末点再向卷面延伸。292 图中有 D1 新训练图 24、同训练来源展示帧 208、旧开放评价来源展示帧 20、其他开放展示帧 40；相邻帧并非独立工况，也不是新的盲测集。原 400 图另 108 张旧入口失败，本方法在其上没有 E1 条件，分母限制仍在。未用网络热图、分割掩码、目标标签或新训练。

按 [方法说明](METHOD.md)，先模糊原图再用 Sobel 求梯度，在 E1 附近估计绳的光学宽度；之后每 4 px 测一次法向剖面，要求两侧反向梯度、绳体相对两侧背景的对比度及宽度一致。仅用最初 500 px 已受支持的截面做一次至多 2° 的直轴重估，最长允许 48 px 无支持间隔；持续消失时红点落在最后一次有支持的位置。抵边界和预算耗尽与梯度消失分开记录。红点是**图像规则定义的光学终点**，不是已验证的真实直绳尾端或入卷接触。

开发阶段仅看预定七个开发来源组各一张：`pilot_001` 在核心仍写入期间运行，哈希与后续版本不同；`pilot_002` 发现入口角度小偏差长期外推，导致一张右图早停；一次有界直轴重估独立保存为 `pilot_003`，同七图均按同一规则处理、审计 7/7 通过。`dual_20260924_111143__pair0315__right` 的末点从 E1 后 1012 px 变成 1252 px，但可见白绳仍有少量延伸，不能称末端已正确。其余六张的末点变化不超过 20 px。随后冻结 `config.json`、源码和清单，未按全量图逐张调参。

正式命令：

```powershell
& 'D:\precise_arrangement\vision_prototype\.venv\Scripts\python.exe' 'D:\precise_arrangement\experiments\entry_conditioned_rope_path\straight_gradient_20261104_v1\run_gradient.py' --phase all --config 'D:\precise_arrangement\experiments\entry_conditioned_rope_path\straight_gradient_20261104_v1\config.json' --workers 2 --run-name all_001
```

实际 292/292 处理完毕，0 异常，进程退出码 0，耗时 177.1 秒。所有 292 张输出状态为 `PARTIAL`、停因 `sustained_gradient_disappearance`；E1 后有支持长度最小 272 px、中位 1292 px、P90 1798 px、最大 2408 px。**这些只是算法产物统计，不是 292 张直绳识别正确，也不是终点准确率。**

独立审计对全部 292 源图重算 SHA，解码全部 292 张四子图，核对画布与栏位、每张原图栏对源图逐像素一致、预测 E1/点列/断段/终点/长度、结果与冻结代码/配置/清单哈希；292/292 通过，0 错误、0 警告。这证明输出完整和来源一致，不证明线画在目标绳上。冻结哈希：输入 `69f37567…8093850`、配置 `5012158f…732c3`、核心 `2f2dc6d0…bd700`、导出器 `3d442409…25e91`，完整值见 [冻结回执](freeze_receipt.json) 与 [运行起始记录](<C:/precise_arrangement_results/straight_gradient_20261104_v1/all_001/run_start.json>)。

## 图像观察和局限

- [冻结后定额目视抽查](<C:/precise_arrangement_results/straight_gradient_20261104_v1/all_001/review/visual_review.md>)按 21 个来源组各取清单中位图，加两张已知失败对照，共 23 张；[21 图联系图](<C:/precise_arrangement_results/straight_gradient_20261104_v1/all_001/review/contact_21.jpg>)与局部裁图已保存。21 张中有 3 张 E1 明确落设备，2 张多斜线身份待辨；这是定额抽查观察，不能外推为全量比例或准确率。
- [左 `pair0315`](<C:/precise_arrangement_results/straight_gradient_20261104_v1/all_001/final_292/dual_20260924_111500__pair0315__left.png>) 的自由斜绳有较长局部贴合；[右 `pair0345`](<C:/precise_arrangement_results/straight_gradient_20261104_v1/all_001/final_292/dual_20260923_101517__pair0345__right.png>) 也沿斜向纹理形成分段支持。这仅是目视局部几何证据。
- [右 `f000720`](<C:/precise_arrangement_results/straight_gradient_20261104_v1/all_001/final_292/R_run_20260923_01__f000720__right.png>) 的固定 E1 落在导杆/设备斜边，黄线沿设备而非下方真实活动绳；[右 `pair0015`](<C:/precise_arrangement_results/straight_gradient_20261104_v1/all_001/final_292/dual_20260923_101517__pair0015__right.png>) 同类错误重现，[右 `pair0075`](<C:/precise_arrangement_results/straight_gradient_20261104_v1/all_001/final_292/dual_20260923_101517__pair0075__right.png>) 则沿工人手持直尺/杆件。它们仍保留于 292 张结果，不改输入、不从统计中删去。
- 部分图的双边梯度穿入卷面投影或旧圈纹理后仍持续，红点可能过晚；背景变亮、遮挡或直绳轻微弯曲时也可能过早消失。梯度的存在既不能确认同根，消失也不能直接定义物理直绳结束。本轮没有逐图人工可信末端标签，不能计算错误率、末端偏差或相对 U-Net/A7/P2 的净增益。

本轮没有读取正式保留集或 S1 人工标签，没有连接相机、光源、PLC、伺服或修改设备权限；原图、P2/P1/H6 与旧失败产物未改动。下一步若要比较此法是否优于 U-Net，应先独立标注 E1 是否落在目标绳和可见自由直段末端，再按来源组隔离评价；不能用本轮 `PARTIAL` 或红点直接作 t0/t1/t2 控制依据。
