# 原图几何与背景方向 G0/G1 全量图库目视抽查

日期：2026-11-03。证据入口：[all_001 摘要](C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/summary.json)、[运行收据](C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/run_receipt.json)、[输入清单](D:/precise_arrangement/experiments/entry_conditioned_rope_path/geometry_direction_20261103_v1/inputs.json)。仅抽查已开放的开发来源组；292 图库中 20 张历史评价来源组展示行及独立历史 8 张开放评价图均未目视读取。无标签、保留集或设备访问。

## 口径与检查方式

`all_001` 实际完成 292/292，退出码 0，无运行失败。摘要记录 G0：266 `PARTIAL`、26 `AMBIGUOUS`；G1：265 `PARTIAL`、27 `AMBIGUOUS`。257 张两者都选中，18 张都未选中，8 张仅 G1 选中，9 张仅 G0 选中；两者都选中的 257 张中有 25 张选中路径点或状态不同。G1 选中路径的 32,276 个节点证据中，3,179 个节点有背景方向，583 个获得正加分，单步最高 0.0544（规则上限 0.08）。这些是产物和评分统计，不是同根识别率。

从 `summary.json` 按左右视角、近似平行、状态变化、端点分离和歧义选择以下 9 张开发来源图。查看正式 `final_292` 图（左栏未绘制原图、右栏 G1 响应/选中路径），并将保存的 G0/G1 `selected.points_xy` 临时叠加在原图内存中比较：G0 为红、G1 为青，终点同色圆点；这项临时叠加没有写入或覆盖产物。下表“案例图”均为正式全量 PNG；G0/G1 逐点证据在相应预测 JSON。

| 开发案例与正式图 | G0/G1 结果 | 原图可见现象与边界 |
| --- | --- | --- |
| 左侧斜段转弯：[L_run_stop_run f003950](C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/final_292/L_run_stop_run_20260922_01__f003950__left.png) | 都 `PARTIAL`；终点相距 732.1 px。G0 沿卷面条纹向右延伸，G1 在转弯附近较早停止。 | 两者的自由斜段前缀贴近原图可见绳；弯后区域有密集旧圈，无法凭此图确认 G0 的长延伸同根，也不能把 G1 较早停止算作正确接续。 |
| 左侧清晰斜段：[dual 111500 pair0555](C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/final_292/dual_20260924_111500__pair0555__left.png) | 都 `PARTIAL`；终点相距 812.6 px。G0 朝黄色法兰上缘转去；G1 沿卷面纵向亮斑到图底，停止原因为 `image_border`。 | 可见自由斜段前缀被覆盖，但分叉后两条都缺少明确的同根图像支持，G1 的纵向跨圈延伸尤其有误接嫌疑。更长路径和到达图边不能视作改善。 |
| 右侧斜段：[dual 111143 pair0585](C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/final_292/dual_20260924_111143__pair0585__right.png) | 都 `PARTIAL`；终点相距 670.6 px。G0 转弯后短向左，G1 沿纵向亮纹到图底。 | 自由斜段局部贴合可见绳；贴卷之后 G1 穿过多条近水平旧圈纹理，缺少同根依据，属于明显需要拒识或独立核验的尾段。 |
| 近似平行：[L_run_stop_run f003857](C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/final_292/L_run_stop_run_20260922_01__f003857__left.png) | G0/G1 路径完全相同；G1 有 29 个已知背景方向节点，但方向加分总和为 0。 | 斜绳进入卷面后与局部旧圈方向接近，方向项保持中性；两者在可见前段后停止。此例只证明该处没有因近平行方向而加分，不能证明末端或身份正确。 |
| 右侧错入口：[dual 101517 pair0015](C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/final_292/dual_20260923_101517__pair0015__right.png) | 两者同一路径且均 `PARTIAL`，方向加分 0。 | 原图有一根明显的浅斜白绳跨过卷面；输出却沿右侧黑色导杆/设备线，未覆盖该白绳主体。该图暴露冻结输入入口可错，后处理在错入口上仍会给出 `PARTIAL`。 |
| G1 单独选中：[L_run f001250](C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/final_292/L_run_20260923_01__f001250__left.png) | G0 `AMBIGUOUS`，G1 `PARTIAL`。 | G1 的可见斜段前缀贴近原绳，但之后转向卷面纵向亮纹；分数打破歧义并不证明挑中了目标同根。 |
| G0 单独选中：[dual 111500 pair0405](C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/final_292/dual_20260924_111500__pair0405__left.png) | G0 `PARTIAL` 并延伸至图底；G1 `AMBIGUOUS`，保留后段竞争路线。 | 两者都有自由斜段前缀，随后围绕卷面亮纹分歧。G1 未单选避免了一条无依据的确定尾线，但拒识不计正确，同根走向仍未知。 |
| 双方歧义：[dual 101517 pair0435](C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/final_292/dual_20260923_101517__pair0435__right.png) | G0/G1 均 `AMBIGUOUS`，均无选中路径。 | 候选共用右侧导杆附近的可疑入口，随后进入卷面；原图白色斜绳与候选入口并不一致。没有选中是保留多解，但候选池可能已缺正确分支。 |
| 右侧大分歧：[R_run_stop_run f002149](C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/final_292/R_run_stop_run_20260922_01__f002149__right.png) | 都 `PARTIAL`；终点相距 735.9 px，G1 方向累计加分仅 0.0205。 | G0 由斜段转向卷面下方，G1 则转往上方黄色法兰附近。二者都未由单帧证明与可见白绳同根；很小的方向分数仍能改选后段，不能把排名变化当净增益。 |

对应逐点记录：[G0 预测目录](C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/predictions/G0)、[G1 预测目录](C:/precise_arrangement_results/geometry_direction_20261103_v1/all_001/predictions/G1)，各案例文件名与上表 PNG 主名相同，扩展名 `.json`。

## 可确认结论和未知项

1. 在若干左、右图中，原图几何确实能沿可见自由斜段形成局部中心候选；f003950、pair0555、pair0585 的入口后短前缀有可目视对应关系。
2. 到达卷面转弯后，旧圈重复纹理、纵向高光和法兰边缘仍会吸引路径。pair0555、pair0585 显示 G1 可把跨多圈的纵向亮纹判作长的 `PARTIAL` 支持；`image_border` 是预算/边界停止原因，不是物理绳末端。
3. 冻结 E1 错入口会直接改变问题；pair0015 和 pair0435 的可见白绳未被所选或保留候选完整覆盖。G0/G1 的比较只是在同一入口条件下比较后处理，不能归因于背景方向解决入口错误。
4. 近似平行的 f003857 中方向项按设计中性；其他大分歧中，软项虽小也可改变近分候选排序。全量 G1 的 `PARTIAL` 数没有较 G0 增加，目视样本不支持总体同根改善结论。
5. 本次是定向质性抽查，没有逐根真值或独立评分，无法给出整根准确率、真实接触点、t0/t1/t2 或生产可用性。20 张图库中历史评价来源组展示行及独立 8 图仍按冻结评价流程处理；本报告未用它们调参或挑图。

本次仅新增此审阅记录；冻结的 `geometry_core.py`、`run_geometry.py`、配置、全量预测及图像均未修改。
