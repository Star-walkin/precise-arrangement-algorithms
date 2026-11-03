# V28 局部卷面结构与有限接续

这是现有工程的离线开发实验，不是新盲测，不产生 t1/t2 或控制动作。只使用历史公开的 2026-09-23 八张开发图片；固定白名单在 `data_eval.py`。旧源码和历史输出保持原样。

在 `D:\precise_arrangement` 的 PowerShell 运行，输出目录必须不存在：

```powershell
.\vision_prototype\.venv\Scripts\python.exe vision_geometry_lab\experiments\v28_local_spool_connection\run_experiment.py --output vision_geometry_lab\outputs\v28_local_spool_connection\v28_20261007_reproduce_001
```

脚本已实际运行的命令、退出码及逐图耗时保存在各运行目录。执行次序：保存配置/八图白名单/源码哈希 → 全部 AUTO 检测、保存候选、同池评分 → 使用已有自由段提示另跑 CONDITIONAL → 最后打开目标标签评价 → 生成可双击的 `index.html`。原图与输入哈希会校验，已有输出拒绝覆盖。无需网络服务。

`--skip-conditional` 可只跑自动输入；`--verify --output <已有目录>` 只核验冻结源码、配置和八图原文件。

本轮已完成，推荐直接双击 `vision_geometry_lab\打开V28接法对照.cmd` 查看最终修正版。重现实验后，下面命令把图层限制在照片矩形内（不修改评分，输出目录必须不存在）：

```powershell
.\vision_prototype\.venv\Scripts\python.exe -m vision_geometry_lab.experiments.v28_local_spool_connection.render_layout_fix --source vision_geometry_lab\outputs\v28_local_spool_connection\v28_20261007_reproduce_001 --output vision_geometry_lab\outputs\v28_local_spool_connection\v28_20261007_reproduce_001\layout_v2
```

## 算法及证据边界

- AUTO 复用 `rope_detector.detect_rope`、`arc_detector.detect_arc`、`automatic_seed`。原图检测可见终点不叫真实切点。检测失败仍计八图分母。
- 局部旋转条带前向90个光学宽度、最多2400px，侧向±18宽。复用 V10 多尺度亮/暗脊线及 V12 原图双边精修，最多6条弧结构，保留暗槽/绳体和半相位歧义。不把旧 V27 的约29px纹理周期当整绳距。
- 与旧结构的接续用有限三次曲线，过长的上游间隔仅标几何推断。另用原图双边剖面的有限束搜索保留最多2条独立新段路径；允许跨旧排列，不据投影交叉判定三维爬层。总池最多8条，去重/截断记日志。
- 全部候选先落盘，三种评分共享完全相同的池。G0看端距和方向；G1增加曲率、光学宽度连续性、沿行次序和推断惩罚；G2增加候选实际分开区域的原图双边、剖面及可见断开代价。候选提取本身已用图像，所以G0不是整个系统完全不看图像。
- G2只取候选横向差异≥0.65宽的前20宽区域，共同长直段不贡献区分证据。这是图像候选自身定义的评分区，不是人工关键弯段；评价器另外检查可信弯段。缺边界、饱和、深暗或低对比记未知，不由补全曲线制造纹理或支持。
- 所有路径以同样原图法向剖面预算评分。宽度、法向偏移±0.1宽及方向±0.6°为预先声明敏感性；候选池保持固定。分差、稳定率均不是身份正确概率。
- 证据少、近并列或排名不稳时拒识。SINGLE_CANDIDATE只表示算法选择。新段候选和暗谷的双边支持都不证明整绳身份、前后遮挡顺序或爬层。

## 单帧评价

仅评价已复核中心线/双边的可信范围，质量排除区和未知尾段不作正负例。沿用旧5px弯段覆盖作几何诊断；另以人工双边形成的目标绳带核查可见接法，只有足够可信双边且竞争候选能在分歧区被该绳带排除时才评价分支。像素贴近和可见绳带匹配都不自动升级为完整材料身份。

CONDITIONAL 仅从已有标注读取独立 `free_seed`，宽度重新在原图测量；不把完整目标中心线加入候选池。它与 AUTO 分表统计。

每图保存 `entry.json`、`structure.json`、`pool.json`、`scores.json`；根目录有 `results.csv`、`evaluation.json`、配置和哈希。报告图实线表示算法认为有原图双边支持，虚线表示推断/无可用支持，绿色人工参考只在事后评价图层出现。干净原图始终可单独查看。

必要测试：

```powershell
.\vision_prototype\.venv\Scripts\python.exe -m pytest vision_geometry_lab\experiments\v28_local_spool_connection\test_v28_contracts.py -q
.\vision_prototype\.venv\Scripts\python.exe -m pytest vision_geometry_lab\experiments\v28_local_spool_connection\test_render_eval_regressions.py -q
```

只运行本轮契约和明确无硬件的少量旧接口回归，不运行可能读取保留集的全库。合成测试通过不等于现场接法改善。
