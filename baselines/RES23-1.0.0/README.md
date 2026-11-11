# RES23-1.0.0 有效基准

**RopeEdgeSuite R23：钢丝绳排绳边界、计数与换层时序融合算法。** 冻结日期 2026-11-11；当前有效研发基准，推荐时刻模块为 CPL1 + MFT2。

完整介绍、图像与快速命令见[仓库首页](../../README.md)。

| 文件/目录 | 作用 |
|---|---|
| [docs/ALGORITHM.md](docs/ALGORITHM.md) | 完整算法原理与公式 |
| [docs/USAGE.md](docs/USAGE.md) | 安装、运行、新视频接入与输出格式 |
| [docs/VALIDATION.md](docs/VALIDATION.md) | 逐视频验证、证据与限制 |
| [docs/SOURCE_MAP.md](docs/SOURCE_MAP.md) | 实际生产模块映射 |
| [src/](src/) | 字节不变的算法与支撑源码；保持原相对目录 |
| [source_inventory.json](source_inventory.json) | 源文件来源、大小、SHA-256 |
| [package_manifest.json](package_manifest.json) | 本发布包逐文件 SHA-256 |
| [tools/baseline.py](tools/baseline.py) | 哈希验证、45 项测试、样例重放、前端检查 |
| [examples/dual_20260924_111500__left/](examples/dual_20260924_111500__left/) | 可运行的真实测量/纹理样例和冻结参考结果 |
| [evidence/](evidence/) | 原验证记录、逐视频结果与原始输出图 |
| [environment-tested.json](environment-tested.json) | 本次实际验证环境版本 |

完整源码含 B750/P1/H6、S1/S2 支撑、SG/BG/CG、F3、d、计数、时序和出图脚本。`src/` 中保留了支撑版本及实验工具，不能把所有脚本按名称顺序一遍运行；请按生产模块映射与应用方法选择入口。

本发布增加包装入口与文档，没有修改冻结算法数值逻辑。原视频和大型缓存不随 Git 分发；完整原视频流程仍需要相应输入数据和项目路径配置。样例重放及单元/行为测试可以直接在本包中执行。
