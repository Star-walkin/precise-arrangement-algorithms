# 运行与应用方法

## 1. 选择运行层级

| 任务 | 入口 | 需要的数据 | 本发布验证范围 |
|---|---|---|---|
| 校验发布内容 | `tools/baseline.py verify` | 本仓库 | 所有发布文件与冻结源码哈希 |
| 执行行为回归测试 | `tools/baseline.py test` | 本仓库 | 45 项测试 |
| 重放真实时序样例 | `tools/baseline.py replay-example` | 内附测量和纹理特征 | 111500 全 601 帧及事件与 R23 一致 |
| 用自己的测量调用 CPL1/MFT2 | `tools/baseline.py predict` | 同步测量、近边事件、纹理特征 | 接口与示例；新场景需单独验证 |
| 原视频前端检查 | `tools/baseline.py frontend-smoke` | 原始视频 | B750/P1/H6/S1/SG5/BG11/CG2 加载及前三帧执行 |
| 原视频完整检测与出图 | `src/` 中各生产阶段 | 原视频、阶段配置和逐帧缓存 | 原工程 11 视频实验；非跨机器一键部署入口 |

完整算法源码与所需的小型 B750 权重均已收录。为了保持基准逻辑不变，本次没有重写原工程的全部数据调度层：部分批处理脚本仍读取原工程的路径、缓存和冻结清单。不能把最后一行误解为“克隆后对任意视频一条命令完成全部检测”。

## 2. 环境安装

Windows 下部分冻结依赖目录较深，克隆时启用 Git 长路径，并尽量选择较短的工作目录：

```powershell
git -c core.longpaths=true clone https://github.com/Star-walkin/precise-arrangement-algorithms.git C:\work\rope-algorithms
```

若已经克隆，在仓库内执行 `git config core.longpaths true`。不要为了缩短路径直接改冻结依赖目录名，否则相对导入和来源哈希会失效。

建议 Python 3.12，在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r baselines/RES23-1.0.0/requirements.txt
.\.venv\Scripts\python.exe baselines/RES23-1.0.0/tools/baseline.py verify
.\.venv\Scripts\python.exe baselines/RES23-1.0.0/tools/baseline.py test
```

处理原视频还需安装热图前端依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r baselines/RES23-1.0.0/requirements-frontend.txt
```

依赖文件给出兼容安装范围；[environment-tested.json](../environment-tested.json) 记录本机实际通过测试的精确版本。本次没有在所有依赖版本组合上验证。完整图表脚本还使用 Matplotlib，中文出图默认引用 Windows `C:/Windows/Fonts/msyh.ttc`；换平台应配置字体并重新检查布局。

B750 权重位于：

```text
src/RopeEdgeSuite_R1_20261107/baseline/source/
  LC1_locked_column_20261105_v1/supplied_assets/step_0750.pt
```

S1/S2、P1/H6 及用户提供 `01` 绳边方法的实际依赖源码也保留在相应 `supplied_assets`、`p1_runtime`、`vendor` 目录中；这些是项目支撑代码，不是整个 Python 虚拟环境。文件来源和 SHA-256 见 [source_inventory.json](../source_inventory.json)。

## 3. 可直接运行的真实样例

```powershell
.\.venv\Scripts\python.exe baselines/RES23-1.0.0/tools/baseline.py replay-example --out replay_output/111500.json
```

样例输入是 R22 的当前图像测量、近边事件和图像纹理特征；候选算法不读取人工时间。运行后逐项比较全部输出和单独存放的冻结结果，成功时打印：

```text
PASS: all per-frame outputs and events match the frozen R23 result
candidate_s: 30.4
confirmation_s: 31.1
```

数值关系为 `t0=24.9`、`T=2.746937...`、`t1=27.646937...`、`t2=30.393874...`。这个样例验证新时序模块及其打包一致性，**没有重跑原图 F3，也不提供新增独立人工真值**。

## 4. 原视频前端检查

在本项目已有视频的机器上，示例命令如下；其他机器替换 `--video` 路径：

```powershell
.\.venv\Scripts\python.exe baselines/RES23-1.0.0/tools/baseline.py frontend-smoke --video "E:\study\program\济宁\双相机同步\dual_20260924_111500\left_DA9627065.avi" --camera left --frames 3 --out replay_output/frontend.json
```

该入口直接解码原视频，加载包内冻结前端；右相机用 `--camera right`。它输出每帧 `track`、`proposal`、`boundary`、`camera` 以及实际加载的包内模块路径。它只用于检验依赖闭包和前端，不输出 F3 计数。前三帧存在初始化过程，首帧无可靠跟踪/边界不等于程序执行失败。

本冻结前端针对 4096×3000 图像；命令会拒绝其他分辨率，避免暗中缩放后继续使用不适用的像素参数。

## 5. 对自己的测量调用时序融合

```powershell
.\.venv\Scripts\python.exe baselines/RES23-1.0.0/tools/baseline.py predict --measurements measurements.json --near-events near_events.json --features features.npz --fps 10 --out replay_output/events.json
```

输入要求：

| 文件 | 内容 |
|---|---|
| `measurements.json` 或 `.json.gz` | 按时间排列的逐帧测量列表，使用与附带样例相同的字段 |
| `near_events.json` | 近边事件列表；至少含 `start_s`、`entry_confirmed_s`，事件应由当前测量门控产生 |
| `features.npz` | `features` 数组，形状为 `[帧数, 区域数, 特征维数]`，和测量逐帧对应 |
| `--fps` | 真实采样帧率；时间戳与纹理数组必须同步，不能丢帧后仍当作等间隔完整序列 |

关键测量字段：`frame`、`time_s`、`epoch`、`available`、`N`、`N_interval`、`current_coil_side_axis`、`boundary_occluded`、`state22.evidence_mode`、`state22.motion`。实际完整字段以附带样例和 `fusion23.py` 为准；`N` 不可用时保留 `null` 和原因，不用 0 替代。

`predict` 不会自动生成缺失的测量/近边事件。若只提供视频，应先执行下面的视觉流程。输入事件可能来自离线计算；新增模块按事件确认时间门控输出，但使用方仍须保证上游数据本身没有未来信息泄漏。

## 6. 完整视觉流程与数据依赖

本地完整实验目录为：

```text
E:\study\program\济宁\precise_arrangement\05_派生与实验结果\05_派生与实验结果
```

其中 R19–R23 为相邻目录。`src/` 保留了同样的相对结构。完整复现需要原视频和各级数据，依赖顺序如下：

| 阶段 | 当前主要实现 | 产物与下一步前提 |
|---|---|---|
| 视频清单和朝向 | R19 `plan.json`、`run_fresh.py` | 源 SHA、帧率、帧数、相机标签；每帧图像哈希 |
| 活动绳/卷面前端 | R19 `engine.py`，SG5/BG11/CG2 | `track`、`boundary`、当前支持/遮挡和相机状态 |
| ROI/F3 与局部修复 | R20 `roi21.py`、`roi22.py`、`region_fragments24.py`、`extract_parallel22.py` | `f3_v2` 当前字段与点、修复记录及前端覆盖引用 |
| 黄点分组与曲线 | R20 `groups20.py`、`group_roi21.py` | `groups_v2`，统一曲线编号、支持区间 |
| 卷面侧轴线和切线列 | R20 `geometry22.py`、`coil_edge_axis_v4.py`、`tangent_column.py`、`measure_geometry_v5.py` | `geometry_v5` |
| 运动分段和固定 d | R20 `phase21.py`、`calibration20.py`、`calibrate_guard.py` | `motion_phases_v2.json`、每视频 `calibration/*.json`；至少 5 根绳审核 |
| 曲线计数 | R20 `count20.py`、`count_all.py` | `counts_v1`，`N`、敏感性范围、完整相位标记 |
| 动态列与最终测量 | R21 `run_measure21.py`、`refine_dynamic21.py`、`final_input21.py`、`finish_final21.py` | 当前动态列与固定列分开；原图叠加和紧凑逐帧测量 |
| 运动状态与纹理 | R22 `prepare22.py`、`extract_rotation22.py`、`run22.py`、`render22.py` | MTD1、纹理特征、R22 原字段；MFT1 仅作历史对照 |
| 两周期融合与出图 | R23 `run23.py`、`render23.py`、`audit23.py` | CPL1/MFT2、完整 R23 视频、逐帧一致性与因果审核 |

此表是数据依赖顺序，不是可以无条件顺序执行的脚本清单。历史修复批处理包含特定实验清单，缓存还绑定源哈希；不能复制旧视频的 `delivery_manifest`、帧记录或 d 到新视频，也不能直接运行旧修复清单来处理新场景。

对于**当前本机已具备全部 R22 数据的 11 个视频**，在实验目录的工作副本中运行以下最后阶段即可重算 R23：

```powershell
# 切换到含 R19–R23 相邻目录的实验工作副本；保留原冻结输出。
python RopeEdgeSuite_R23_20261111/run23.py
python RopeEdgeSuite_R23_20261111/render23.py
python RopeEdgeSuite_R23_20261111/audit23.py
python RopeEdgeSuite_R23_20261111/review23.py
```

之后实际查看关键帧与完整视频，再更新实验记录和报告。原 `report23.py` 包含本轮已经完成的查看结论，不能把它的固定文字当成对新算法的自动人工审核。修改路径时也应检查 JSON/NPZ 引用和哈希回执；单改 `SOURCE` 字符串不足以迁移全部缓存。

## 7. 新视频的推荐接入步骤

1. 新建独立实验目录和视频清单，记录相机、分辨率、帧率、完整解码帧数和源 SHA-256。
2. 核对镜像方向及计数坐标；先检查原视频前端的活动绳身份、可见卷面边界和遮挡排除。
3. 使用同一当前 ROI 生成 F3 黄点和分组曲线，再计算卷面侧轴线和 TC1；保留当前证据及来源哈希。
4. 分运动阶段筛选至少 5 根绳的参考帧，标定各段 d；直接弯曲证据不足时明确输出条件几何/动态列等级。
5. 计算原始 N、范围、完整 F3 相位、未知原因，再接入 MTD1。先验证计数正确性，不能用状态平滑弥补边界错误。
6. 提取同步纹理并接入 CPL1/MFT2。保留 t0、首次爬升、t1 参考、t2 参考、图像候选及核验完成时间。
7. 采用与基准相同的完整解码、逐帧来源、遮挡、缺边、错误归类及因果测试，查看跨场景画面；用独立人工标记评价后，再决定是否晋级为下一有效基准。

## 8. 输出的正确读取方式

- 先检查 `available` / 当前证据 / 遮挡标记，再读取数值。
- `N_interval` 与 `t2_interval` 是模型敏感性范围，不是校准后的 95% 置信区间。
- `fixed_d=null` 且 `measurement_mode=current_dynamic_column` 时不能把当前列说成已经锁定的 d。
- `history_hold` 为显示保持；不得用其触发接触或电机动作。
- `timeline.t2` 是预测；`candidate_s` 是图像支持候选，只有到 `confirmation_s` 才完成相应核验。
- 本版本只提供检测结果，没有电机/PLC 写入接口。
