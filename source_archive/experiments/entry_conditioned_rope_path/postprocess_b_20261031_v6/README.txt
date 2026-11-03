热图接续 H6 / IR-H6.0.0：F3来源中心轨迹与非跨轨接续

四栏图库：D:\precise_arrangement\experiments\entry_conditioned_rope_path\postprocess_b_20261031_v6\all_001\四栏检测结果
实际292/292、301个原自动入口条件，退出0，522.6秒。输出分类{'straight_free_to_F3_center_track': 103, 'free_fragment_no_resolved_connection': 189}，不是识别正确率。
仍为原开放400图；108个旧入口未解决不改称无绳。未训练、未新增模型前向、未访问标签/保留集或连接设备。

改动文件：surface_tracks.py复用F3接受向量/来源谱系，替代旧raster连通块配对；topology.py检查完整几何；connect_paths.py拒跨轨桥并锁单轨尾；run_h6.py保留完整原直轴、任务前缀、竞争候选及四栏图。
F3完整C4范围与独立SAM2分开保存。灰实线为原F3支持，灰虚线为源内推断。源缺失不能用固定间距或邻近行强补。中心线固定双边来源，几何不随B热点漂移。
自由段图像投影相交与贴面跳轨分开。交点自身必须落在连续原图联合支持内，不能只因附近亮纹放行。弱尾跨轨被单列为unresolved_surface_projection，完整H5观测仍在full_free_observation。
全量49图任务前缀发生裁剪，移出的后段中仍包含432.0原图联合支持像素长度；这是保留的覆盖代价，不能将全部裁尾称为无证据或正确拒绝。
计入最后接续路径后，按固定距离口径保留的原直段图像支持长度相对H5净变化-126.0原px，3图下降。这也是覆盖诊断，不是身份准确率。
桥及贴面段不可碰穿非目标灰轨；目标只允许末端一次捕获或之后精确不离线，按两曲线联合节点检查。贴面尾保持同一来源边界对，不换线。
第三栏仅画拓扑许可候选；图像弱的合法候选标“弱证据未采用”，不能获选。橙虚线为弱证据，绿色也不等于同根真值。

固定16开发（旧14+用户0345/0255）与冻结后旧8开放回归；24图子集与全量JPG逐字节相同。图1弱直尾拦截后有单轨接续候选，图2的非法桥被去掉但完整接续未解决。不能把截短、无交叉或单轨约束符合称为整根识别成功。
首轮与共线后缀合成失败保留，修复结果另存。原图/旧F3/H5/model均不改。参数/输入/源码哈希、命令、失败/跳过见inputs.json、freeze_final.json、每个run的run_start/source_snapshot/summary/progress。
浮点B仍在每entry的response_file，以原hash绑定；坐标原图=2*crop_half_xy+[0,1050]。

实际全量命令：vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v6/run_h6.py --run all_001 --subset all --workers 4
既有CPU环境，不安装/修改依赖。复跑用新run名，勿覆盖历史。
