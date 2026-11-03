"""Build concise tables from saved results; never changes predictions."""
from pathlib import Path
import argparse,json,hashlib

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pct(x):return '—' if x is None else f'{100*x:.2f}'
def num(x):return '—' if x is None else f'{x:.1f}'
def main(run):
    run=Path(run).resolve();s=read(run/'evaluation/summary.json')['summary']
    v=read(run/'verification.json');inf=read(run/'batch/inference_receipt.json')
    post=read(run/'batch/postprocess_receipt.json');render=read(run/'batch/render_receipt.json')
    formal=read(run/'formal/receipt.json');lines=[];dist=[];cost=[]
    for mode in ['CONDITIONAL','AUTO']:
        for model in ['A','B']:
            for method in ['P0','P1']:
                r=s[model][method][mode];name=('条件' if mode=='CONDITIONAL' else '自动')+f' {model}/{method}'
                cells=[]
                for stage in ['all','free','transition','attached','edge_near']:
                    cells.append(pct(r['geometry']['supported_only'][stage]['coverage_5px'])+'/'+pct(r['geometry']['all_geometry'][stage]['coverage_5px']))
                lines.append('|'+name+'|'+str(r['selected_images'])+'/8|'+ '|'.join(cells)+'|')
                d=[num(r['geometry']['all_geometry'][stage]['reference_to_prediction_mean_px_available_only']) for stage in ['free','transition','attached','edge_near']]
                dist.append('|'+name+'|'+' / '.join(d)+'|'+num(r['geometry']['all_geometry']['all']['longest_uncovered_run_px'])+'|')
                lens=r['selected_claimed_lengths_px'];support=lens['heatmap_supported']+lens['image_supported']
                cost.append('|'+name+'|'+str(r['rejected_images'])+' / '+str(r['entry_unavailable_images'])+'|'+str(r['candidate_count_total'])+'|'+num(support)+' / '+num(lens['bridge'])+' / '+num(lens['ambiguous'])+'|'+pct(r['selected_bridge_fraction_of_noninput_length'])+'|'+f"{r['timing']['mean_seconds']:.3f}"+'|')
    report=f'''# 冻结A/B热图：线段提取与短缺口连接实验

运行：postprocess_20261030_001。**实际完成，结论为未达到完整目标路径改善；保留P0作原基线，P1作为诊断候选，不替换。**

## 先回答本轮问题

1. **明显斜段能提取，但不能说主要靠短缺口连接即可稳定恢复。** A条件模式自由侧仅支持覆盖47.35%→49.36%，过渡91.08%→100%；桥只增加0.34个百分点的总体覆盖。局部改善主要来自线段提取与路径选择的组合，本轮未做二者独立消融，不能进一步作因果分摊。
2. **过渡之后未稳定保持正确目标。** A条件模式后段全几何覆盖31.77%→17.92%；四组P1近边缘参考覆盖均为0。B条件模式小幅改善，AUTO明显退化。没有独立完整末端参考，完整弧段成功率未定义。
3. **桥接没有解决主要缺失。** P1支持/含桥几何覆盖相差0.03～0.34个百分点，已选路径桥长占0.66%～3.35%；已选结果的弱热图`image_supported`分支为0。`heatmap_supported`仍只是局部线状证据，不等于身份确认。
4. **确有“连着走却走错”的反例。** 左450在约(1600,2093)偏离可信中心后仍走371px，追随上侧近旁亮纹，末点距可信部分中心39.33px；其中4.47px短桥并没有纠正错误。不能进一步断言它具体属于哪一历史圈或哪条股纹。
5. **困难包括正确线段缺失、片段接入和竞争接续，超过单纯短断点。** 下一步优先修清晰入口的投影接入、检查交汇处片段拆分/连续方向及正确候选保留；暂不需要先训练关系模块。若后续在正确候选连通条件具备后仍选错，可聚焦左450约(1600,2093)及桥(1624.5,2092.5)→(1628.5,2090.5)的局部选择。本轮没有修改已冻结门槛来补救失败。

## 图片交付

- [292张四联图：原图＋A原始热图＋B原始热图＋P1中心路径覆盖](四联图_原图_A_B_中心线示意/)。299个已存入口，7张双入口图各保留两行。第四面板A青/B橙；支持实线、桥接虚线、歧义紫虚线，竞争候选细虚线，入口白色；拒识明确标出。线宽仅用于显示，**不是真实绳体宽mask**。
- [108张入口缺失诊断](无入口诊断_108/)保留原400清单剩余样例。A/B面板明确“未推理”，不是全零热图；缺入口不能解释成无绳。400=292+108，无新入口救场或事后挑图。
- [正式8图固定全景及四阶段局部对照](visualizations/)：32张PNG，包含四种组合；[重点pair0075双入口诊断](example_pair0075/REPORT.md)：22张PNG；[三个原图失败证据](independent_qa/)。

## 冻结、输入和运行口径

A/B严格为`run_20261030_002`共同750步：A SHA256 `{inf['models']['A']['sha256']}`；B `{inf['models']['B']['sha256']}`。正式8图的32份原始float响应直接复用，P0按原`connect.py`步长12、beam18、评分与停止配置重放，32份候选/选择/配置与原存档完全相同。没有用后来其他H系列后处理替代P0。

旧292图图库未留float，故仅按已核验同checkpoint、同入口、同灰度/归一化补598次前向；没有从JPEG反推响应。完整原图4096×3000；计算域[0,1050,4096,3000]，模型半尺度975×2048，底部中性补齐至992，去补边后双线性映回原图。域外为NaN，原图真实0保留；未使用I>0或排除自由绳的卷面mask。原响应、模型、旧结果未覆盖。

P1只用4个训练来源各1图校准，A/B共享绝对高/低阈值0.45/0.20；正式评价前冻结。中心峰法向剖面显示两侧低响应可辅助中心定位，未取反追踪暗带。派生副本Gaussian σ=2原图px、截断3σ；法向NMS3px、对比采样6px，最短片段24px、重采样4px。原图光学宽度参考16/24/36px只是结构检验，不是物理绳径或真实边界。

桥单段≤40px、总≤120px、最多6处；切向差≤48°、横移≤10px、带宽/原图支持检验。每端点最多6后继，片段≤1500，边检查≤12000、搜索≤5000扩展，最多3条不同竞争路径；提取/图构造/搜索各15/8/5秒检查上限。全局候选先限80000脊点：598个批量热图均触发此上限，可能丢失弱正确片段，明确作为限制保留。配置、校准、冻结、逐边接受/拒绝理由和派生滤波/脊线缓存均存盘，没有用人工后段裁搜索走廊或给末端提示。

CPU原环境运行，无安装、训练或微调。正式64例{formal['seconds']:.1f}s；批量前向{inf['seconds']:.1f}s；4进程后处理{post['seconds']:.1f}s；四联渲染{render['seconds']:.1f}s。P0/P1状态空间与计算量不同，不能以beam相同称预算相同。参数、实际命令、退出码见各receipt和[COMMANDS.md](COMMANDS.md)。

## 固定8图部分参考评价

3个来源组，均为既有开放开发评价用途，不是新盲测；训练来源不回流。CONDITIONAL只给原已知短入口，AUTO保留原3/8入口失败，不补提示。先保存/校验64份预测再加载参考。排除实际输入入口及5px胶囊；容差5原图px，按参考弧长加权，拒识保留8图分母。阶段是冻结的视觉空间代理，并非物理接触真值；“近边缘”也不是独立目标末端标签。

表中每格为 **仅支持段 / 包含桥与歧义的全部输出几何覆盖（%）**。不同入口模式排除的已知长度不同，主比较只在同模式同模型内进行。

|组合|暂选|总体|自由侧|过渡|后段|近边缘|
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

P1已选路径没有ambiguous长度；不代表身份无歧义，只表示局部证据状态未触发该类别。P0的“支持”为旧热图＋原图联合判据，P1为本轮线状判据，两者不是同一校准置信度。P0未显式桥接，其“全部−支持”包含旧歧义段。以下支持/桥/歧义长度排除入口，单位原图px；候选数含被拒绝的候选。AUTO平均时间包含三次入口失败的快速返回。

|组合|拒识/其中入口失败|候选总数|支持/桥/歧义长度px|桥占比%|平均秒/图|
|---|---:|---:|---:|---:|---:|
{chr(10).join(cost)}

下表距离为**存在所选输出时**参考→全部输出几何的均值，拒识距离未定义、不能以0补齐；覆盖表仍计拒识。不同暂选集合使可计距离分母不同，不能仅比较均距宣称更准。大距离常由早停后参考剩余段造成，不等于已检测段法向误差。每阶段分母、匹配内误差、首次≥60px连续漏段坐标和逐图统计见[evaluation/summary.json](evaluation/summary.json)及各逐图JSON。

|组合|自由/过渡/后段/近边缘平均距离px|最长连续漏参考长度px|
|---|---:|---:|
{chr(10).join(dist)}

## 具体失败及重点样例

左450：P1全局片段后段几何可用性71.16%，但入口选中路径仅20.22%；约(1600,2093)先偏离后继续，最终(1958.5,2124.5)。存在部分近参考线段，**不等于完整正确接续候选已经齐全**。右500：在距参考3.94px处早停，约1054px参考未覆盖；后段支持覆盖P0 73.31%→P1 23.11%，全局片段可用性80.08%。左550后段支持33.23%→12.46%，全局片段可用性63.20%。失败局部图是评分后的诊断，不用于选参数或改变评价区域。

`dual_20260923_101517__pair0075__right`来自历史训练来源，只作展示，未并入8图统计。manifest分别记录pair_id与源frame字段，保留原id，不自行换算。入口1 `recovered_01`在原图外观上疑似手持金属工具：P0仍给长路，P1 A/B均拒识；这不证明入口分类已经解决。入口2 `recovered_00`有明显自由绳纹理：A/P1已有近入口线段，却因投影距离4.353px略超4px局部门槛及前向条件被拒；B/P1新增支持540.70px＋桥23.63px，止于(3426.5,2138.5)，尚在自由侧。两入口始终分别处理，未事后择优，均未确认完整末端身份。

## 核验与保留边界

[verification.json](verification.json)：598份float逐字节哈希、1196份批量路径、模型哈希、坐标/NaN/补线界限通过；292张JPEG完整性及6张全像素解码通过。独立复核64份正式预测/32份原float哈希及32PNG解码；重点例另核22PNG。批量P1 A暂选191/299、B207/299，仅为输出状态数，**不是识别正确率**。

所有原始图、冻结网络/热图和既有协议保留；未访问9月25日等保留集、未连接设备、未改权限。没有证明贴面同根身份、完整到边缘成功或真实遮挡恢复。下一步应先处理已明确的入口接入及局部片段接续失败，再决定是否需要额外关系模型；本轮保留简单P1供诊断，不宣称它已稳定解决问题。
'''
    (run/'REPORT.md').write_text(report,encoding='utf-8')
    receipts=['batch/inference_receipt.json','formal/receipt.json','batch/postprocess_receipt.json','batch/render_receipt.json','evaluation/summary.json']
    cmd=['# 实际运行命令与环境','',r'工作目录：D:\precise_arrangement。图像/后处理使用 vision_prototype\.venv\Scripts\python.exe；仅冻结前向使用 vision_training\.venv\Scripts\python.exe。无训练命令。','',
      '新运行应另建目录；当前输出是冻结证据，不覆盖重跑。先完成数据准备/训练侧校准并冻结p1_config.json、p1_freeze.json及evaluation_protocol.json，再正式运行。','',
      '```powershell',
      'vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_batch.py prepare --run experiments/entry_conditioned_rope_path/postprocess_20261030_001']
    for receipt in receipts:
        d=read(run/receipt);exe='vision_training' if receipt.startswith('batch/inference') else 'vision_prototype'
        cmd.append(exe+'/.venv/Scripts/python.exe -B '+' '.join(d['command']))
    for script in ['postprocess_visualize.py','postprocess_missing_entries.py','example_postprocess.py','postprocess_verify.py','postprocess_report.py']:
        cmd.append('vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/'+script+' --run experiments/entry_conditioned_rope_path/postprocess_20261030_001')
    cmd+=['```','','准备、校准和原图审计的细节见data_audit.json、calibration_report.json与calibration/frozen/receipt.json。正式分组、参考口径见evaluation_protocol.json；路径和逐边拒绝理由在formal/及batch/conditions/；提取缓存p1_cache/只含派生数据。','',
    '源代码：p1_segments.py为新算法；postprocess_batch.py为冻结数据/前向；postprocess_run.py重放P0与运行P1；postprocess_evaluate.py独立评分；postprocess_visualize.py和postprocess_render.py绘图。旧connect.py、model.py、训练实现未修改。']
    (run/'COMMANDS.md').write_text('\n'.join(cmd),encoding='utf-8')
    print(json.dumps({'report':str(run/'REPORT.md'),'characters':len(report)},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(p.parse_args().run)
