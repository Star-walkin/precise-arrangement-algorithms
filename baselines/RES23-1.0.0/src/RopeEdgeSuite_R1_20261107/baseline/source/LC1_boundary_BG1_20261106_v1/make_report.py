from pathlib import Path
import json,math,numpy as np
from PIL import Image,ImageDraw,ImageFont
from run_full import ROOT,OLD,dump,sha
FONT='C:/Windows/Fonts/msyh.ttc'
def main():
    run=ROOT/'full_002';rows=[]
    for p in sorted((run/'frames').glob('*.json')):
        r=json.loads(p.read_text(encoding='utf8'))
        rows.append({**{k:r[k] for k in ['frame_index','count_valid','display_N','F3_called']},
                     'region':{k:r['region'][k] for k in ['valid','edge_y']},
                     'F3':None if not r['F3'] else {'continuous_pitch_distance':r['F3']['continuous_pitch_distance']}})
    assert len(rows)==601
    old=[]
    for i in range(601):
        r=json.loads((OLD/f'frame_{i:06d}.json').read_text(encoding='utf8'))['surface']
        old.append({'surface':{k:r[k] for k in ['edge_y','edge_current_frame_confirmed','count_valid','continuous_pitch_distance']}})
    metrics=dict(frames=len(rows),region_passed=sum(r['region']['valid'] for r in rows),
                 F3_calls=sum(r['F3_called'] for r in rows),count_valid=sum(r['count_valid'] for r in rows),
                 displayed=sum(r['display_N'] is not None for r in rows),
                 first_display_frame=next((r['frame_index'] for r in rows if r['display_N'] is not None),None),
                 edge_source='current_yellow_to_wound_surface_interface',absolute_accuracy=None)
    # Same frame pairs; no causality or precision inference from a smoother line.
    pairs=[]
    for i in range(122,242):
        a,b=rows[i-1:i+1];oa,ob=old[i-1:i+1]
        if a['count_valid'] and b['count_valid'] and oa['surface']['count_valid'] and ob['surface']['count_valid']:
            pairs.append([b['F3']['continuous_pitch_distance']-a['F3']['continuous_pitch_distance'],
                          ob['surface']['continuous_pitch_distance']-oa['surface']['continuous_pitch_distance']])
    pairs=np.asarray(pairs).reshape(-1,2)
    metrics['shared_adjacent_pairs_121_to_241']=len(pairs)
    if len(pairs):
        metrics['shared_pairs_new_increases']=int(np.sum(pairs[:,0]>0))
        metrics['shared_pairs_old_increases']=int(np.sum(pairs[:,1]>0))
        metrics['shared_pairs_new_p95_abs_change']=float(np.quantile(abs(pairs[:,0]),.95))
        metrics['shared_pairs_old_p95_abs_change']=float(np.quantile(abs(pairs[:,1]),.95))
    metrics['new_numeric_increases_not_smoothed']=sum(a['display_N'] is not None and b['display_N'] is not None and b['display_N']>a['display_N'] for a,b in zip(rows[:-1],rows[1:]))
    grid=json.loads((ROOT/'grid_scale_check/summary.json').read_text(encoding='utf8'))
    metrics['grid_scale_check_frames']=[r['frame'] for r in grid['records']]
    metrics['grid_scale_max_abs_difference']=max(abs(r['difference']) for r in grid['records'] if r['difference'] is not None)
    dump(ROOT/'delivery/metrics.json',metrics)
    im=Image.new('RGB',(1700,1040),'#101923');d=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,24);small=ImageFont.truetype(FONT,19)
    d.text((45,25),'BG1：先核对当前边界，再统计当前可见绳距',font=ImageFont.truetype(FONT,30),fill='white')
    def panel(y,h,vmin,vmax,label,newv,oldv):
        x0,x1=100,1635;y0,y1=y+45,y+h-35
        def pos(i,v):return (x0+i/600*(x1-x0),y1-(v-vmin)/(vmax-vmin)*(y1-y0))
        d.text((45,y),label,font=font,fill='white')
        for i,r in enumerate(rows):
            if not r['region']['valid']:
                x,_=pos(i,0);d.rectangle((x,y0,x+3,y1),fill='#38262e')
        for v in np.linspace(vmin,vmax,5):
            _,yy=pos(0,v);d.line((x0,yy,x1,yy),fill='#324150');d.text((20,yy-12),f'{v:.1f}',font=small,fill='#a0b5c5')
        for v,col in [(oldv,'#889aa9'),(newv,'#50e8aa')]:
            for i in range(1,601):
                if v[i-1] is not None and v[i] is not None and vmin<=v[i-1]<=vmax and vmin<=v[i]<=vmax:
                    d.line([pos(i-1,v[i-1]),pos(i,v[i])],fill=col,width=2)
        for sec in range(0,61,10):
            xx,_=pos(sec*10,0);d.text((xx-14,y1+6),str(sec),font=small,fill='#a0b5c5')
    newedge=[r['region']['edge_y'] if r['region']['valid'] else None for r in rows]
    oldedge=[r['surface']['edge_y'] if r['surface']['edge_current_frame_confirmed'] else None for r in old]
    panel(85,410,2290,2370,'当前上侧接触边界 y（原始像素；不是黄色外轮缘）',newedge,oldedge)
    newn=[r['F3']['continuous_pitch_distance'] if r['count_valid'] else None for r in rows]
    oldn=[r['surface']['continuous_pitch_distance'] if r['surface']['count_valid'] else None for r in old]
    panel(530,410,0,10,'当前视觉绳距（仅绘制通过各自校验的结果；未施加单调拟合）',newn,oldn)
    d.text((80,963),'绿色：新 BG1 + 01/F3    灰色：旧窄条检测    暗红背景：BG1 边界门控未通过',font=small,fill='#d8e6ee')
    d.text((80,995),'横轴：视频秒数。曲线差异不等于准确率；尚无独立逐帧边界/根数真值。',font=small,fill='#e4c486')
    im.save(ROOT/'delivery/BG1_boundary_and_count_comparison.png')
    receipt=json.loads((run/'receipt.json').read_text(encoding='utf8'))
    text=f'''BG1 双边界校验与卷面门控：本轮结果

目标：使用当前图像先确认黄色轮缘、卷面顶部接触界面和可用卷面，再调用临时文件/01提供的绳边方法。边界不通过时停止 F3 和根数输出。

验证范围
原左目视频 dual_20260924_111500/left_DA9627065.avi，4096×3000，601帧，10 FPS，60.1秒。只使用这一段既有允许视频；未读取保留视频。固定 d=2238.7 原图像素，数值仍为用户临时约束，没有重新标定 d。结果不外推到右目、其他照明或其他层。

检测流程
1. 在固定相机场景搜索范围内，以黄色色差、连通结构提取轮缘颜色候选。强反光白色部分需要当前暗到亮边缘支持；二次弧线仅提出搜索位置，不能单独填充轮缘。
2. 用局部纹理、结构张量和连通区域提取卷面候选。沿黄色到钢丝绳的过渡，在原图精修顶部接触边界；四次曲线剔除离群点，计数列使用邻近真实边界观测修正。
3. 黄色与卷面掩码互斥。仅在当前纹理仍支持的内部小孔补全；遮挡处、长跨度缺乏边界支持处和暗侧不强行外推。左右轮廓是保守可用ROI边界，不代表不可见的完整物理轮廓；底部为图像截断。
4. 外轮缘证据、顶部支持覆盖、d两侧支持、局部残差、计数列附近完整性共同通过后，才调用01中的surface_observation.py。判据包括顶部覆盖≥0.65、d附近覆盖≥0.70、局部MAD≤0.20×预设绳距、计数邻域有效覆盖≥0.95。它们是工程拒识规则，并非经过标定的正确率。
5. 01原始全卷面D4观测链和5个依赖文件均保持原字节；依赖哈希与提供的protocol.json核对。新视频没有对应的历史C3/F2缓存，因此本轮运行的是其全卷面观测核心，而非历史400图的完整surface_fusion.py缓存融合流程。没有把其他图的旧边界当成当前观测。
6. F3在2048×1500网格处理完整可用卷面，pitch、d、S1交点同步乘0.5，显示坐标换回4096×3000。卷面边界本身仍在原图精修。原网格试运行保留在full_001，因耗时中断，不算完整视频。最终全视频为full_002。
7. S1使用既有逐帧结果，逐帧解码像素SHA-256完全一致才复用。活动绳已观测线段被排除；没有把S1外推线当成观测绳边。根数来自当前实测沟槽的相位差，未知大间隙、缺少端点支撑或遮挡时不给数值；有效视觉距离在[0,5]绳距内才显示。

完整视频统计
共{metrics['frames']}帧；边界门控通过{metrics['region_passed']}帧，拦截{601-metrics['region_passed']}帧；实际调用F3 {metrics['F3_calls']}帧；当前计数通过{metrics['count_valid']}帧；五圈内显示数值{metrics['displayed']}帧。首个数值显示帧为{metrics['first_display_frame']}。
处理用时{receipt['elapsed_s']:.1f}秒，平均吞吐{601/receipt['elapsed_s']:.3f}帧/秒（含并行工作进程；不等于单帧实时延迟）。目前为离线验证版，尚不满足10 FPS在线检测。

如何读视频
橙色是黄色轮缘及外缘证据；紫色是计数用卷面顶部界面观测；蓝色是保守可用卷面；绿色是本帧实际沟槽观测；黄线为x=d；白虚线为当前S1解析直线，包含外推部分。界面未通过时显示灰色候选轮廓供诊断，F3不运行，不用上帧结果填空。放大窗口和原图都对应当前源帧。
数值是到当前上侧卷面边界的连续绳距，不是已标定的整数根数，也不能据此宣称已经触边。换向后到同一个上侧边界的距离可以增大，本轮没有强制全视频单调下降。

限制与下一步
门控通过只代表当前图像满足这些规则；没有独立人工逐帧边界/根数真值，不能报告准确率。record内edge_uncertainty_px是局部残差的工程量，不能当作校准置信区间或真实误差上界。半分辨率检测也可能漏细节。F3显示坐标采用比例缩放，采样网格为2个原图像素；尚未进行亚像素几何标定，不能把d的小数位解释为0.1像素的实际定位精度。
正确边界是计数的必要条件，S1外推偏差、漏检沟槽、局部绳距/相位变化和触边零点仍可能使数值跳动。因此本轮保留原始跳动用于诊断，没有用单调拟合掩盖问题。后续可在人工确认边界与触边帧后，再标定零点、评价误差并恢复时序估计。
对比曲线中，边界y仍有约一个绳距量级的周期变化。目前不能区分其中多少来自真实卷面起伏、轮体运动或颜色定位偏差；需要在原图上标注边界点进行独立核对。视觉上的平滑或门控通过均不能替代这项检查。

补充检查：7项门控/平移测试通过；在第240帧的合成亮度与模糊扰动中，通过门控的边界位移均不超过约1.2像素，重度模糊被拒识。第127、200、240帧原分辨率与半分辨率计数均有效，两种网格最大差约{metrics['grid_scale_max_abs_difference']:.3f}绳距。这些是有限样本的鲁棒性/网格敏感性检查，不是真值准确率。

主要文件
delivery/BG1_boundary_gated_full.mp4：601帧全视频。
delivery/BG1_boundary_and_count_comparison.png：边界与视觉绳距对比。
full_002/frames：逐帧来源、边界观测、拒绝原因、F3观测与计数。
full_002/masks：每帧黄色轮缘、可用卷面0/255掩码。
audit.json：逐帧来源绑定、门控、掩码与观测包含关系检查；结构审计不等同于精度验证。
test_region_gate.py：黑图、遮挡、缺少黄色、纯黄色、垂直平移等7项测试。
run_full_fixed.py：后续重复试验入口，需要用--name指定新的输出目录。原run_full.py保留为本次实际执行的冻结源码。本次601帧全部计算完成后，旧入口在写汇总时出现局部变量作用域错误；finalize_full.py已核对并恢复汇总，没有改动任何检测结果和掩码。修复入口通过了单帧完整流程复验。
'''
    (ROOT/'delivery/BG1_结果与限制.txt').write_text(text,encoding='utf-8-sig')
    print(json.dumps(metrics,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
