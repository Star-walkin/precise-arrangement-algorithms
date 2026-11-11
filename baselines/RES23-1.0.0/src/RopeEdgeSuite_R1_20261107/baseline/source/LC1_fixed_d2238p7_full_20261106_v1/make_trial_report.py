"""Summarize completed, audited analysis; does not perform inference."""
from pathlib import Path
import json,csv,collections
ROOT=Path(__file__).resolve().parent
def main():
    r=json.loads((ROOT/'analysis_anchored/receipt.json').read_text(encoding='utf-8'))
    v=json.loads((ROOT/'video_analysis_anchored/receipt.json').read_text(encoding='utf-8'))
    assert r['status']==v['status']=='completed' and r['processed_frames']==r['source_frames']==v['decoded_frames']==601
    rows=list(csv.DictReader((ROOT/'analysis_anchored/measurements.csv').open(encoding='utf-8-sig',newline='')))
    first=r['first_inside_confirmation_frame']
    distances=[float(x['display_distance_N']) for x in rows if x['emit_distance']=='True']
    reasons=collections.Counter(x['F3_reasons'] for x in rows if x['current_region']=='unknown')
    text=[
        '固定 d=2238.7 像素、五绳距触发：完整左相机视频试验',
        '',
        '输入：'+r['source_video'],
        f"完整处理：{r['processed_frames']}/{r['source_frames']} 帧，逐帧不跳帧；原帧率 {r['source_fps']:g} FPS，回放 {r['processed_frames']/r['source_fps']:.1f} 秒。",
        '视频：'+v['video'],
        '测量表：'+str(ROOT/'analysis_anchored/measurements.csv'),
        '',
        '本次约束与显示规则：',
        '1. 原图为 4096×3000；不镜像。d 固定为原图 x=2238.7，并非自动标定结果。',
        '2. 每帧使用当前 S1 直线 Ax+By+C=0，与 x=d 求交；自由绳 E1 的空间起点固定，避免沿材料运动逐渐走向卷面。',
        '3. 原 F3 在该列识别当前边界、局部绳距和实测绳边。只有本帧 S1 与 F3 均有效且 N≤5 时显示距离；N>5 不启用，失效时显示未知，不保留旧距离。',
        '4. S1 顺序处理；F3 为独立当前帧的三个进程计算，按源帧顺序输出，无未来帧平滑。',
        '',
        '实际结果：',
        f"S1 有效帧：{r['S1_valid_frames']}/{r['processed_frames']}；实际 S1 core 调用 {r['actual_S1_core_calls']} 次。",
        f"F3 可计数帧：{r['F3_valid_count_frames']}；五绳距以内显示数值 {r['prediction_frames']} 帧。",
        '状态分布：'+json.dumps(r['state_counts'],ensure_ascii=False),
        f"首次有效五绳距触发：源帧索引 {first}（从0计），视频回放 {first/r['source_fps']:.1f} 秒，采集相对时间 {r['first_inside_confirmation_time_s']:.4f} 秒。" if first is not None else '本片段没有有效的五绳距触发。',
        f"显示距离范围：{min(distances):.3f}–{max(distances):.3f} 根绳距。" if distances else '无显示距离。',
        f"实际 P1 重捕模型前向 {r['actual_P1_reacquisition_forwards']} 次；只复用对应同帧、像素哈希绑定的 P1 seed {r['reused_current_frame_P1_seed_count']} 次，旧解析式未用作当前结果。",
        f"分析总耗时：{r['wall_seconds']:.1f} 秒；按原10 FPS回放，不能据此称算法已实时。",
        '',
        '本次结果的解释：',
        'N 是连续绳距坐标，尚未进行整数根数的相位标定，也未逐帧人工标注验证误差。首次触发表示首次有效观测，不能视为真实跨越五绳距的精确时刻。',
        '固定 d 能否覆盖整段的不同层和换向状态，需要对照视频另行确认；交点越过边界或端点支持不足时，本版保留未知。',
        '旧 analysis 为取样沿绳移动问题中止的诊断运行；最终结果只采用 analysis_anchored 和 video_analysis_anchored。',
        '',
        '未知帧 F3 原因组合（原始字段）：',
    ]
    text.extend(f'{n} 帧：{reason or "无F3拒绝原因；查S1/gate"}' for reason,n in reasons.most_common())
    text.extend(['','完整输入 SHA256：'+r['source_sha256'],'输出视频 SHA256：'+v['video_sha256']])
    (ROOT/'试运行结果说明.txt').write_text('\n'.join(text)+'\n',encoding='utf-8-sig')
    print('\n'.join(text))
if __name__=='__main__':main()
