"""Summarize operator review and prepare a small SAM 2 anchor-label task."""
from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'vision_prototype' / 'dataset_bootstrap' / 'batch_v1'
REVIEW_PATH = BATCH / 'operator_review.json'
REPORT_PATH = BATCH / 'report.json'
TASK_DIR = Path(__file__).resolve().parent / 'task_v1'
MANIFEST_PATH = TASK_DIR / 'manifest.json'
SUMMARY_PATH = ROOT / 'vision_prototype' / '第二阶段复核结论与SAM2标注说明.md'

# Six anchors deliberately cover: correct baseline, static hard failure,
# start, stop, restart and a later winding thickness/state.
ANCHORS = [
    'R_static_20260922_01__t000.000__f000000.jpg',
    'R_static_20260922_01__t007.093__f000216.jpg',
    'R_run_stop_run_20260922_01__t012.033__f000368.jpg',
    'R_run_stop_run_20260922_01__t030.081__f000920.jpg',
    'R_run_stop_run_20260922_01__t047.607__f001456.jpg',
    'R_run_stop_run_20260922_01__t073.503__f002248.jpg',
]


def main():
    review_payload = json.loads(REVIEW_PATH.read_text(encoding='utf-8'))
    report = json.loads(REPORT_PATH.read_text(encoding='utf-8'))
    reviews = review_payload['annotations']
    rows = {row['image']: row for clip in report['clips'] for row in clip['frames']}
    missing = [name for name in ANCHORS if name not in reviews or name not in rows]
    if missing:
        raise SystemExit(f'锚点缺少人工复核或算法记录：{missing}')

    comparable = []
    for name, item in reviews.items():
        manual = item.get('remaining_rope_count', '').strip().lower()
        algorithm = rows[name]['cold_start']['distance_rope']
        if manual not in ('', 'unknown') and algorithm is not None:
            comparable.append(float(algorithm) - float(manual))

    tasks = []
    for order, name in enumerate(ANCHORS, 1):
        human = reviews[name]
        row = rows[name]
        tasks.append({
            'order': order,
            'image': str((BATCH / 'frames' / name).resolve()),
            'image_name': name,
            'clip': human['clip'],
            'time_s': human['time_s'],
            'remaining_rope_count': human['remaining_rope_count'],
            'quick_review': {
                'incoming_rope': human['incoming_rope'],
                'inner_edge': human['inner_edge'],
                'tangent': human['tangent'],
                'note': human['note'],
            },
            'algorithm_distance_rope': row['cold_start']['distance_rope'],
        })

    sha256 = hashlib.sha256(REVIEW_PATH.read_bytes()).hexdigest()
    payload = {
        'schema_version': 1,
        'purpose': 'SAM 2少量锚点掩膜；不是设备控制输入',
        'source_review_sha256': sha256,
        'classes': [
            'wound_rope_surface',
            'flange_inner_face',
            'incoming_rope_free_span',
        ],
        'required_point': 'tangent_point',
        'tasks': tasks,
    }
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    (TASK_DIR / 'labels').mkdir(exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    counts = {
        field: Counter(item[field] for item in reviews.values())
        for field in ('incoming_rope', 'inner_edge', 'tangent')
    }
    mae = sum(abs(v) for v in comparable) / len(comparable)
    bias = sum(comparable) / len(comparable)
    within_one = sum(abs(v) <= 1.0 for v in comparable)
    text = f'''# 第二阶段复核结论与SAM 2标注说明

日期：{date.today().isoformat()}。本阶段只处理离线图片，不连接相机或控制设备。

## 23张优先关键帧复核结论

| 项目 | 正确 | 错误 | 未显示 | 正确率口径 |
|---|---:|---:|---:|---:|
| 活动绳 | {counts['incoming_rope']['正确']} | {counts['incoming_rope']['错误']} | 0 | {counts['incoming_rope']['正确']/len(reviews):.1%} |
| 内侧边界 | {counts['inner_edge']['正确']} | {counts['inner_edge']['错误']} | 0 | {counts['inner_edge']['正确']/len(reviews):.1%} |
| 切点/距离线 | {counts['tangent']['正确']} | {counts['tangent']['错误']} | {counts['tangent']['未显示']} | {counts['tangent']['正确']/len(reviews):.1%} |

共有{len(comparable)}张同时具备人工根数和算法数值。算法误差绝对值平均为{mae:.2f}根，算法相对人工的平均偏差为{bias:.2f}根，±1根内为{within_one}/{len(comparable)}。这里的偏差方向为“算法值－人工值”，因此负值表示少算。

结论：误差来自活动绳、内侧边界和切点的组合定位错误，并非单一固定偏移。不能直接给结果统一加2.4根，否则在换层、左右相机和不同厚度时会失效。

## 本轮为什么只标6张

第一轮锚点覆盖正确基准、静止失败、启动、停止、再次启动和后期卷绕状态。每张只处理三个区域和一个点：

1. `wound_rope_surface`：卷筒上已经卷入的钢丝绳可见表面。
2. `flange_inner_face`：紧邻卷绕面的金色法兰内侧可见面，不是最外轮廓。
3. `incoming_rope_free_span`：排线臂/导向处到首次接触卷绕面之间的活动绳。
4. `tangent_point`：活动绳第一次接触卷绕面的点。

SAM 2只负责根据点击给出候选掩膜；候选必须由人确认。6张完成后先检查细绳分割是否可用，再决定是否扩到其余图片，避免无效劳动。

## 操作顺序

1. 双击`安装SAM2辅助标注环境.cmd`。只需首次执行。首轮6张辅助标注使用独立CPU版环境，避免下载约2.6 GB的GPU训练包；正式训练仍会使用RTX 3080 Ti。
2. 安装完成后双击`启动SAM2辅助标注.cmd`。
3. 每个区域先点少量左键正样本；误选区域用右键点背景。使用“上一候选/下一候选”选择最贴合目标的掩膜。若三个候选和补点后仍明显错误，点击“SAM无法正确分割”，不要接受错误标签。
4. 切换到切点模式，左键点活动绳第一次接触卷绕面的位置。
5. 三个掩膜和切点都确认后点击“保存并下一张”。

所有结果保存在`vision_training/task_v1/labels`，不会改动原图和视频。
'''
    SUMMARY_PATH.write_text(text, encoding='utf-8')
    print(f'Reviewed {len(reviews)} frames; prepared {len(tasks)} SAM 2 anchor tasks.')
    print(f'Count MAE={mae:.3f}; bias(algorithm-human)={bias:.3f}; within_1={within_one}/{len(comparable)}')


if __name__ == '__main__':
    main()
