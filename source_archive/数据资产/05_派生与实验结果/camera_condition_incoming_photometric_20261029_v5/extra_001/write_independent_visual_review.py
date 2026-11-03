"""Persist independent, post-freeze visual observations; never run inference."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

RUN = Path(__file__).resolve().parent
ROOT = RUN.parent
ASSETS = ROOT.parent
ORIGINAL = ASSETS / 'camera_condition_images_20261009_v2' / 'images'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# These notes were written after actual view_image inspection of every listed
# qa image and the three candidate images. They are not generated metric labels.
observations = [
    ('left_camera', 'L_run_stop_run_20260922_01__f002742__left', True,
     'A4 与 A5 均覆盖自由斜绳和亮带上的可见斜纹。A5 接续尾段接近直线并略有弯曲；候选图中较早向上拉平的紫线与可见斜纹分离，所选蓝线较贴合可见走势。',
     '候选之间仍竞争；暗尾端逐像素贴合及同根身份不能由此总览确认。未确认覆盖到物理消失端。',
     'plausible_selected_path_with_unverified_tail'),
    ('left_camera', 'dual_20260923_101643__pair0015__left', False,
     'A4/A5 均为空输出。当前 QA 裁图中没有清楚可辨的自由斜绳。',
     '没有独立绳存在标签；不能将 UNKNOWN 计为正确负例或成功。',
     'unknown_presence_not_scored'),
    ('left_camera', 'dual_20260924_111143__pair0015__left', False,
     'A4/A5 均为空输出。当前 QA 裁图主要显示卷面，没有清楚可辨的自由斜绳。',
     '未取得全图存在真值；UNKNOWN 保持在分母内，不计正确负例。',
     'unknown_presence_not_scored'),
    ('left_camera', 'dual_20260924_111500__pair0195__left', True,
     'A5 比 A4 多给出右侧向下弯曲的尾段。候选图显示相邻轨道分叉，所选更弯的路径仍位于卷面边侧；总览未见新增金色背景长尾。',
     '弯曲延伸是否选中原活动绳而非邻圈仍不确定，不能把更长覆盖称整根正确。',
     'curved_coverage_extension_identity_ambiguous'),
    ('right_camera', 'dual_20260923_101643__pair0375__right', False,
     '原图清晰可见由右上进入、穿过亮带向左下延续的长活动绳，但 A4/A5 均没有掩膜。记录显示自动入口无可用种子。',
     '这是可见活动绳漏覆盖，不能以 UNKNOWN 遮蔽错误；未进入轨道候选阶段。',
     'explicit_visible_rope_missed_at_automatic_entry'),
    ('right_camera', 'dual_20260923_101517__pair0585__right', True,
     '自由斜绳覆盖保持；A5 比 A4 增加左侧向下弯曲的卷面候选，候选图显示两条非常接近的弯曲竞争路径。总览中路径仍在卷面上，未见明显金色背景漏出。',
     '接触位置、邻圈身份和最终消失端均未确认，保留歧义。',
     'curved_coverage_extension_identity_ambiguous'),
    ('right_camera', 'dual_20260924_111143__pair0165__right', False,
     'A5 沿可见细斜绳输出，未明显跳到画面中另一根粗亮工具；在左侧暗卷面接续前停止。',
     '仅直段部分覆盖，缺少可验证的卷入弯段；不称整根成功。',
     'partial_straight_only'),
    ('right_camera', 'R_run_20260923_01__f000092__right', False,
     '相对 A4，A5 明显延长了亮带内原图清楚可见的暗斜绳覆盖。这是可见直段覆盖增益。',
     '仍为 straight fallback，结束于左侧暗过渡附近，没有确认弯段和消失端。',
     'visible_straight_coverage_gain_partial_only'),
]

summary = json.loads((RUN / 'summary.json').read_text(encoding='utf-8'))
freeze = json.loads((RUN / 'freeze_snapshot.json').read_text(encoding='utf-8'))
rows = []
viewed = []
for role, stem, candidate_viewed, observation, limit, category in observations:
    rp = RUN / '记录' / role / (stem + '.json')
    r = json.loads(rp.read_text(encoding='utf-8'))
    w, f = r['whole'], r['free']
    chosen = next((c for c in w.get('candidates', []) if c['candidate_id'] == w.get('selected_candidate_id')), None)
    images = [RUN / 'qa' / role / (stem + '.jpg')]
    if candidate_viewed:
        images.append(RUN / 'qa_candidates' / role / (stem + '.jpg'))
    evidence = [{'path': str(p), 'sha256': sha(p), 'inspection': 'actual view_image visual inspection'} for p in images]
    viewed.extend(evidence)
    original = ORIGINAL / role / (stem + '.png')
    rows.append({
        'camera_role': role, 'stem': stem, 'status': w['status'],
        'selected_candidate_id': w.get('selected_candidate_id'),
        'g0_selected_candidate_id': w.get('g0_selected_candidate_id'),
        'selected_endpoint_xy': chosen['xy'][-1] if chosen else None,
        'automatic_entry_reason': f.get('diagnostics', {}).get('reason'),
        'whole_reason': w.get('diagnostics', {}).get('reason'),
        'candidate_count': len(w.get('candidates', [])),
        'observation': observation, 'limitation': limit, 'qualitative_category': category,
        'record_path': str(rp), 'record_sha256': sha(rp),
        'original_path': str(original), 'original_sha256': sha(original),
        'original_inspection_scope': 'original panel in existing QA comparison; no manual target annotation',
        'viewed_evidence': evidence,
        'same_material_identity': 'not_evaluated_no_independent_labels',
        'physical_contact_or_disappearance_verified': False,
    })

freeze_checks = []
for p, expected in freeze['sources_sha256'].items():
    path = Path(p)
    snap = RUN / 'source_snapshot' / path.name
    freeze_checks.append({
        'source_path': str(path), 'expected_sha256': expected,
        'current_sha256': sha(path), 'snapshot_sha256': sha(snap),
        'current_matches_freeze': sha(path) == expected,
        'snapshot_matches_freeze': sha(snap) == expected,
    })

report = {
    'reviewer': '/root/incoming_review independent visual QA',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'version': 'IR-A5.0.0', 'run': str(RUN),
    'scope': '8 preset open-development images after source freeze; observational check, not blind test, no image tuning',
    'run_summary': {k: summary[k] for k in ['completed', 'planned', 'statuses', 'failures', 'sources_unchanged', 'exit_code']},
    'actual_view_count': {'three_panel_qa': 8, 'candidate_overlays': 3, 'total_unique_images': len(viewed)},
    'all_preset_cases_retained': True,
    'denominator_rule': 'Initial surface/free-entry failure remains in full400 denominator; UNKNOWN and rejection are not successes.',
    'no_new_core_edits_or_tuning': True,
    'devices_connected': False, 'reserved_data_read': False,
    'source_freeze_path': str(RUN / 'freeze_snapshot.json'),
    'source_freeze_sha256': sha(RUN / 'freeze_snapshot.json'),
    'source_freeze_checks': freeze_checks,
    'all_9_sources_and_snapshots_match_freeze': all(x['current_matches_freeze'] and x['snapshot_matches_freeze'] for x in freeze_checks),
    'conclusions': [
        '实际查看 8/8 QA 和 3 张歧义候选图。3 UNKNOWN、3 歧义轨道候选、2 直段输出均保留。',
        'Rf000092 有可见直段覆盖增益；两张侧缘候选增加弯曲覆盖，但均未证明同根身份。',
        'Rdual101643 pair0375 明确可见活动绳却入口失败；该漏覆盖保持原样。',
        '其余两张 UNKNOWN 只说明当前未输出；没有存在真值，不当作正确负例。',
        '冻结后未调参。没有本组新的明确金色背景长尾证据；这不等于排除邻圈错接、保证全程覆盖或确认准确率。',
    ],
    'quantitative_accuracy_or_identity_rate': None,
    'cases': rows,
}
target = RUN / 'independent_visual_review.json'
with target.open('x', encoding='utf-8') as out:
    json.dump(report, out, ensure_ascii=False, indent=2)
lines = ['IR-A5.0.0 extra_001 冻结后独立视觉复核',
         '实际查看：8 张原图/A4/A5 三联图，3 张候选图。开放开发观察，非盲测。',
         '不改算法、不调参、不连接设备、不读保留集；所有失败保留。',
         '9 个冻结源及运行快照哈希一致：' + str(report['all_9_sources_and_snapshots_match_freeze']), '']
for item in rows:
    lines.extend([item['stem'] + ' | ' + item['status'], item['observation'], item['limitation'], ''])
lines.extend(report['conclusions'])
with (RUN / 'independent_visual_review.txt').open('x', encoding='utf-8') as out:
    out.write('\n'.join(lines) + '\n')
print(json.dumps({'report': str(target), 'sha256': sha(target), 'actual_viewed': len(viewed),
                  'freeze_matches': report['all_9_sources_and_snapshots_match_freeze']}, ensure_ascii=False))
