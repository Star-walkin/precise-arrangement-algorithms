"""Build the current result index from completed, source-bound receipts."""
from pathlib import Path
from collections import Counter
import csv
import gzip
import json
import math
from lc1_video import dump

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))


def main():
    live_name='runs/live_observed_turn_left_10_26'
    live=read(live_name+'/run_receipt.json')
    assert live['status']=='completed' and live['processed_frames']==16 and live['stride']==1
    with (ROOT/live_name/'measurements.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert [int(r['frame_index']) for r in rows]==list(range(10,26))
    with gzip.open(ROOT/live_name/'diagnostics.jsonl.gz','rt',encoding='utf-8') as stream:
        details=[json.loads(line) for line in stream if line.strip()]
    assert len(details)==16
    s2=[r['front']['diagnostics']['S2'] for r in details]
    s2_states=Counter(r['diagnostics']['guidance_result']['status'] for r in s2)
    reasons=Counter(r['turn_reason'] for r in rows)
    uncertainty=[float(r['turn_uncertainty_px']) for r in rows if r['turn_uncertainty_px'] and math.isfinite(float(r['turn_uncertainty_px']))]
    sampled=read('runs/actual_models_observed_turn_20261105/receipt.json')
    tests=read('verification_receipt.json')
    output=dict(algorithm='LC1.2',assets_received_and_verified=True,assets_missing=[],
        actual_P1_H6_S1_algorithms_connected=True,formal_automatic_d_and_count_validated=False,
        source_assets_original='E:/study/program/济宁/临时文件',
        source_assets_preserved_copy='supplied_assets',model_sha256=live['live_model_sha256'],
        continuous_run=dict(path=live_name,source_frame_range_exclusive=[10,26],source_frames=16,
            encoded_frames=16,source_sampling_stride=1,actual_P1_model_forwards=live['actual_P1_model_forward_count'],
            actual_S1_core_frames=live['actual_S1_core_execution_frames'],
            actual_S2_guidance_audit_frames=sum(r['diagnostics'].get('guidance_executed') is True for r in s2),
            S2_guidance_status_counts=dict(s2_states),
            automatically_locked_d_frames=sum(bool(r['d_original_px']) for r in rows),
            accepted_count_frames=live['valid_count_frames'],whole_path_supported_frames=live['complete_path_candidate_frames'],
            turn_rejection_reasons=dict(reasons),median_algorithm_ms=live['median_algorithm_ms'],
            wall_seconds=live['wall_seconds'],
            finite_turn_uncertainty_min_max_px=[min(uncertainty),max(uncertainty)] if uncertainty else None,
            unmodified_source_video=live['source_stable'],original_speed_video=live_name+'/overlay.mp4',
            labelled_slow5x_review_video=live_name+'/review_slow5x.mp4'),
        sampled_current_model_results=dict(total_sampled_source_frames=12,not_continuous_inference=True,
            newer_observed_turn_estimator=True,runs=sampled['runs']),
        test_summary=dict(passed=tests['success'],tests_run=tests['tests_run'],scope=tests['scope']),
        limitation='No calibrated d or accepted integer rope count; current upstream F3/D4 scene adaptation is not complete historical F3 fusion.')
    dump(ROOT/'experiment_summary.json',output)
    lines=['# LC1.2 实际运行结果','',
        '**资产缺失已解除。** 用户提供的 B750 权重与 S1/S2 代码已验证并原样备份。当前实现可以运行真实算法，但本批视频还没有通过自动 d 标定，不能报告可靠整数剩余根数。','',
        '## 连续逐原帧核验','',
        f'源视频为左相机 dual_20260924_111500，源帧 10–25，共 16 帧，无跳帧。每帧真实执行 B750/H6、S1、独立 S2 自动提案审计与 F3；B750 实际前向 {live["actual_P1_model_forward_count"]} 次、S1 实际核心 {live["actual_S1_core_execution_frames"]} 帧。','',
        f'[原速视频：1.6 秒]({live_name}/overlay.mp4) · [带标识的 5 倍慢放：8 秒]({live_name}/review_slow5x.mp4) · [逐帧 CSV]({live_name}/measurements.csv) · [运行记录]({live_name}/run_receipt.json)','',
        f'全部 16 帧维持 calibrating，自动 d=0，正式接受计数=0。当前配置中位耗时 {live["median_algorithm_ms"]/1000:.2f} s/帧，总耗时 {live["wall_seconds"]:.1f} s；没有达到实时推理速度。源视频未改动。','',
        '转折拒绝原因（不是“真实没有弯曲”的判定）：','']
    labels={'insufficient_current_supported_surface_points':'当前曲段观测点不足',
            'no_unique_legal_H6_curve_connection':'缺少唯一合法 H6 曲连接',
            'straight_or_unresolved_supported_surface_curvature':'曲率尚不能可靠区分',
            'turn_localization_uncertain':'d 定位范围过宽',
            'turn_profile_open_or_boundary_limited':'包含锚线扰动后定位范围未收敛'}
    lines += [f'- {labels.get(reason,reason)}：{count} 帧。' for reason,count in reasons.items()]
    lines += ['',f'S2 独立提案状态：{dict(s2_states)}。S2 未用于替代 S1 的 ABC/C，也未用于提升完整路径许可。','',
        '## 左右抽样核验：修正后的转折估计','',
        '| 相机 | 实际源帧 | 显示方式 | 自动 d / 正式计数 |','|---|---|---|---|',
        '| [左视频](runs/actual_models_observed_turn_20261105/left/overlay.mp4) | 0、15、30、45、60、75 | 每 1.5 s 取一帧，逐样本停留；输出 9 s | 0 / 0 |',
        '| [右视频](runs/actual_models_observed_turn_20261105/right/overlay.mp4) | 240、270、300、330、360、390 | 每 3 s 取一帧，逐样本停留；输出 18 s | 0 / 0 |','',
        '两段重放已验证的真实 P1/H6 和 S1 结果，在同一源图上重新运行 F3 和修正后的转折估计。不是连续每源帧推理，停留画面不构成新检测。H6 连接参考列和被拒绝的 d 候选均以橙色显示，不能称为锁定 d。抽样间隔超过原状态机 1 s 连续性门槛，故它们只用于找候选窗口；上面的连续 16 帧单独核验了连续性。','',
        '## 修复与判断','',
        '旧全局拟合会裁掉偏差最大的 10% 点，可能把仅占约 8% 的真实曲段整体删除。LC1.2 真实后端改为独立当前 S1 锚线与受支持曲段拟合，不使用 Hermite 过渡或无观测点，并计入锚线和空间相关性的不确定度。10 项新回归覆盖曲段占比、自由段加密、伪观测、镜像及无合法连接等反例。','',
        '修复后，真实左 f0 能拟合出曲段，但其条件定位包络约 ±111 px，远大于半个绳距约 11.6 px；因此仍需拒绝锁列。这不是缺权重，也不意味着概念必然不可行，而是当前视频和路径结果尚未给出足够局部约束。对当前数据不能宣布 A/B/d 恒定假设已验证。','',
        '若继续改进，应优先让转折附近有更长、可区分的活动绳曲段观测，提升 H6 曲支选择，并用人工转折点/根数建立独立评价；不能仅靠把拒绝阈值调宽获得成功状态。当前 F3 为当前图像观测核心与 D4 场景适配，未复现旧完整融合缓存。','',
        f'工程测试：{tests["tests_run"]} 项通过，验证记录见 [verification_receipt.json](verification_receipt.json)；这些测试不等于图像精度或物理根数验证。','',
        '[算法与运行说明](README.md) · [完整资产接入报告](ASSET_INTEGRATION_RESULTS.md) · [转折独立审阅](LC1_REAL_CALIBRATION_REVIEW.md) · [S1/S2 说明](S1_S2_INTEGRATION_NOTES.md) · [最新输出核验](final_live_models_audit.json)','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(continuous_frames=16,accepted_count_frames=live['valid_count_frames'],tests=tests['tests_run'],S2_states=dict(s2_states)),ensure_ascii=False))


if __name__=='__main__':main()
