"""Compact actual-run receipt and software reproduction notes."""
from collections import Counter
from datetime import datetime,timezone
import sys
from common import HERE,H2,load,dump,sha

def main():
    s=load(HERE/'all_001/summary.json');i=load(HERE/'inputs.json');f=load(HERE/'freeze_final.json')
    assert s['exit_code']==0 and s['completed']==292 and s['source_unchanged']
    assert all(sha(p)==v for p,v in (f['sources']|f['reused_sources']).items())
    counts=dict(Counter(r['kind'] or 'no_candidate' for r in s['rows']))
    report=dict(version='IR-H3.0.0',completed_utc=datetime.now(timezone.utc).isoformat(),
      images=292,conditions=301,original_open_source_count=400,unresolved_entry_images=108,
      source_groups=len({j['source_group'] for j in i['jobs']}),unresolved_is_not_absent_rope=True,
      outputs=str(HERE/'all_001/四栏检测结果'),path_output_counts=counts,
      ambiguous_selected_images=sum(r['ambiguous'] for r in s['rows']),
      run_seconds=s['elapsed_seconds'],exit_code=s['exit_code'],B_model=i['model_B'],
      reused_float_responses=301,new_network_inference=0,training=False,
      implementations=dict(free='A7 observed axis, all existing H2 bounded same-axis pieces retained; robust TLS fallback explicitly identified; exact projection and raw/B recheck; old unknown never upgraded',
        surface='one saved F3 adjacent-boundary center track; surface points exactly equal source points in ordered source_indices; no off-track direct H1 continuation',
        score='0.50 image +0.40 direction +0.10 common120 half-px lane horizon coverage, minus unsupported bridge, abandoned straight evidence, width change and weak lane penalties',
        direction='24 half-px chord tangents; angle amplitude, total angular variation, width-normalized RMS curvature, entrance/landing tangent jump; lower directional change increases its score',
        straight_retention='reject bridge departing by over0.65 local width from more than10% of remaining observed straight points'),
      parameters=f['parameters'],pilots=['pilot_001','pilot_002'],
      initial_failure='pilot_001 used A7 last observed point as endpoint and truncated supported straight rope, causing premature coil join; preserved, corrected once by retaining H2 observed free extent',
      limitations=['Straight projection residual is an implementation constraint, not independent localization error',
        'F3 is a geometric candidate source, not target identity truth; exact adherence does not establish the correct rope',
        'Direction preference is within the new constrained pool; different/shorter paths do not prove improvement against H2',
        'Missing center tracks or incompatible joins remain free-only/unknown; no fixed-spacing fill',
        'No new independent target identity or endpoint reference was used'],
      labels_read=False,reserved_data_accessed=False,devices_connected=False,baseline_replaced=False,
      freeze_sha256=sha(HERE/'freeze_final.json'))
    comparison=HERE/'comparison_all_001/summary.json'
    if comparison.exists():
        cmp=load(comparison);b=cmp['by_image']
        report['paired_comparison']=dict(path=str(comparison),sha256=sha(comparison),summary=cmp)
        report['observed_changes']=dict(total_path_length_percent=100*(b['H3']['length_sum_original_px']/b['H2']['length_sum_original_px']-1),
            supported_length_percent=100*(b['H3']['supported_length_sum_original_px']/b['H2']['supported_length_sum_original_px']-1),
            common_direction_eligible_images=b['common_direction']['eligible_count'],
            universal_smoothness_improvement_demonstrated=False,full_identity_improvement_demonstrated=False)
    audit=HERE/'all_001/independent_geometry_audit.json'
    if audit.exists():
        a=load(audit);report['artifact_audit']=dict(path=str(audit),sha256=sha(audit),status=a['status'],images=a['checked_images'],conditions=a['checked_entry_conditions'])
    dump(HERE/'experiment_report.json',report)
    text=f'''热图接续 H3 / IR-H3.0.0 — A7直线修正与卷面中心轨迹上的方向变化评分

图库：{HERE/'all_001/四栏检测结果'}
292张四栏图：原图 / 冻结B热图 / 连接候选 / 最高评分候选，8192×1655，热图固定0–1。
输入沿用H2全部292图、301入口条件和浮点热图；原400来源中108张入口仍未解决，不称无绳。没有新模型前向或训练。

1. 未卷入段用匹配A7实测轴修正为直线；无可靠A7时用标记清楚的稳健直线回退。原H2可见范围和有界缺口保留，重新检查原图与B，仅降级支持，不把未知变观测。
2. 贴面点严格取一条F3灰色中心轨迹源点。H1离轨曲线不能作为主结果；无法接入则保留自由段，不用任意平滑线替代灰轨道。
3. 固定评分由图像0.50、方向0.40、共同尾部视野覆盖0.10及弱证据/丢失直段/宽度惩罚构成。方向项测量24半尺寸px弦的角幅、总变差、局部曲率、两端切向差，统一看接续后最多120半尺寸px；短于该视野会扣覆盖分。
4. 桥若偏离后续仍有图像支持的直段超过0.65局部宽的点数比例大于10%，直接排除。不会仅为少转弯而丢掉已观察直段。

实际292/292，退出0，{s['elapsed_seconds']:.1f}秒；产物类别{counts}。这些计数不是完整目标身份准确率。
绿=自由段、青=过渡、紫=贴面候选，橙虚线=弱证据/推断。A7轴/F3中心的精确符合仅说明约束已实现，不证明同根、真实接触或末端正确。
pilot_001提前截断反例保留；唯一修订后pilot_002再冻结，全量all_001。旧H2、B模型、原图、旧标签不变。
记录all_001/记录/*.json.gz，points_xy为半尺寸crop坐标，原图=2×xy+[0,1050]。surface_source_indices指向同entry.candidate_track_geometry的源点。
配置/源码哈希：inputs.json、freeze_final.json、all_001/run_start.json；逐图来源及响应哈希在记录。原环境vision_prototype/.venv，无依赖安装、标签/保留集读取、设备连接。

实际命令（D:/precise_arrangement）：
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v3/run_h3.py --run pilot_001 --subset pilot --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v3/run_h3.py --run pilot_002 --subset pilot --workers 4
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/postprocess_b_20261031_v3/run_h3.py --run all_001 --subset all --workers 4
复跑必须用新run名，不能覆盖旧结果。
'''
    if comparison.exists():text+='\n与H2同条件对比见comparison_all_001/summary.json：自由线约束已实现；总路径长度-17.76%，局部图像支持长度-9.22%，仅自由10→74。189图有足够共同下游，方向幅度差中位约+0.0034°、总变差+3.15°，没有普遍变平滑或完整识别改善的证据。必须连同路径/支持长度和共同范围不足数一起读，不能用截短造成的小角度宣称更好。固定12例旧新末栏并排在该目录。\n'
    if audit.exists():text+='独立产物核验见all_001/independent_geometry_audit.json，不是身份准确率评价。\n'
    (HERE/'README.txt').write_text(text,encoding='utf-8')
    print(counts)
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
