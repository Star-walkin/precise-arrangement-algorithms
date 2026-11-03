"""Save read-only A5 diagnosis and proposed A6 evaluation; no inference imports."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
ASSETS = OUT.parents[1]
A5 = ASSETS / 'camera_condition_incoming_photometric_20261029_v5'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol = json.loads((A5 / 'protocol.json').read_text(encoding='utf-8'))
crops = json.loads((OUT / 'crop_manifest.json').read_text(encoding='utf-8'))
notes = {
 'ruler_right_05': {
   'role':'right_camera', 'run':'pilot_003',
   'visible_evidence':'自由绳在卷面外有连续双外缘和斜股纹；由亮变暗发生在进入卷面之前，固定灰度均值或固定亮暗极性不能作为身份模板。原像素裁图中，左侧暗端仍有弱条带和逐渐变弯的纹理，但越靠侧缘越接近邻圈外观。',
   'search_requirement':'从可用自由段沿原图连续证据逐步推进，允许缓慢改变宽度/切线；不要把首次进入 SAM 搜索包络当转弯或贴轨点。保留相邻两侧轨道竞争，搜到最后可辨前段后再讨论 F3 邻轨。',
   'a5_limitation':'A5 选中弯曲轨道与纯几何 G0 相同；图像支持不能据此证明该轨道与自由段同根。暗端物理消失点未验证。'
 },
 'ruler_left_05': {
   'role':'left_camera', 'run':'pilot_003',
   'visible_evidence':'亮带内存在显著宽暗斜条带，宽度明显大于单股沟，穿过高亮背景后进入右侧暗面。原像素图上其对比度逐渐下降，右端邻圈纹理与它混合；没有可靠依据将整个暗侧强补为唯一曲线。',
   'search_requirement':'不能只从 F3 合法 Hermite 接点发出弯曲候选。需要沿可见斜条带连续推进，并在亮暗过渡附近保留不同开始弯曲时刻/曲率分支；允许暂不贴轨。',
   'a5_limitation':'A5 有可测自由宽度约25.86px，但所有160个几何接点均未合法，最终只有 straight_photo。候选缺失是结构问题，降低最终分数阈值不会生成晚弯候选。'
 },
 'dual_20260923_101643__pair0315__right': {
   'role':'right_camera', 'run':'pilot_003',
   'visible_evidence':'外部白亮自由绳经工具附近进入卷面，亮带中变为宽暗斜条带；原像素图中可辨条带向左下延续，A5 最终较平的上方轨道与这条走势存在偏离嫌疑。工具遮挡和照明变化必须允许低信息短缺口，但不能靠某个较亮旧圈接续。',
   'search_requirement':'候选至少同时包括继续沿原图暗斜带向左下推进和邻圈较平轨道；不要先把中心点吸附到 F3，再用 F3 双沟证明自身。比较应放在二者真正分叉位置，不被共同长自由段平均分淹没。',
   'a5_limitation':'最终 lane_path_01 端点 [1480,2519.4787878787915] 与上轮相同；抑制局部S形没有消除此错轨嫌疑。当前记录不能证明图像排序优于 G0。'
 },
 'dual_20260923_101643__pair0375__right': {
   'role':'right_camera', 'run':'extra_001',
   'visible_evidence':'原图中可清楚看到右上方白亮自由绳及亮带内宽暗斜条带，入口附近有工具遮挡；左暗侧后续逐渐难分。整体没有不可见到无法启动的理由。',
   'search_requirement':'缺旧 A1/A2 seed 时需要从本张原图的卷面外高对比条带产生多个自动直线种子，逐个检验双边、宽度、长度及是否朝向卷面。工具边和金色背景褶皱也必须作为竞争/拒绝情况，不能手给目标端点。',
   'a5_limitation':'free_rope 的 no_usable_automatic_free_seed 令搜索直接退出，0个候选；这属于明确可见绳漏覆盖，不应称 UNKNOWN 正确拒识。'
 }
}
cases = []
for stem, note in notes.items():
    rec = A5 / note['run'] / '记录' / note['role'] / (stem+'.json')
    j = json.loads(rec.read_text(encoding='utf-8')); f,w = j['free'],j['whole']
    c = next((v for v in w.get('candidates',[]) if v['candidate_id']==w.get('selected_candidate_id')),None)
    crop = next(v for v in crops if Path(v['source']).stem==stem)
    cases.append({
      'stem':stem, **note,
      'actual_viewed_original':{'path':crop['source'],'sha256':crop['source_sha256'],'method':'view_image of complete unmodified original'},
      'actual_viewed_native_crop':crop,
      'crop_role':'diagnostic display only; no crop coordinate or inferred endpoint is an AUTO input or target label',
      'a5_record_path':str(rec),'a5_record_sha256':sha(rec),
      'free_status':f['status'],'free_reason':f.get('diagnostics',{}).get('reason'),
      'free_width_px':f.get('geometry',{}).get('width_px'),
      'a5_status':w['status'],'candidate_count':len(w.get('candidates',[])),
      'selected_candidate_id':w.get('selected_candidate_id'),
      'selected_endpoint_xy':c['xy'][-1] if c else None,
      'material_identity_truth_available':False,'physical_contact_truth_available':False,
    })

sources=[A5/n for n in ['free_rope.py','photometric_ribbon.py','photometric_trace.py','lane_candidates.py','protocol.json']]
reviews=[A5/'pilot_003'/'independent_visual_review.json', A5/'extra_001'/'independent_visual_review.json']
issues=[
 {'code':'free_rope.py:132 and :470','finding':'只接收已保存 A2/A1 observed 种子，seed缺失即退出；原图不重新提名自由条带。','impact':'Rdual375 明确可见活动绳却完全没有模板和候选。','minimal_change':'新独立自动seed fallback；保留所有原入口失败在400总分母，记录新seed来源和拒绝理由。'},
 {'code':'photometric_ribbon.py:81, :160-170','finding':'自身信息主要是一个股纹方向统计，只以0.025权重给弱辅助；主体仍为通用亮暗差和外缘响应，没有自身剖面/纹理模板。','impact':'重复旧圈可给同样的双沟或高分；照明改变也使固定均值模板不合适。','minimal_change':'用多个可靠自由截面建立鲁棒宽度、归一化横剖面、相对轴线股纹方向/频带模板；保留原灰度可测性，饱和模板与可靠纹理模板分开，不从歧义尾自更新。'},
 {'code':'photometric_trace.py:154; lane_candidates.py:246-308','finding':'先从F3相邻轨道及有限Hermite接点产生候选，再由RGB评分；图像只评分，无法产生缺失的弯法。','impact':'L05全接点被几何拒绝；R315仍只能在预设旧圈轨道中选择。','minimal_change':'原图先行的有界多分支搜索，跨若干逐步曲率/宽度选项；图像支持的局部弯段成立后才把F3作为邻接/方向约束。'},
 {'code':'photometric_trace.py:175-195','finding':'存在合格lane时优先选lane，再输出单条whole_mask；歧义仅加review标记。','impact':'用户看到的是一条完整尾，容易把有标记的错接当成已识别整根。','minimal_change':'主掩膜输出共同且有证据的前缀；分叉后保留多候选另图/JSON，候选完整掩膜与可信前缀明确区分。'},
]
pilot_inputs=list(protocol['pilot_inputs'])
missing_seed='images/right_camera/dual_20260923_101643__pair0375__right.png'
if missing_seed not in pilot_inputs: pilot_inputs.append(missing_seed)
evaluation={
 'status':'recommendation_for_parent_to_freeze_before_A6_prediction; not an executed protocol',
 'main_denominator':400,'left':200,'right':200,
 'fixed_development_inputs':pilot_inputs,
 'fixed_development_count':len(pilot_inputs),
 'sample_scope':'继承 A5 固定20张，加本次明确指定的漏seed375。该图已在A5额外组查看，不再称独立额外验证。此建议不打开其余额外图或保留集。',
 'budget_recommendation':'最多3轮固定清单开发；每轮只针对已实证通用问题，保留运行和失败；冻结后同400全量只导出/检查，不根据新挑出的图片继续调参。',
 'baseline':'A5 all_001 unchanged, same original400 and saved F3 dependencies; no replacement of F3/C3/D4',
 'comparisons':[
   'A5 原输出 vs A6 同原图；入口fallback是否把可见绳从无输出恢复，是否出现工具/背景新误检。',
   'A6 同一保存候选池下几何排序、无自模板图像排序、自模板图像排序；必须保存不满足门槛的候选及原因，不能对G0先应用G2筛选。',
   '若图像搜索阶段已使用模板，排序消融只评价排序增益；不能冒称整个候选生成的无模板对照。可另保存一个固定无模板beam运行，预算相同且全21输入。',
   '逐图看共同前缀及分叉之后：可见条带是否被覆盖、是否跳到邻圈、是否新增背景尾、是否过早终止、歧义是否明确输出。',
 ],
 'required_per_case_fields':['automatic_seed_status_and_source','candidate_pool_before_selection','candidate_generation_mode','image_supported_prefix','branch_disagreement_location','raw_measurement_availability','original_gray_contrast','self_template_match','neighbor_template_match','width_and_tangent_trace','F3_constraint_activation_reason','stop_reason','supported_mask','uncertain_alternatives','failures'],
 'main_visual_gate':[
   '375：自动取得真实可见自由条带且亮带前段覆盖；出现任意非空线不算修复。',
   'L05：先确认候选池新增晚弯/跨轨图像路径，再评选择；单纯少候选或缩短不能算改进。',
   'R315：比较原暗斜条带与上方旧圈分支；原错路退为可信前缀是风险减少，但不能称同根弯段正确。',
   'R05：保留原可见自由段及合理弯曲支持，不用更早停止掩盖困难区域。',
   '固定20中的f15、113305pair15/195亮带覆盖和工具/背景失败样本保持对照，不能只看新四张。',
 ],
 'continue_stop':'有限开发轮次内若没有可见条带覆盖/候选合理性的新证据，或新背景错路抵消改善，不以路径更长、得分更高或拒识更多宣布改善；保留可信前缀和多解并报告研究未验收。全400按用户要求保留所有图及未知状态。',
 'metrics_limits':'无逐根/接触真值，不报整根准确率；像素长度、支持率、分差只是诊断。用户约50%是质性观察，不作为数值真值。'
}
report={
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'reviewer':'/root/incoming_review','task':'A6 preimplementation independent read-only A5 diagnosis',
 'scope':'Offline original4 visual inspection; A5 code and saved20/8 review records; no inference, no core edits, no new extra/reserved imagery.',
 'actual_images_viewed':{'complete_originals':4,'native_unmodified_crops':4,'unique_source_images':4},
 'source_hashes':[{'path':str(p),'sha256':sha(p)} for p in sources],
 'prior_review_hashes':[{'path':str(p),'sha256':sha(p)} for p in reviews],
 'cases':cases,'a5_structural_issues':issues,'minimal_closed_loop_evaluation':evaluation,
 'search_design_constraints':[
   '全部搜索坐标来自自动seed/原图/F3，不读本诊断裁图框，不用人工暗端或切点。',
   '保留现有约2800px前向预算作为统一上限起点，不能只搜索SAM刚入域的短窗口。R315/R05源自由端到暗端相隔千像素；依证据逐步推进。',
   '采用局部切线坐标滚动分支，横向竞争涵盖邻绳；不能始终限制为原直线附近窄条或强制先落在F3。精确步长/beam预算由主协议统一冻结。',
   '模板需对局部增益/偏置和亮暗反转有受控适应；原图低信号或饱和不能凭归一化制造证据。相位重复的模板高分不自动确认材料身份。',
   '短遮挡可保留有预算的未决分支；没有持续可测支持时停止主前缀，多候选另存，避免无限外推/确认消失点。',
 ],
 'no_auto_ground_truth_injection':True,'devices_connected':False,'reserved_data_read':False,
 'core_algorithms_modified':False,'prediction_or_accuracy_experiment_run':False,
}
p=OUT/'a6_readonly_diagnosis.json'
with p.open('x',encoding='utf-8') as out: json.dump(report,out,ensure_ascii=False,indent=2)
text=['A6实现前独立诊断：只读A5，未跑新算法。','实际查看4张完整原图与4张未增强原像素裁图。诊断裁图绝不作为AUTO输入。','']
for issue in issues: text += [issue['code'],issue['finding'],issue['impact'],issue['minimal_change'],'']
for c in cases: text += [c['stem'],c['visible_evidence'],c['search_requirement'],c['a5_limitation'],'']
text += ['建议固定20+明确指定375共21开发图；原400分母不变。',evaluation['budget_recommendation'],evaluation['continue_stop'],evaluation['metrics_limits']]
with (OUT/'a6_readonly_diagnosis.txt').open('x',encoding='utf-8') as out: out.write('\n'.join(text)+'\n')
print(json.dumps({'path':str(p),'sha256':sha(p),'fixed_development_count':len(pilot_inputs)},ensure_ascii=False))
