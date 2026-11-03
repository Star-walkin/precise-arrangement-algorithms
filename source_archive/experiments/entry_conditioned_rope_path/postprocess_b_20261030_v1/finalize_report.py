"""Collect existing H1 evidence into a delivery report; never fabricate results.

Run only after the final repaired pilot, extra001, all001 and independent audits
have finished. Earlier trials and failed audits remain visible in the report;
they are not required to become successful retroactively. Default mode requires
the final repaired pilot and all001 to have passing latest independent audits.
``--allow-incomplete`` is an explicit diagnostic option: missing
evidence is retained as missing and the report cannot claim completion.
No detector is imported, no image prediction is changed, and no old source,
model, annotation, project index or AGENTS.md is edited.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
FINAL_PILOT='pilot_003_repaired_003'
FINAL_RUNS=(FINAL_PILOT,'extra_001','all_001')


def discover_run_names():
    # This directory was created for H1. Avoid following any Windows reparse
    # point while inventorying immediate pilot directories.
    pilots=[]
    for p in HERE.iterdir():
        if p.name.startswith('pilot_') and p.is_dir():
            attributes=getattr(p.lstat(),'st_file_attributes',0)
            if attributes & 0x400:continue
            pilots.append(p.name)
    if FINAL_PILOT not in pilots:pilots.append(FINAL_PILOT)
    return tuple(sorted(pilots))+('extra_001','all_001')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)


def reference(path):
    path=Path(path)
    return {'path':str(path.resolve()),'sha256':sha(path),'bytes':path.stat().st_size}


def short_audit(data):
    # Keep actual failures, counts and interpretation limits. Huge per-pixel
    # / per-case evidence remains in its original hashed audit file.
    return {k:v for k,v in data.items() if k not in ('cases','checks','sources')}


def collect_run(name,evidence):
    run=HERE/name;start_path=run/'run_start.json';summary_path=run/'summary.json'
    item={'run':name,'directory':str(run),'start_present':start_path.exists(),'summary_present':summary_path.exists()}
    if start_path.exists():
        start=load(start_path);evidence.append(reference(start_path))
        item['start']={k:v for k,v in start.items() if k!='input_ids'}
        item['input_ids']=start.get('input_ids',[])
        source_checks=[]
        for path,expected in start.get('sources',{}).items():
            current=Path(path);snapshot=run/'source_snapshot'/current.name
            source_checks.append({'path':path,'expected_sha256':expected,
                'snapshot':str(snapshot),'snapshot_matches':snapshot.exists() and sha(snapshot)==expected,
                'current_matches':current.exists() and sha(current)==expected})
        item['source_checks']=source_checks
    if summary_path.exists():
        summary=load(summary_path);evidence.append(reference(summary_path))
        item['summary']={k:v for k,v in summary.items() if k!='rows'}
        item['completed_ids']=[r['id'] for r in summary.get('rows',[])]
    audits=[]
    if run.exists():
        for p in sorted(run.glob('independent_audit*.json')):
            audits.append({'file':reference(p),'report':short_audit(load(p)),'mtime_ns':p.stat().st_mtime_ns})
            evidence.append(reference(p))
    item['independent_audits']=audits
    if audits:
        latest=max(audits,key=lambda a:a['mtime_ns'])
        item['latest_independent_audit']=latest['file']
        item['latest_independent_audit_passed']=latest['report'].get('exit_code')==0 and latest['report'].get('passed') is True
    else:item['latest_independent_audit_passed']=False
    return item


def collect_reviews(evidence,run_names):
    # Bounded named directories only; do not recurse through source assets,
    # junctions, labels or arbitrary adjacent experiments.
    directories=[HERE,HERE/'visual_diagnosis']
    for name in run_names:directories += [HERE/name,HERE/name/'qa_visual']
    found={}
    for directory in directories:
        if not directory.is_dir():continue
        for p in directory.glob('*review*.json'):
            found[str(p.resolve())]=p
    out=[]
    for p in sorted(found.values()):
        ref=reference(p);evidence.append(ref)
        out.append({'file':ref,'report':load(p)})
    return out


def count_final_outputs(run_item,protocol,evidence):
    out={'count_semantics':'artifact/output counts only; not correct-rope counts or accuracy'}
    if not run_item['summary_present']:return out
    root=HERE/'all_001';lookup={j['id']:j for j in protocol['jobs']}
    statuses=Counter();terminations=Counter();camera=Counter();groups=Counter();files=Counter()
    lengths=[];condition_count=0;candidate_count=0;support_points=0;points=0;record_count=0;missing=[];tail_count=0
    for case_id in run_item.get('completed_ids',[]):
        path=root/'记录'/(case_id+'.json')
        if not path.exists():missing.append(str(path));continue
        record=load(path);record_count+=1
        job=lookup[case_id];camera[job['camera_role']]+=1;groups[job['source_group']]+=1
        files['records_json']+=1
        for entry in record['entries']:
            condition_count+=1;candidates=entry.get('candidates',[]);candidate_count+=len(candidates)
            tails=entry.get('f3_tail_review',{}).get('candidates',[]);tail_count+=len(tails)
            if tails:statuses['entries_with_F3_review_tails']+=1
            if len(tails)>1:statuses['entries_with_competing_F3_review_tails']+=1
            if entry.get('automatic_entry_conflict'):statuses['entry_conflicts']+=1
            if candidates:
                statuses['entries_with_candidate']+=1;top=candidates[0]
                terminations[top.get('termination','unspecified')]+=1
                lengths.append(float(top.get('length_px',0)))
                supported=top.get('supported',[]);support_points+=sum(bool(x) for x in supported);points+=len(supported)
                if any(supported):statuses['entries_with_local_supported_points']+=1
                else:statuses['entries_without_local_supported_points']+=1
            else:statuses['entries_without_candidate']+=1
        for rel in record.get('files',[]):
            p=root/rel
            if not p.exists():missing.append(str(p))
            parts=Path(rel).parts
            if parts and parts[0]=='分步图像':files['step_jpg']+=1
            elif parts and parts[0]=='长条对照图':files['long_strip_jpg']+=1
            elif parts and parts[0]=='掩膜':files['mask_png']+=1
            else:files['other_listed_output']+=1
    out.update(record_count=record_count,entry_conditions=condition_count,candidate_count=candidate_count,
               F3_review_tail_count=tail_count,F3_review_tails_excluded_from_main_masks=True,
               statuses=dict(statuses),termination_counts=dict(terminations),camera_counts=dict(camera),
               source_group_counts=dict(groups),formal_file_counts=dict(files),missing_files=missing,
               selected_path_point_count=points,selected_local_supported_point_count=support_points,
               selected_length_px={'minimum':min(lengths) if lengths else None,'maximum':max(lengths) if lengths else None,
                                   'mean':sum(lengths)/len(lengths) if lengths else None},
               point_count_and_length_are_not_identity_accuracy=True,
               directories={'strips':str(root/'长条对照图'),'steps':str(root/'分步图像'),
                            'candidate_masks':str(root/'掩膜/候选绳带'),'local_support_masks':str(root/'掩膜/局部支持绳带'),
                            'records':str(root/'记录')},
               panel_scale=.5,panel_photo_shape_hw=[1500,2048],
               strip_width_px=12288,strip_height_px_per_entry=1600,
               multiple_entries='separate rows per image; separate per-entry masks; not merged into one claimed rope',
               masks='original 3000x4096 estimated optical ribbon, background0 foreground255',
               jpeg_display_not_original_pixel_equality=True)
    return out


def algorithm_text(report):
    status=report['execution_status'];output=report['outputs'];summary=report['runs'][-1].get('summary',{})
    counts=output.get('formal_file_counts',{})
    audit=report['numeric_response_audit']['report']
    lines=[
      '热图接续 H1（IR-H1.0.0）算法与结果说明',
      '生成时间：'+report['created_utc'],
      '',
      '一、这次做了什么',
      '使用用户指定方案B的原模型与原自动入口，重新保存数值热图，再结合原图、A6/A7自由段提示和F3卷面边界搜索活动绳位置。没有新增训练，原模型、旧实验及原图保持不变。',
      f"输入来自开放400图。旧图库实际可用{report['data']['eligible_images']}张、{report['data']['entry_conditions']}个入口条件；原有{report['data']['upstream_missing_entries']}张缺入口，未在本次冒充已识别或无绳。",
      '这些图包含旧模型训练使用过的采集来源组，不是新的独立测试集。',
      '',
      '二、长条图从左到右的六步',
      '1. 原图：以50%尺寸显示。显示图使用JPEG；原始4096×3000文件内容不变。',
      '2. 方案B数值热图：原权重第750步推理，输入是灰度与最初120原图像素的自动入口。计算域[0,1050,4096,3000]，半尺寸2048×975；灰区未计算。固定0～1色标，响应是中心结构强度，不是概率或真实绳宽。原浮点响应保存在responses/<图名>/entry_XX.npz。',
      '3. 局部对比诊断：原图与热图的不同尺度高斯滤波差（DoG）显示去掉缓慢亮度变化后的亮暗结构，仍会含相邻旧圈和工具。它是诊断展示，不是路径搜索实际使用的二值候选掩膜。搜索时另对每一个位置、方向和宽度提案测量法向剖面，检查亮绳体、暗绳体或双侧亮边，并沿绳方向适度汇聚以减少细股纹干扰。',
      '4. 自动几何提示：沿用每个原自动入口，A6提供对应直段的入口、方向与宽度，A7用于兼容性复核和直轴提示；实际外观剖面在本次原图自由段重新测量，保存的A6外观模板只留作来源记录，未直接参与评分。F3提供卷面边界，仅是局部弱约束，不能按最近一条边界独自决定同根身份。',
      '5. 多路径搜索：同时保留原图与热图融合搜索、热图评分搜索，以及有原图支持时的固定直轴接续候选。自由段轴线经双边剖面修正；长弱间隙两端可按同轴重现产生候选，但间隙不算实测。根据方向、宽度、转弯平滑度和图像证据排序，保留有限竞争路径，热图搜索也使用同一几何及停止约束。末端还可基于F3成对边界提出最多3条相邻轨道尾段供复核，保留多解，不强制唯一身份。',
      '6. 最终叠加：绘制暂选路径及估计宽度。绿表示局部原图/热图匹配，橙表示弱证据待确认；仅靠清晰热图继续的部分仍为橙色。橙色虚线是F3竞争尾段，只在第5、6图展示，不加入主掩膜。若入口与其他已接受A7轴冲突，保留其候选供检查但不标为局部支持绳。去除无支持的末端延长；并未证明抵达正确整根终点。',
      '',
      '三、重要颜色和掩膜含义',
      '绿色不等于已确认同一根活动绳。光学宽度是原图像素估计，没有主体掩膜真值监督，也不是毫米绳径。',
      '候选绳带包含橙色弱证据连接，仅供检查；局部支持绳带只保留相邻点都具备局部支持的部分。长遮挡推断和待确认尾段不能当作实测像素。',
      'F3橙色虚线尾段保存在记录的entries[].f3_tail_review中，坐标为半尺寸裁剪坐标；不写入候选绳带或局部支持绳带。多条相邻轨道可能都吻合图像，因此不把排序第一当作已确认接法。',
      '每图有多个入口时分行展示；不会把工具入口与绳入口热图直接合并。',
      '',
      '四、实际执行与检查（以下从已保存报告读取）',
      f"收集状态：{status}",
      f"全量计划/完成：{summary.get('planned','尚无报告')}/{summary.get('completed','尚无报告')}；退出码：{summary.get('exit_code','未知')}；失败条数：{len(summary.get('failures',[])) if summary else '未知'}。",
      f"实际长条图：{counts.get('long_strip_jpg','未知')}；分步图：{counts.get('step_jpg','未知')}；掩膜：{counts.get('mask_png','未知')}；图像记录：{counts.get('records_json','未知')}。",
      f"数值响应审计退出码：{audit.get('exit_code','未知')}；已检查入口：{audit.get('checked_entry_conditions','未知')}；原B图库逐入口数值范围最大差：{audit.get('maximum_gallery_B_range_error','未知')}。",
      '数值范围一致不等于整张浮点图逐位相同；结构核验通过也不等于视觉识别正确。独立审计、逐轮视觉复核与失败记录详见experiment_report.json中带哈希的证据路径。',
      '原计划最多三轮先行调试，实际另有明确记录的修复执行；不能描述为只有三次且全部首次成功。原pilot003存在运行期间源码变化及图外弱热响应接续；repaired001增加可见卷面外缘退出保护，但仍有主路径与基线共用证据字典的问题；repaired002修复finalize_path中的字典共享，随后发现反向自由段前缀仍共享证据；repaired003再复制prepend_free_extent的前缀证据。两次字典隔离修复不用于改搜索几何或调参，其图像等价以实际独立回执为准。原输出、失败审计和所有修复补充协议均保留。',
      '实际各轮状态：'+ '；'.join(r['run']+'（退出码='+str(r.get('summary',{}).get('exit_code','未知'))+
          '，源码未变='+str(r.get('summary',{}).get('source_unchanged','未知'))+
          '，最新审计通过='+str(r.get('latest_independent_audit_passed',False))+'）' for r in report['runs']),
      '冻结最终先行版本：'+report['execution_history']['final_pilot']+'。修复前后图像是否逐字节相同，以收集的独立比较回执为准，不由修复目的推定。',
      '序列化修复图像核验实际回执：'+json.dumps([e['report'] for e in report['execution_history']['serialization_repair_image_equivalence']],ensure_ascii=False),
      '最终视觉复核证据：'+json.dumps(report['execution_history']['final_visual_review_equivalence'],ensure_ascii=False),
      '',
      '五、仍未验证的部分',
      '密集旧圈与目标活动绳的远侧同根身份、完整终点、真实接触点尚无完整真值验收。早期先行轮记录过导杆处提前转弯、工具入口误跟、亮带附近早转旧圈和弱端接续；最终哪些仍在以保存的最终视觉复核记录为准，不能因线变长或变平滑宣布解决。',
      '原图仍存在过曝、欠曝和遮挡，网络计算域边缘也可能有伪响应。拒识、局部支持、候选数及长度均不直接换算准确率。',
      '没有连接设备、改权限、读取保留集或读取S1标签；本结果不确认t0/t1/t2，不接入执行机构。',
      '',
      '六、文件位置',
    ]
    for name,path in output.get('directories',{}).items():lines.append(name+'：'+path)
    lines += ['输入数值与推理回执：'+str(HERE/'responses'),
              '输入冻结协议：'+str(HERE/'protocol_inputs.json'),
              '实验参数与范围：'+str(HERE/'postprocess_protocol.json'),
              '各轮命令、源码快照、进度及退出码：对应run_start.json、source_snapshot、progress.jsonl和summary.json。',
              '无HTML输出。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--report-name',default='experiment_report.json')
    args=parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8');tick=time.perf_counter()
    report_path=HERE/args.report_name
    assert report_path.parent.resolve()==HERE.resolve() and report_path.suffix=='.json'
    text_path=HERE/'H1算法与结果说明.txt';manifest_path=HERE/'delivery_file_hashes.json'
    for p in (report_path,text_path,manifest_path,HERE/'finalize_report_source.py',HERE/'finalize_report_receipt.json'):
        assert not p.exists(), 'refuse to overwrite '+str(p)
    protocol=load(HERE/'protocol_inputs.json');evidence=[reference(HERE/'protocol_inputs.json')]
    missing=[];run_names=discover_run_names();runs=[collect_run(name,evidence) for name in run_names]
    historical_failures=[]
    for item in runs:
        problems=[]
        if not item['summary_present'] or not item['start_present']:problems.append('missing start/summary')
        elif item['summary'].get('exit_code')!=0 or item['summary'].get('completed')!=item['summary'].get('planned'):
            problems.append('processing incomplete or failed')
        if item.get('summary',{}).get('source_unchanged') is not True:problems.append('source changed during run or not established')
        if any(not s['snapshot_matches'] for s in item.get('source_checks',[])):problems.append('source snapshot mismatch')
        if item.get('independent_audits') and not item['latest_independent_audit_passed']:problems.append('latest independent audit failed')
        item['preserved_execution_problems']=problems
        item['role']='frozen_final_pilot' if item['run']==FINAL_PILOT else ('final_delivery_run' if item['run'] in FINAL_RUNS else 'preserved_historical_trial')
        if problems:
            if item['run'] in FINAL_RUNS:missing += [item['run']+': '+p for p in problems]
            else:historical_failures.append({'run':item['run'],'problems':problems})
    final_pilot=next(r for r in runs if r['run']==FINAL_PILOT)
    if not final_pilot['latest_independent_audit_passed']:missing.append(FINAL_PILOT+': latest independent audit missing or failed')
    if any(not s['current_matches'] for s in final_pilot.get('source_checks',[])):missing.append(FINAL_PILOT+': final source mismatch')
    final=runs[-1]
    if final.get('summary',{}).get('completed')!=protocol['eligible_count']:missing.append('all001 does not complete all eligible images')
    if not final['latest_independent_audit_passed']:missing.append('all001 latest independent audit missing or failed')
    if any(not s['current_matches'] for s in final.get('source_checks',[])):missing.append('all001 current runtime source mismatch')
    numeric_path=HERE/'numeric_response_audit.json'
    numeric={'file':reference(numeric_path),'report':short_audit(load(numeric_path))} if numeric_path.exists() else {'file':None,'report':{}}
    if numeric_path.exists():evidence.append(reference(numeric_path))
    if numeric['report'].get('exit_code')!=0:missing.append('numeric response audit missing or failed')
    reviews=collect_reviews(evidence,run_names)
    if not reviews:missing.append('no saved visual review evidence')
    outputs=count_final_outputs(final,protocol,evidence)
    if outputs.get('missing_files'):missing.append('declared final output files missing')
    frozen=[]
    for source in protocol['frozen_sources']:
        current=sha(source['path']);ok=current==source['sha256']
        frozen.append(source|{'current_sha256':current,'matches':ok})
        if not ok:missing.append('upstream hash changed: '+source['path'])
    addenda=[]
    for p in sorted(HERE.glob('protocol_*repair_addendum*.json')):
        ref=reference(p);evidence.append(ref);addenda.append({'file':ref,'report':load(p)})
    for expected in ('protocol_repair_addendum.json','protocol_serialization_repair_addendum.json','protocol_serialization_repair_addendum_002.json'):
        if not any(Path(a['file']['path']).name==expected for a in addenda):missing.append('repair history missing: '+expected)
    equivalence=[];equivalent_edges=[];run_lookup={r['run']:r for r in runs}
    for run_name in run_names:
        for p in sorted((HERE/run_name).glob('*equivalence*.json')):
            value=load(p);ref=reference(p);evidence.append(ref)
            equivalence.append({'file':ref,'report':{k:v for k,v in value.items() if k!='files'}})
            old=Path(value.get('old_run','')).name;new=Path(value.get('new_run','')).name
            expected=[]
            for name in (old,new):
                s=run_lookup.get(name,{}).get('summary',{})
                expected.append(8*s['entries']+s['completed'] if 'entries' in s and 'completed' in s else None)
            complete=(value.get('exit_code')==0 and value.get('file_count',0)>0 and
                      value.get('file_count')==value.get('equal_count') and
                      expected[0]==expected[1]==value.get('file_count'))
            if complete:equivalent_edges.append((old,new))
    for old,new in (('pilot_003_repaired_001','pilot_003_repaired_002'),
                    ('pilot_003_repaired_002',FINAL_PILOT)):
        if (old,new) not in equivalent_edges and (new,old) not in equivalent_edges:
            missing.append('full rendered-image equivalence missing or failed: '+old+' -> '+new)
    # A review can be inherited ONLY through actual full-rendered-byte
    # equivalence, not by a claim that a source change should be harmless.
    reviewed_runs={name for name in run_names if any(name in Path(r['file']['path']).parts for r in reviews)}
    reachable=set(reviewed_runs);changed=True
    while changed:
        before=len(reachable)
        for old,new in equivalent_edges:
            if old in reachable:reachable.add(new)
            if new in reachable:reachable.add(old)
        changed=len(reachable)!=before
    visual_equivalence={'directly_reviewed_runs':sorted(reviewed_runs),'full_byte_equivalence_edges':equivalent_edges,
                        'final_pilot_review_supported':FINAL_PILOT in reachable,
                        'inherited_review_is_not_new_independent_accuracy_measurement':True}
    if FINAL_PILOT not in reachable:missing.append('final repaired pilot lacks direct or full-byte-equivalent visual review')
    # Missing evidence remains explicit. No write occurs before the default
    # completeness gate, so a premature call cannot create a false final file.
    if missing and not args.allow_incomplete:raise RuntimeError('Evidence not ready: '+'; '.join(missing))
    supplementary=[]
    for directory in [HERE,*[HERE/n for n in run_names]]:
        if not directory.exists():continue
        for pattern in ('*receipt*.json','*repeatability*.json','*freeze*.json','*equivalence*.json','*byte*.json','*reproducibility*.json'):
            for p in sorted(directory.glob(pattern)):
                if p.name.startswith('finalize_report'):continue
                ref=reference(p);evidence.append(ref);supplementary.append(ref)
    post=HERE/'postprocess_protocol.json'
    if post.exists():evidence.append(reference(post))
    model_environment=HERE.parent/'runs/run_20261030_002/branches/B/model/environment.json'
    training_metadata=load(model_environment) if model_environment.exists() else {}
    if model_environment.exists():evidence.append(reference(model_environment))
    report={'version':protocol['version'],'name':'热图接续 H1','created_utc':datetime.now(timezone.utc).isoformat(),
       'execution_status':'completed_artifacts_with_preserved_trial_failures_pending_user_visual_acceptance' if not missing else 'incomplete_evidence',
       'missing_or_failed_evidence':missing,
       'execution_history':{'initial_tuning_round_budget':3,'actual_pilot_runs':[r['run'] for r in runs if r['run'].startswith('pilot_')],
                            'actual_repair_executions':sum('repaired' in r['run'] for r in runs if r['summary_present']),
                            'final_pilot':FINAL_PILOT,'historical_failures_preserved':historical_failures,
                            'repair_addenda':addenda,'serialization_repair_image_equivalence':equivalence,
                            'final_visual_review_equivalence':visual_equivalence,
                            'all_pilots_first_time_success_claim':False},
       'data':{'source_count':protocol['source_count'],'eligible_images':protocol['eligible_count'],
               'entry_conditions':protocol['entry_condition_count'],'upstream_missing_entries':protocol['upstream_skipped_count'],
               'upstream_missing_is_not_absent_rope':True,'source_manifest_and_jobs':protocol['frozen_sources'],
               'old_training_source_groups':training_metadata.get('train_groups',[]),
               'input_is_open_development_not_independent_test':True,'manual_target_geometry_injected':False},
       'network':{'model_B':protocol['model_B'],'step':protocol['checkpoint_step'],
                  'semantics':protocol['response_semantics'],'new_training':False,'entries_merged':False,
                  'input_shape_hw':protocol['input_shape_hw'],'bbox_xyxy':protocol['bbox_xyxy']},
       'algorithm':{'method':'B numeric cross-profile + original bilateral evidence + bounded straight support + multi-route search + weak F3 compatibility',
                    'final_runtime_parameters':final.get('start',{}).get('parameters',{}),
                    'final_runtime_source_checks':final.get('source_checks',[]),
                    'preserved_comparator':'heat-profile scoring with same geometry/termination constraints; not ground truth',
                    'conditional_input':'same original automatically generated short entry for each B condition',
                    'mask_is_optical_width_estimate':True,'green_means_local_support_not_same_rope':True,
                    'stage03_is_DoG_diagnostic_not_binary_search_candidate_mask':True,
                    'F3_review_tails':{'record_field':'entries[].f3_tail_review','maximum_candidates':3,
                        'display':'orange dashed in stages05 and06','included_in_main_masks':False,
                        'unique_material_identity_enforced':False,'coordinate_system':'half-size crop xy'}},
       'runs':runs,'outputs':outputs,'numeric_response_audit':numeric,'visual_reviews':reviews,
       'frozen_upstream_checks':frozen,'supplementary_receipts':supplementary,
       'known_limits':[
           'dense neighboring rows can still resemble the desired same-rope continuation',
           'long bright/dark/occluded gaps and calculation-border responses remain possible error sources',
           'early pilot failures and final case-specific remaining failures are preserved in the visual review reports above',
           'no complete same-rope ground truth or independent accuracy evaluation; masks are estimated optical bands',
           'no physical contact, true terminal point, t0/t1/t2 or production-control acceptance'],
       'safety_scope':{'devices_connected':False,'permissions_changed':False,'reserved_data_decoded':False,
                       'S1_labels_read':False,'old_raw_images_models_predictions_modified':False},
       'evidence_files':list({r['path']:r for r in evidence}.values()),
       'report_command':[sys.executable,*sys.argv],'report_source_sha256':sha(__file__)}
    write_new(report_path,report)
    with text_path.open('x',encoding='utf-8') as handle:handle.write(algorithm_text(report))
    snapshot=HERE/'finalize_report_source.py';assert not snapshot.exists();shutil.copy2(__file__,snapshot)
    receipt_path=HERE/'finalize_report_receipt.json'
    write_new(receipt_path,{'command':[sys.executable,*sys.argv],'exit_code':0,
        'execution_status':report['execution_status'],'seconds':time.perf_counter()-tick,
        'python':sys.version,'platform':platform.platform(),'source_sha256':sha(__file__),
        'report':reference(report_path),'algorithm_note':reference(text_path),
        'interpretation':'report generation succeeded; only the contained run/audit evidence supports processing completion'})
    files=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.json','.txt') and p!=manifest_path]
    write_new(manifest_path,{'version':protocol['version'],'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'H1 root code/protocol/report files; actual image-output hashes remain in independent all001 audit',
        'files':[reference(p) for p in sorted(files)]})
    print(json.dumps({'report':str(report_path),'execution_status':report['execution_status'],
                      'final_output_counts':outputs.get('formal_file_counts',{}),'exit_code':0},ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        # Preserve a failed collection attempt, without overwriting evidence or
        # creating a misleading success report. Retry uses a new timestamp.
        detail=traceback.format_exc();print(detail,file=sys.stderr)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        write_new(HERE/f'finalize_report_failure_{stamp}.json',{
            'command':[sys.executable,*sys.argv],'exit_code':1,'failure':detail,'source_sha256':sha(__file__)})
        raise SystemExit(1)
