"""只在全量运行和独立审计完成后登记F3结果；不修改预测。"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json
from surface_io import HERE,ROOT,sha,read_json,dump,sources,baseline_unchanged

def main():
    run=HERE/'all_001';s=read_json(run/'summary.json');a=read_json(run/'audit.json')
    assert (s['planned'],s['completed'],s['failed'],s['exit_code'])==(400,400,0,0)
    assert a['passed']==400 and a['failed']==0 and not a['global_errors']
    assert sources()==read_json(HERE/'freeze_all.json')['source_sha256'] and baseline_unchanged()
    assert not (HERE/'experiment_report.json').exists()
    output_counts={}
    for stage in ('01_D4全卷面候选','02_F3融合边界'):
        output_counts[stage]={role:len(list((run/'图像结果'/stage/role).glob('*.png'))) for role in ('left_camera','right_camera')}
        assert output_counts[stage]=={'left_camera':200,'right_camera':200}
    eq=[]
    folders=['图像结果/01_D4全卷面候选','图像结果/02_F3融合边界']+['记录/masks/'+k for k in ('merged','inferred','new','removed','transition','full_d4','full_d4_inferred','exclusion','full_d4_observed')]
    for rel in read_json(HERE/'pilot_002/run_context.json')['selection']:
        p=Path(rel)
        for folder in folders:
            q=Path(folder)/p.parent.name/p.name
            same=sha(run/q)==sha(HERE/'pilot_002'/q);assert same
            eq.append({'path':q.as_posix(),'equal':same,'sha256':sha(run/q)})
    dump(run/'pilot_full_equivalence.json',{'file_comparisons':len(eq),'all_equal':True,'files':eq})
    rows=[json.loads(x) for x in (run/'run_log.jsonl').read_text(encoding='utf8').splitlines()]
    modes=Counter();reasons=Counter();groups=0
    for r in rows:
        d=r['fusion_diagnostics'];modes.update(d.get('accepted_modes',{}));reasons.update(d.get('rejection_counts',{}));groups+=d.get('accepted_groups',0)
    evidence=['protocol.json','freeze_all.json','diagnosis_before.json','diagnosis_before_statistics.json',
      'pilot_001/audit_corrected.json','pilot_001/记录/qa_independent_gap_review.json',
      'pilot_002/audit.json','pilot_002/记录/endpoint_shape_review_002.json',
      'pilot_002/记录/qa_surface_visual_review.json','pilot_002/记录/qa_independent_gap_review.json',
      'all_001/summary.json','all_001/audit.json','all_001/audit_provenance.json','all_001/output_manifest.json',
      'all_001/pilot_full_equivalence.json','all_001/记录/qa_additional_left.json','all_001/记录/qa_additional_right.json',
      'all_001/记录/endpoint_shape_review_all.json']
    ev={p:sha(HERE/p) for p in evidence}
    report={'version':'RB-F3.0.0','name':'全卷面协同 F3','date':'2026-10-28','run':'all_001',
      'summary':s,'output_image_counts':output_counts,'accepted_fragment_count':groups,'accepted_fragment_modes':dict(modes),
      'endpoint_raster_shape_review':read_json(run/'记录/endpoint_shape_review_all.json')['totals'],
      'rejection_reason_counts':dict(reasons),'images_without_additions':[r['output_file'] for r in rows if not r['pixels']['new']],
      'zero_addition_diagnostics':[{'image':r['output_file'],'full_surface_points':r['new_full_surface_points'],
        'full_surface_curves':r['full_surface_curves'],'fusion':r['fusion_diagnostics']} for r in rows if not r['pixels']['new']],
      'independent_structural_audit':{'passed':a['passed'],'failed':a['failed'],'global_errors':a['global_errors']},
      'pilot_full_exact_image_mask_comparisons':len(eq),'evidence_sha256':ev,
      'diagnosis':'F2 discards new observations outside accepted old-two-endpoint bridges; fresh fitted D4 fragments are not independent outputs. Fixed12 has45796 new points but12534 not assigned to accepted bridges. This is not primarily a uniform brightness-threshold problem.',
      'method':['Actual unchanged D4 detection/grouping/fit on full saved surface S with continuous normalization and old+new exclusions.',
        'Compare new full D4 paths with old C3/F2 image-row evidence, preserving all old F2 boundaries.',
        'Accept image-supported missing fragments inside frozen F2 T with one-side/reference support, or stricter independent-neighbor checks.',
        'Use bounded smooth endpoint correction; reject unsafe attachment while retaining supported fragment. Actual source observations only.'],
      'history':[{'run':'pilot_001','status':'Prediction completed; short direct snapping produced small vertical end steps. Preserved.',
        'audit_note':'Initial audit Counter(dict) programming error retained as audit.json; corrected keys-only audit_corrected.json passed without prediction change.'},
        {'run':'pilot_002','status':'Only endpoint correction changed; detector output107426 points/1290 curves unchanged.226 accepted fragments retained,11 ends detached instead of unsafe short snapping.'},
        {'run':'all_001','status':'Same400 after source freeze; no full400 tuning.'}],
      'limitations':['No truth-labeled accuracy or physical material identity/contact validation.',
        'Existing C3/D4/F2 offsets and some gaps remain. Full-surface candidates are not all final accepted boundaries.',
        'Old SAM omissions, active-rope exclusion and low-information areas remain; no invented yellow points to fill them.',
        'Full D4 sees entireS, but new final additions are intentionally limited to F2 transitionT; no blanket replacement of bright C3.'],
      'accuracy':None,'baseline_replacement':False,'equipment_access':False,'reserved_access':False,'html':False,
      'finished_utc':datetime.now(timezone.utc).isoformat()}
    dump(HERE/'experiment_report.json',report)
    text=(f"全卷面协同 F3 / RB-F3.0.0\n\n最终融合结果：{run/'图像结果/02_F3融合边界'}\n"
      f"完整D4候选：{run/'图像结果/01_D4全卷面候选'}\n两组各400张，左右各200，4096×3000，卷面外0。\n\n"
      f"实际400/400运行完成、独立结构审计400/400通过；{s['images_with_additions']}张增加边界片段，"
      f"新增{groups}个片段、{s['new_accepted_points']}个接受观测点、{s['added_pixels']}个边界像素。数量不是准确率。\n"
      "F2没有黄点的原因主要包括：需要两侧旧端点、整桥被拒后局部有用点不显示、新D4拟合段没有独立输出。"
      "F3让D4实际跑完整卷面，再用C3/F2视觉行和原图宽沟、绳体、邻距检查接纳新段；端部使用有界平滑修正。\n\n"
      "候选图展示全部合格新黄点，不代表均被融合采用。正式融合图优先供检查；黄色十字为真观测，青线为有图像支持拟合，紫虚线为短范围推断。\n"
      "仍有未接段、旧局部偏位与SAM漏面；遮挡区继续保留。C3/D4/F1/F2源码及输出不改，没有设备或保留集访问，无HTML。\n"
      "详情见算法说明.txt、experiment_report.json及各run记录。\n")
    (HERE/'结果说明.txt').write_text(text,encoding='utf8')
    changes=[]
    for path,prefix in [(ROOT/'数据资产/README.md','05_派生与实验结果/'),(ROOT/'数据资产/05_派生与实验结果/README.md',''),(ROOT/'工程导航/05_实验版本档案/README.md','../../数据资产/05_派生与实验结果/')]:
        before=sha(path);base=prefix+HERE.name
        entry=(f"\n\n## 2026-10-28：全卷面协同F3 / RB-F3.0.0\n\n"
          f"[400张融合边界]({base}/all_001/图像结果/02_F3融合边界)、[400张全卷面D4候选]({base}/all_001/图像结果/01_D4全卷面候选)及[实验记录]({base}/experiment_report.json)。"
          "针对F2有新点却因旧两端/整桥限制未输出的问题，在完整已有卷面上实际重跑原D4，以C3/F2已有视觉行、原图宽沟/绳体和邻距筛选补入接缝空缺；允许单侧及严格独立片段。"
          f"固定12两轮后冻结43项源码，400/400运行及结构审计通过；{s['images_with_additions']}张增加片段，共{groups}段、{s['new_accepted_points']}个新接受点。"
          "端部竖段问题已在第二轮用有界平滑修正，不能安全短接时保留断端；失败和审计程序错误修正证据保留。"
          "既有F2边界和T外显示/边界保持不变，原C3/D4/F1/F2基准不改。固定6及冻结后额外12图实看仍有漏段、局部偏位和掩码缺失；"
          "点数、连接数量和结构审计不是逐根准确率或同根身份。无设备/保留集访问，无HTML。\n")
        with path.open('a',encoding='utf8') as f:f.write(entry)
        changes.append({'path':str(path),'before_sha256':before,'after_sha256':sha(path)})
    dump(HERE/'delivery_receipt.json',{'version':'RB-F3.0.0','changed_indexes':changes,'source_sha256':sources(),
      'report_sha256':sha(HERE/'experiment_report.json'),'output_manifest_sha256':sha(run/'output_manifest.json'),
      'baseline_unchanged':baseline_unchanged(),'original_images_unchanged':True,
      'new_algorithm_files':['surface_io.py','surface_observation.py','surface_fusion.py','run_surface.py'],
      'auxiliary_files':['audit_surface.py','make_surface_preview.py','finalize_delivery.py','算法说明.txt','结果说明.txt'],
      'equipment_access':False,'reserved_access':False})
    print(json.dumps({'registered':True,'final_images':400,'candidate_images':400,'new_fragments':groups,'new_points':s['new_accepted_points']},ensure_ascii=False))
if __name__=='__main__':main()
