"""完成实际运行与独立审计后登记证据；不修改预测或基准。"""
from pathlib import Path
from datetime import datetime,timezone
import json
from seam_io import HERE,ROOT,F1,sources,sha,read_json,dump,baseline_unchanged

def main():
    run=HERE/'all_001'; summary=read_json(run/'summary.json'); audit=read_json(run/'audit.json')
    assert summary['completed']==400 and summary['failed']==0 and summary['source_unchanged']
    assert audit['passed']==400 and not audit['failed'] and not audit['global_errors']
    assert sources()==read_json(HERE/'freeze_all.json')['source_sha256'] and baseline_unchanged()
    if (HERE/'experiment_report.json').exists(): raise FileExistsError('delivery already registered')
    comparisons=[]
    for rel in read_json(HERE/'pilot_002/run_context.json')['selection']:
        q=Path(rel);role=q.parent.name
        for folder in ['图像结果']+['记录/masks/'+k for k in ['transition','merged','inferred','added','removed','replacement','baseline','new_join','simple_bridge','exclusion']]:
            part=Path(folder)/role/q.name
            comparisons.append({'path':part.as_posix(),'equal':sha(run/part)==sha(HERE/'pilot_002'/part)})
    assert all(c['equal'] for c in comparisons)
    dump(run/'pilot_full_equivalence.json',{'comparisons':len(comparisons),'all_equal':True,'files':comparisons})
    from collections import Counter
    rejection=Counter();accepted_sources=Counter();no_join=[]
    for item in audit['cases']:
        part=Path(item['output_file'])
        case=read_json(run/'记录/cases'/part.parent.name/(part.stem+'.json'))
        rejection.update(r for g in case['rejections'] for r in g['reasons'])
        accepted_sources.update('+'.join(j['statistics']['new_sources']) for j in case['joins'])
        if not case['joins']: no_join.append(case['record']['output_file'])
    # 明确列出所有需交付的检查文件，缺失时停止登记而不是宣称已完成。
    evidence=[HERE/'freeze_all.json',HERE/'protocol.json',HERE/'diagnosis_before.json',
      HERE/'pilot_storage_fix_equivalence.json',HERE/'pilot_visual_review_root.json',
      HERE/'pilot_002/audit.json',HERE/'pilot_002/记录/endpoint_topology_review_001.json',
      HERE/'pilot_002/记录/seam_shape_review_001.json',HERE/'pilot_002/记录/observation_support_review.json',
      HERE/'pilot_002/记录/qa_independent_diagnosis.json',
      run/'summary.json',run/'audit.json',run/'output_manifest.json',run/'audit_provenance.json',
      run/'pilot_full_equivalence.json',run/'记录/qa_additional_left.json',run/'记录/qa_additional_right.json',
      run/'记录/endpoint_topology_review_001.json',run/'记录/endpoint_topology_exceptions_001.json']
    ev={str(p.relative_to(HERE)):sha(p) for p in evidence}
    report={'version':'RB-F2.0.0','name':'重叠接续 F2','date':'2026-10-28','run':'all_001',
      'scope':'Same frozen open400 images, left200/right200. New C3 and D4 observations in common seam overlap; image-supported local visual boundary correspondence.',
      'summary':summary,'independent_structural_audit':{'passed':audit['passed'],'failed':audit['failed'],'global_errors':audit['global_errors']},
      'accepted_join_sources':dict(accepted_sources),'rejection_reason_counts_multilabel':dict(rejection),'images_without_new_joins':no_join,
      'pilot_full_exact_equivalence_count':len(comparisons),'evidence_sha256':ev,
      'endpoint_topology_review':{'bridge_count':21758,'endpoint_count':43516,'adjacent_retained_source_pixels':43451,
        'exceptions':{'old_fragment_natural_termination':64,'preexisting_source_gap_13px':1},
        'new_outer_disconnections_found':0,'meaning':'Source-raster continuity only, not true boundary correspondence accuracy.'},
      'development_history':[{'run':'pilot_001','algorithm':'final algorithm','storage_defect':'Full-size detector masks accidentally expanded into case JSON. Original output retained; visual masks unchanged.'},
         {'run':'pilot_002','change':'Only detector-mask PNG references and F1case input hash verification. All132 display/formal mask hashes identical to pilot001.'},
         {'run':'all_001','change':'Frozen unchanged code/parameters, full400, no all400 tuning.'}],
      'method':['Continuous local mean/std normalization over saved surface, same6pitch-overlap and exclusion for both detectors.',
        'Frozen B1/C3 with original weak-valley recheck and frozen D4 redetection; original codes unchanged.',
        'Combine new point candidates within curve corridors; endpoint tangent, image groove/body support, unique ordered pairing and competing candidates.',
        'Endpoint position/tangent-preserving Hermite plus small zero-end-derivative correction. No global refit outsideT.'],
      'limits':['Some joins have onlyD4 new-point support; running both does not mean both agree on every join.',
        'Local humps can arise jointly from old endpoint/tangent mismatch and point-driven correction. Image support and smoothness are not truth.',
        'Partial rejection remains in shadow, occlusion and missing original surface areas. No guarantee of all boundaries connected.',
        'No per-rope truth accuracy or physical same-rope identity/contact/event validation.'],
      'accuracy':None,'baseline_replacement':False,'equipment_access':False,'reserved_access':False,'html':False,
      'finished_utc':datetime.now(timezone.utc).isoformat()}
    dump(HERE/'experiment_report.json',report)
    text=(f"重叠接续 F2 / RB-F2.0.0\n\n最终图像：{run/'图像结果'}\n左右各200张，总计400张，4096×3000，卷面外为0。\n"
      f"实际运行400/400完成；独立结构审计400/400通过。{summary['images_with_joins']}张新增接续，合计{summary['accepted_joins']}段。"
      f"其余{len(no_join)}张没有接受新接续，保留F1结果；所有样本仍在400总分母内。连接数不是准确率。\n\n"
      "接缝使用同一归一化重叠带重新运行C3/D4，综合新黄点、原图宽沟、端切向、行序和间距进行配对及局部平滑。"
      "接缝带外原F1逐像素保持不变。黄色十字为观测，青线为图像支持拟合，紫色虚线为局部缺少支持的几何部分。\n\n"
      "仍有未接行、极暗区域缺口及少数局部拱起，ruler_left_05继续无接续。不得把平滑连接视为同根材料或真实接触点。\n"
      "原C3、D4与F1代码/结果均未改；本版独立保存，不替换已确认基准。没有设备、保留集或HTML操作。\n\n"
      "详细原理见算法说明.txt，运行/源哈希/失败/视觉复查见experiment_report.json及all_001/记录。\n")
    (HERE/'结果说明.txt').write_text(text,encoding='utf8')
    entries=[(ROOT/'数据资产/README.md','05_派生与实验结果/'),
      (ROOT/'数据资产/05_派生与实验结果/README.md',''),
      (ROOT/'工程导航/05_实验版本档案/README.md','../../数据资产/05_派生与实验结果/')]
    changes=[]
    for path,prefix in entries:
        before=sha(path);base=prefix+HERE.name
        addition=(f"\n\n## 2026-10-28：重叠接续F2 / RB-F2.0.0\n\n"
          f"[400张完整卷面边界结果]({base}/all_001/图像结果)及[实验记录]({base}/experiment_report.json)："
          f"在F1明暗交界的共同重叠区重新运行C3/D4，以新黄点、原图宽沟/绳体证据和端切向建立有序接续。"
          f"400/400运行及独立结构审计通过，{summary['images_with_joins']}张新增接续、共{summary['accepted_joins']}段；"
          f"无接续图保留原分母。接缝带外边界/推断/显示逐像素保持F1，固定12图132项pilot/full哈希一致。"
          "原C3/D4/F1源码与结果不改，F2为独立实验候选。固定12及冻结后额外左右各6图进行图像复查；"
          "仍有未接行、遮挡/极暗缺口和局部拱起。结构审计、连接数并非真值准确率或同根身份；"
          "不修改卷面掩码，不接触设备或保留集，不生成HTML。首轮记录存储膨胀问题及等价修复证据保留。\n")
        with path.open('a',encoding='utf8') as f:f.write(addition)
        changes.append({'path':str(path),'before_sha256':before,'after_sha256':sha(path)})
    dump(HERE/'delivery_receipt.json',{'date':'2026-10-28','changed_indexes':changes,
      'source_sha256':sources(),'report_sha256':sha(HERE/'experiment_report.json'),
      'output_manifest_sha256':sha(run/'output_manifest.json'),'baseline_unchanged':baseline_unchanged(),
      'new_code':['seam_io.py','seam_observation.py','seam_join.py','run_seam.py'],
      'auxiliary_files':['audit_seam.py','make_preview.py','finalize_delivery.py','算法说明.txt','结果说明.txt'],
      'original_images_unchanged':True,'no_hardware_or_reserved_access':True})
    print(json.dumps({'registered':True,'images':400,'joins':summary['accepted_joins'],'images_with_joins':summary['images_with_joins']},ensure_ascii=False))
if __name__=='__main__':main()
