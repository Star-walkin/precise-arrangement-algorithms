"""Read-only prediction audit and explicitly posthoc failure illustrations."""
from pathlib import Path
import argparse, gzip, json, sys
import cv2
import numpy as np
from PIL import Image
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE))
import evaluate as ev
import visualize as v
import postprocess_visualize as pv
from compare_evaluate import read,save,sha

CASES=[('dual_20260923_113745__left__f00450',[1480,2020,2060,2220]),
       ('dual_20260923_113305__right__f00500',[1900,1840,2290,2040]),
       ('dual_20260923_113745__left__f00550',[1170,2020,1510,2200])]


def main(run):
    summary=read(run/'evaluation/summary.json');receipt=read(run/'visualization_receipt.json')
    assert summary['formal_saved_pairs']==64 and receipt['png_count']==32
    hashes={}
    for case in summary['prediction_save_boundary']:
        for key in ['paths','response']:
            p=case['files'][key]
            if p not in hashes:hashes[p]=sha(p)
            assert hashes[p]==case['files'][key+'_sha256']
    for p in receipt['files']:
        with Image.open(p) as im:im.verify()
    out=run/'independent_qa';out.mkdir(exist_ok=False)
    records=[]
    for sid,box in CASES:
        score=read(run/'evaluation/A/P1/CONDITIONAL'/(sid+'.json'));pred=read(score['files']['paths'])
        p0=read(run/'evaluation/A/P0/CONDITIONAL'/(sid+'.json'))
        candidate=next(c for c in pred['candidates'] if c['id']==pred['selected_candidate_id'])
        points=np.asarray(candidate['points_xy']);ref=score['source_reference']['trusted_clipped_centerline']
        distance,nearest=ev._nearest(points,ref);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        bad=(distance[1:]>5)&(np.array(candidate['segment_status'])!='entry_condition')
        starts=np.flatnonzero(bad&~np.r_[False,bad[:-1]]);ends=np.flatnonzero(bad&~np.r_[bad[1:],False])+1
        bad_runs=[{'from_xy':points[a+1].tolist(),'to_xy':points[b].tolist(),'path_arclength_px':float(arc[b]-arc[a]),
                   'start_reference_distance_px':float(distance[a+1]),'end_reference_distance_px':float(distance[b])}
                  for a,b in zip(starts,ends) if arc[b]-arc[a]>=60]
        image_path=pred['image_path'];assert sha(image_path)==pred['image_sha256']
        gray=cv2.imdecode(np.fromfile(image_path,np.uint8),cv2.IMREAD_GRAYSCALE)
        response=np.load(score['files']['response'],mmap_mode='r',allow_pickle=False)
        with gzip.open(pred['segments_path'],'rt',encoding='utf-8') as handle:bundle=json.load(handle)
        board=Image.new('RGB',(2450,680),'#e6edf4')
        v.text(board,(14,10),sid+' | A / CONDITIONAL | 事后失败诊断，不用于选择参数或评分区域',22)
        v.text(board,(14,45),'原图0–255 / 原响应0–1；洋红=旧可信部分中心；绿=已选P1支持；青细线=全局片段(不代表目标身份)',18)
        p=v.panel(gray,box,'原图：无标记供核查',(600,450));panels=[p]
        p=v.panel(gray,box,'原始响应A + 保存P1路径',(600,450),response,True);pv.path_overlay(p,pred,'P1',pred['actual_seed_xy']);panels.append(p)
        p=v.panel(gray,box,'原图：P1与可信部分参考',(600,450))
        for line in ref:pv.draw_path(p,line,'#ff31cf',2,True)
        pv.path_overlay(p,pred,'P1',[]);panels.append(p)
        p=v.panel(gray,box,'全局片段可用性：非已连接正确候选',(600,450))
        for s in bundle['segments']:v.line(p,s['points_xy'],'#15c9fa',1)
        for line in ref:pv.draw_path(p,line,'#ff31cf',2,True)
        panels.append(p);pv.put(board,panels,85,600)
        enddistance=float(distance[-1]);s=score['selected']['supported_only'];old=p0['selected']['supported_only']
        def frac(x):return 'unknown' if x is None else f'{100*x:.2f}%'
        v.text(board,(14,550),f"P0→P1后段支持覆盖 {frac(old['stages']['attached']['covered_fraction'])} → {frac(s['stages']['attached']['covered_fraction'])}；P1末点距部分参考 {enddistance:.2f} 原图px",19)
        v.text(board,(14,583),'首次持续参考漏段：'+str(s['first_sustained_deviation_proxy']['first_sustained_deviation']),15)
        v.text(board,(14,618),'片段可用性只说明原热图中存在局部近参考线；不证明这些线可从入口正确连接，也不提供完整末端身份。',18)
        file=out/(sid+'.png');board.save(file)
        records.append({'sample_id':sid,'mode':'CONDITIONAL','branch':'A','posthoc_zoom':box,'figure':str(file),
            'P0_supported_attached':old['stages']['attached'],'P1_supported_attached':s['stages']['attached'],
            'first_sustained_reference_gap':s['first_sustained_deviation_proxy'],
            'P1_end_xy':points[-1].tolist(),'nearest_partial_reference_xy':nearest[-1].tolist(),
            'P1_end_reference_distance_px':enddistance,'predicted_path_continuous_deviation_runs_over60px':bad_runs,
            'selected_fragment_ids':candidate['segment_ids'],'bridges':candidate['bridges'],
            'fragment_availability':score['all_extracted_fragment_availability'],
            'termination':candidate['termination'],'source_prediction_sha256':score['files']['paths_sha256']})
    save(out/'case_evidence.json',{'status':'completed','cases':records,'posthoc_only':True,'no_prediction_or_parameter_changes':True})
    save(run/'independent_qa_summary.json',{'status':'completed','command':sys.argv,'source_sha256':sha(__file__),
        'evaluation_summary_sha256':sha(run/'evaluation/summary.json'),'formal_prediction_pairs_rehashed':64,
        'distinct_source_float_files_rehashed':32,'visualization_pngs_decode_verified':32,
        'dash_contract':read(run/'visualization_dash_contract.json'),
        'case_evidence':str(out/'case_evidence.json'),'figures':[r['figure'] for r in records],
        'findings':[
            'A/P1: clearer continuous free/transition coverage does not extend to complete target; supported attached and whole-reference coverage regress versus P0.',
            'All P1 combinations have zero scored edge-near coverage; no verified target endpoint success exists.',
            'A left450 continues approximately371px after departing the reviewed center by>5px, ending39.33px away; not merely an early stop. Departure begins inside s2 before its4.47px bridge to s9.',
            'A right500 stops near the reference at(2034.5,1922.5), leaving about1054px uncovered; substantial global near-reference fragments are not an entry-connected correct path.',
            'A left550 stops early at(1322.5,2098.5); global fragment availability exceeds final path coverage but does not establish which route is correct.',
            'Global fragment proximity and final-pool coverage are posthoc geometry diagnostics, not correct-identity candidates or evidence for a trained relation model.',
            'Bridge coverage is not detected rope; allgeometry-minus-supported also includes ambiguous segments and cannot be attributed exclusively to bridges.'
        ],'original_images_predictions_models_parameters_unchanged':True,'complete_identity_accuracy':None})
    print(json.dumps({'status':'completed','formal_pairs':64,'figures':len(records)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run.resolve())
