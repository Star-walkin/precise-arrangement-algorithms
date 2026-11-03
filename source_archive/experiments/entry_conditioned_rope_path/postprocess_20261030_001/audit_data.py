"""Source-only audit and entry diagnostics for frozen AB750 postprocessing.

Does not read evaluation response values, modify old data, or choose parameters
from evaluation results. The specified example remains pair0075 by identity.
"""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import cv2
import numpy as np

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
ROOT=BASE.parents[1]
DATA=ROOT/'数据资产/05_派生与实验结果/camera_condition_images_20261009_v2'
EXAMPLE='dual_20260923_101517__pair0075__right'
CALIBRATION_IDS=['dual_20260924_111500__left__f00240','dual_20260924_111143__right__f00130',
 'C02__dual_20260923_101517__pair0550__right','C07__R_run_stop_run_20260922_01__t012.033__f000368']

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def save(p,a):cv2.imencode('.png',a)[1].tofile(str(p))

def main():
    diag=OUT/'input_diagnostics';diag.mkdir(parents=True,exist_ok=True)
    manifest=DATA/'manifest.csv';rows=list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
    row=next(r for r in rows if Path(r['output_file']).stem==EXAMPLE)
    assert row['dataset_use_status']=='unlabeled_open_development_no_new_test_split' and row['historical_use']=='old_train'
    assert '20260925' not in row['source_file']+row['output_file']
    jobs_path=BASE/'batch_20261030_001/jobs.json';jobs=load(jobs_path)
    job=next(j for j in jobs['jobs'] if j['id']==EXAMPLE);image_path=DATA/row['output_file']
    assert Path(job['image_path']).resolve()==image_path.resolve() and sha(image_path)==row['output_sha256']==job['image_sha256']
    source=Path(job['seeds'][0]['source']);a6=load(source);entries={e['entry_id']:e for e in a6['result']['entries']}
    assert len(job['seeds'])==2 and len(entries)==2
    im=cv2.imdecode(np.fromfile(image_path,np.uint8),cv2.IMREAD_COLOR);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    overlay=im.copy();cv2.line(overlay,(0,1050),(4095,1050),(170,100,100),2)
    details=[];new_float_records=[]
    roles={'recovered_01':'handheld rigid tool-like object; not the visible incoming wire-rope arc',
           'recovered_00':'right-hand visible free wire rope; later target identity remains unconfirmed'}
    colors=[(0,165,255),(255,255,0)]
    for k,s in enumerate(job['seeds']):
        p=np.asarray(s['xy'],float);e=entries[s['entry_id']];g=e['free']['geometry']
        model=(p-[0,1050])*.5;length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
        assert length<=120.000001 and np.all((model[:,0]>=0)&(model[:,0]<2048)&(model[:,1]>=0)&(model[:,1]<975))
        cv2.polylines(overlay,[np.rint(p).astype(np.int32)],False,colors[k],7)
        center=p.mean(0);tx=int(np.clip(center[0]-350,20,3300));ty=int(np.clip(center[1]-80,80,2940))
        cv2.putText(overlay,f'{k+1}: {s["entry_id"]}',(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,1.6,colors[k],3,cv2.LINE_AA)
        x0=max(0,int(center[0]-250));x1=min(4096,int(center[0]+250));y0=max(0,int(center[1]-170));y1=min(3000,int(center[1]+170))
        raw=im[y0:y1,x0:x1].copy();marked=raw.copy()
        cv2.polylines(marked,[np.rint(p-[x0,y0]).astype(np.int32)],False,colors[k],3)
        save(diag/f'{s["entry_id"]}__raw_and_seed.png',np.concatenate([raw,marked],1))
        d=p[-1]-p[0];d/=np.linalg.norm(d);n=np.array([-d[1],d[0]])
        centers=p[0]+np.linspace(.1,.9,9)[:,None]*(p[-1]-p[0]);offs=np.linspace(-60,60,241)
        q=centers[:,None,:]+offs[None,:,None]*n[None,None,:]
        valid=(q[:,:,0]>=0)&(q[:,:,0]<4096)&(q[:,:,1]>=0)&(q[:,:,1]<3000)
        values=cv2.remap(gray,q[:,:,0].astype(np.float32),q[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        mean=np.divide((values*valid).sum(0),valid.sum(0),out=np.zeros(len(offs),np.float32),where=valid.sum(0)>0)
        plot=np.full((360,760,3),245,np.uint8);xx=60+(offs+60)/120*650;yy=295-mean*230
        for v in [0,.25,.5,.75,1]:
            y=int(295-v*230);cv2.line(plot,(60,y),(710,y),(210,210,210),1);cv2.putText(plot,str(v),(9,y+5),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,40,40),1)
        cv2.line(plot,(385,65),(385,295),(150,150,150),1)
        cv2.polylines(plot,[np.column_stack([xx,yy]).astype(np.int32)],False,colors[k],2)
        cv2.putText(plot,f'Original-gray normal profiles: {s["entry_id"]}',(16,27),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1,cv2.LINE_AA)
        cv2.putText(plot,'Average9positions in saved120px seed; no model response used',(16,51),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1,cv2.LINE_AA)
        for v in [-60,-30,0,30,60]:cv2.putText(plot,str(v),(int(60+(v+60)/120*650)-10,322),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,30,30),1)
        cv2.putText(plot,'normal offset: original pixels; intensity fixed0..1',(140,349),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
        save(diag/f'{s["entry_id"]}__gray_profile.png',plot)
        new_dir=OUT/'batch/conditions'/EXAMPLE/f'entry_{k+1:02d}'
        response_profiles={}
        for variant,color in [('A',(210,60,20)),('B',(40,30,210))]:
            rp=new_dir/variant/'response.npy';mp=new_dir/variant/'inference.json'
            if not (rp.exists() and mp.exists()):continue
            meta=load(mp);before=sha(rp)
            assert before==meta['response_sha256'] and meta['id']==EXAMPLE and meta['entry_xy']==s['xy'] and meta['step']==750
            response=np.load(rp,mmap_mode='r');assert response.shape==(3000,4096) and response.dtype==np.float32
            assert np.isnan(response[:1050]).all() and np.isfinite(response[1050:]).all()
            sampled=cv2.remap(response,q[:,:,0].astype(np.float32),q[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            curve=np.divide((np.nan_to_num(sampled)*valid).sum(0),valid.sum(0),out=np.zeros(len(offs),np.float32),where=valid.sum(0)>0)
            response_profiles[variant]={'mean_response_normal_profile':curve.tolist(),
              'mean_at_saved_seed_center':float(curve[np.argmin(np.abs(offs))]),
              'profile_peak_offset_original_px':float(offs[np.argmax(curve)]),
              'peak_offset_is_true_localization_error':False}
            cv2.polylines(plot,[np.column_stack([xx,295-curve*230]).astype(np.int32)],False,color,2)
            assert sha(rp)==before
            new_float_records.append({'entry_number':k+1,'variant':variant,'path':str(rp),'sha256':before,
              'checkpoint_sha256':meta['checkpoint']['sha256'],'float32_original_shape_verified':True,'outside_domain_nan_verified':True,'source_unchanged':True})
        if response_profiles:
            plot[:60]=245
            cv2.putText(plot,f'Saved entry normal profiles: {s["entry_id"]}',(16,25),cv2.FONT_HERSHEY_SIMPLEX,.58,(20,20,20),1,cv2.LINE_AA)
            cv2.putText(plot,'gray-source orange/cyan | A blue | B red; all fixed0..1',(16,49),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1,cv2.LINE_AA)
            save(diag/f'{s["entry_id"]}__gray_A_B_normal_profiles.png',plot)
        details.append({'entry_number':k+1,'entry_id':s['entry_id'],'ambiguous':s['ambiguous'],'saved_seed_xy_original':p.tolist(),
          'seed_xy_model':model.tolist(),'seed_length_original_px':length,'a6_observed_segment_xy':g['observed_segment_xy'],
          'a6_direction_xy':g['direction_xy'],'a6_optical_width_px':g['width_px'],'width_is_calibrated_rope_diameter':False,
          'original_image_visual_diagnosis':roles[s['entry_id']],'diagnosis_is_independent_ground_truth':False,
          'source':str(source),'source_sha256':sha(source),'profile_normal_offsets_px':offs.tolist(),'profile_gray_mean':mean.tolist(),
          'crop_xyxy':[x0,y0,x1,y1],'physical_contact_claim':False,'complete_terminal_identity':'unknown',
          'response_profiles_specified_development_example_only':response_profiles})
    cv2.putText(overlay,'Saved entry candidates only; no forced identity selection or terminal continuation',(30,70),cv2.FONT_HERSHEY_SIMPLEX,1.4,(255,255,255),3,cv2.LINE_AA)
    save(diag/'specified_example_original_and_entries.png',cv2.resize(overlay,(2048,1500),interpolation=cv2.INTER_AREA))
    ab=BASE/'runs/run_20261030_002';paired=load(ab/'paired_comparison.json');assert paired['main_comparison_step']==750
    index=load(ab/'data/index.json');train={i['id']:i for i in index['items'] if i['split']=='train'}
    calibration=[]
    for sid in CALIBRATION_IDS:
        i=train[sid];calibration.append({'id':sid,'source_group':i['source_group'],'split':'train','image_path':i['image_path'],
          'image_sha256':i['image_sha256'],'seed_mode':'CONDITIONAL existing entry only',
          'seed_xy_original':i['known_seed_xy'],'data_cache':i['cache'],
          'response_paths':{key:str(ab/'branches'/key/'predictions/CONDITIONAL'/sid/'response.npy') for key in ['A','B']},
          'label_scope':'existing reviewed partial center; C07 limited to C7; no complete endpoint identity',
          'selection_basis':'one representative per training source group, before reading new postprocessing outcomes'})
    inventory=[]
    for key in ['A','B']:
        branch=ab/'branches'/key;summary=load(branch/'inference_final_summary.json')
        expected=paired['main_checkpoints'][key];assert sha(expected['path'])==expected['sha256']==summary['model_sha256']
        # Inspect file inventory/metadata only, never evaluation response arrays.
        records=[]
        for i in index['items']:
            for mode in ['CONDITIONAL']+(['AUTO'] if i['split']=='eval' else []):
                p=branch/'predictions'/mode/i['id']/'response.npy'
                records.append({'id':i['id'],'split':i['split'],'mode':mode,'path':str(p),'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None})
        inventory.append({'version':key,'checkpoint':expected,'response_records':records,'count_existing':sum(r['exists'] for r in records),
          'content_read_for_parameter_selection':False,'binding_verified_from_inference_metadata':True})
    result={'command':sys.argv,'exit_code':0,'source_sha256':sha(__file__),'specified_example':{'id':EXAMPLE,'manifest_row':row,
      'image_path':str(image_path),'image_sha256':sha(image_path),'identity_rule':'preserve pair0075 exact id; do not rename or substitute an f000075 image',
      'a6_status':a6['result']['status'],'entries':details,'existing_batch_float_available':False,
      'inference_needed':'no old batch floats; new frozen A/B750 re-forward completed' if len(new_float_records)==4 else 're-run frozen A/B750 on exact original PNG and each saved short entry; do not invert JPG colors',
      'newly_recomputed_float_records':new_float_records},
      'manifest':{'path':str(manifest),'sha256':sha(manifest),'rows':len(rows)},
      'model_coordinate_mapping':{'original_shape_hw':[3000,4096],'search_bbox_xyxy':[0,1050,4096,3000],
        'model_shape_hw':[975,2048],'scale':.5,'nominal_map':'model_xy=(original_xy-[0,1050])/2',
        'image_preprocessing':'original grayscale/255, INTER_AREA half-scale; same frozen ROI normalization',
        'seed_gaussian_sigma_original_px':4,'outside_search_roi':'NaN in original float response; never supervised background'},
      'reusable_existing_floats':inventory,'calibration_samples':calibration,'calibration_groups':4,
      'specified_example_use':'training-source open-development display/diagnostic; not independent evaluation and not a new label',
      'evaluation_images_or_prediction_values_read_for_selection':False,'reserved_data_accessed':False,
      'limitations':['a6 recovered_01 follows a visible handheld tool-like object; continuity alone cannot repair wrong entry identity',
        'right visible free-rope entry does not establish downstream or terminal identity',
        'saved batch JPGs/color LUT cannot reconstruct original unquantized response floats']}
    dump(OUT/'data_audit.json',result)
    dump(OUT/'calibration_samples.json',{'calibration_samples':calibration,'specified_display_example_id':EXAMPLE,'selection_uses_evaluation':False})
    print(json.dumps({'example':EXAMPLE,'saved_entries':len(details),'calibration_images':4,'calibration_groups':4,
      'existing_floats_per_AB':[r['count_existing'] for r in inventory],'example_requires_frozen_inference':len(new_float_records)!=4,'exit_code':0}))

if __name__=='__main__':main()
