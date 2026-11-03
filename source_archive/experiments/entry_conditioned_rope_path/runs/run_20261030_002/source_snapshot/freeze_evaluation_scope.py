"""Freeze image-only evaluation regions before any new A/B predictions exist.

The two neighbor strips are engineering visual proxies, not new human truth.
No response values, prior candidate rankings, or training images are read.
"""
from pathlib import Path
import argparse, hashlib, json, sys
from datetime import datetime, timezone
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from vision_geometry_lab.experiments.v28_local_spool_connection import data_eval as d
from vision_geometry_lab.centerline_path_metrics_v6 import clip_scoring_segments

PAIR_IDS={'dual_20260923_113745__left__f00450','dual_20260923_113305__right__f00500'}
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def interpolate(line,fraction,length=180.):
    p=np.asarray(line,float);ar=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    center=fraction*ar[-1];s=np.linspace(max(0,center-length/2),min(ar[-1],center+length/2),31)
    return np.column_stack([np.interp(s,ar,p[:,k]) for k in [0,1]])
def rect(x0,y0,x1,y1):return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
def box(point,shape,size=480):
    h,w=shape;x,y=point;x0=max(0,min(w-size,int(x)-size//2));y0=max(0,min(h-size,int(y)-size//2));return [x0,y0,x0+size,y0+size]
def freeze(run,old):
    run,old=Path(run).resolve(),Path(old).resolve();folder=run/'evaluation_scope';folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'scope.json').exists(),'Never overwrite frozen scope'
    for branch in ['A','B']:
        assert not list((run/'branches'/branch/'predictions').glob('*/*/response.npy')),'Freeze must precede A/B predictions'
    index={i['id']:i for i in read(old/'data/index.json')['items']};rows=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for item in d.list_inputs(verify_images=True):
        sid=item['sample_id'];reference=d.load_evaluation_reference(sid);scene_file=old/'data/scenes'/f'{sid}.json';scene=read(scene_file)
        if '113745' in sid:
            polygons=[rect(1050,1400,1980,1750),[[2160,1690],[2740,1390],[2800,1470],[2200,1800]]]
        elif '113305' in sid:
            polygons=[rect(1700,1470,2630,1680),[[1000,1410],[1500,1630],[1440,1700],[970,1500]]]
        else:
            polygons=[rect(2120,1470,3000,1770),[[1300,1500],[1940,1760],[1870,1850],[1260,1580]]]
        backgrounds=[{'id':name,'polygon_xy':poly,'type':'background','source':'original_image_engineering_visual_proxy',
          'acceptance_reason':'clearly visible control enclosure/buttons or structural metal support, spatially separated from actual target and spool; selected without response values',
          'uncertainty':'not independent human segmentation truth; applies only inside this fixed polygon'} for name,poly in zip(['control_box','metal_support'],polygons)]
        pair=[]
        if sid in PAIR_IDS:
            target=interpolate(reference['centerline'],.65);tangent=np.gradient(target,axis=0)
            normal=np.column_stack([-tangent[:,1],tangent[:,0]]);normal/=np.linalg.norm(normal,axis=1)[:,None]
            normal*=np.where(normal[:,1:2]>=0,1,-1);neighbor=target+20*normal
            sections=d._sections(reference,[target.tolist()],[{'id':'neighbor','points_xy':neighbor.tolist()}])
            usable=[s for s in sections if s['paired_edges_valid']]
            outside=[s for s in usable if s['candidate_offsets_px']['neighbor'] is not None and
                     (s['candidate_offsets_px']['neighbor']<min(s['edge_offsets_px'])-2 or s['candidate_offsets_px']['neighbor']>max(s['edge_offsets_px'])+2)]
            assert len(usable)>0 and len(outside)==len(usable)
            pair=[{'id':'target_neighbor_01','target_points_xy':target.tolist(),'neighbor_points_xy':neighbor.tolist(),
              'target_source':'old reviewed quality-valid centerline, middle65percent arclength +/-90px',
              'neighbor_source':'original_image_engineering_visual_proxy',
              'acceptance_reason':'original-resolution crop shows a separate neighboring rope band at these points, not only a dark groove; every valid reviewed target cross-section places it >2px outside target sides',
              'paired_sections':len(usable),'outside_reviewed_target_sections':len(outside),
              'construction':'local positive-y normal offset20px only at this inspected180px segment; not a global spacing or calibrated rope diameter',
              'uncertainty':'engineering adjacent-band proxy, not independently confirmed full path identity; Delta is proxy contrast only'}]
        segments=clip_scoring_segments([reference['centerline']],ignore_polygons=reference['ignore_polygons'])
        free=interpolate(reference['centerline'],.13,40).mean(0);attached=interpolate(reference['centerline'],.65,40).mean(0)
        contact=scene['contact_proxy'].get('xy')
        if contact is None:contact=interpolate(reference['centerline'],.35,40).mean(0).tolist()
        b=np.asarray(polygons[0]).mean(0)
        zooms=[{'id':'free','bbox':box(free,item['image_shape_hw']),'basis':'fixed old reviewed reference arclength13percent'},
          {'id':'contact_proxy','bbox':box(contact,item['image_shape_hw']),'basis':'old frozen scene; if unknown fixed reference35percent context, not contact truth'},
          {'id':'attached_visual','bbox':box(attached,item['image_shape_hw']),'basis':'fixed reviewed reference65percent; visual stage only'},
          {'id':'background','bbox':box(b,item['image_shape_hw']),'basis':'fixed control enclosure polygon center'}]
        row={**item,'common_scene':str(scene_file),'common_scene_sha256':sha(scene_file),'reviewed_reference':reference,
          'quality_clipped_center_segments_xy':segments,'background_regions':backgrounds,'target_neighbor_pairs':pair,
          'neighbor_identity_status':'engineering_proxy_only' if pair else 'unknown_no_accepted_neighbor_reference',
          'zooms':zooms,'seed_exclusion':'actual supplied short seed5px capsule and entry_condition path edges; unchanged old evaluator',
          'inference_target_reference_access':'evaluation only; never used by train, entry or decoder'}
        rows.append(row)
        raw=cv2.imdecode(np.fromfile(item['image_path'],np.uint8),cv2.IMREAD_GRAYSCALE)
        im=Image.fromarray(raw).convert('RGB').resize((1024,750));dr=ImageDraw.Draw(im)
        for region in backgrounds:dr.polygon([tuple(np.array(p)/4) for p in region['polygon_xy']],outline='#ff595e',width=2)
        for seg in segments:dr.line([tuple(np.array(p)/4) for p in seg],fill='yellow',width=1)
        for p in pair:dr.line([tuple(np.array(x)/4) for x in p['neighbor_points_xy']],fill='cyan',width=2)
        dr.text((10,10),sid,font=font,fill='white');dr.text((10,720),'red background proxy | yellow reviewed target | cyan neighbor proxy',font=font,fill='white')
        im.save(folder/(sid+'__frozen_regions.png'))
    payload={'schema_version':1,'frozen_utc':datetime.now(timezone.utc).isoformat(),'new_run':str(run),'old_run':str(old),
      'new_A_B_prediction_artifacts_present':False,'coordinate_selection_used_new_or_old_response_values':False,
      'selection_sources':'original pixels, reviewed partial target geometry and preexisting frozen scenes only',
      'evaluation_only':True,'train_images_read':False,'population_images':8,'source_groups':3,
      'background_regions':16,'target_neighbor_pairs':2,'neighbor_pair_source_groups':2,
      'neighbor_metric_status':'engineering proxy contrast only, never independent identity accuracy',
      'complete_terminal_identity_reference':False,'geometry_match_tolerance_px':5.,
      'first_sustained_deviation_proxy':{'reference_distance_threshold_px':5.,'minimum_contiguous_gap_length_px':60.,'definition':'earliest trusted reference arclength gap after input exclusion, not physical identity first error'},
      'response_statistics':{'background':['mean','p90','p95','fraction_gt0.5'],'target':'arclength-weighted center samples, four frozen visual stages','Delta':'target mean minus neighbor mean, at two frozen proxy pairs'},
      'fixed_heatmap_scale':[0,1],'fixed_grayscale_scale':[0,255],
      'scope_code_sha256':sha(__file__),'items':rows,'command':sys.argv,'exit_code':0}
    save(folder/'scope.json',payload)
    save(folder/'freeze_receipt.json',{'scope_sha256':sha(folder/'scope.json'),'frozen_utc':payload['frozen_utc'],'images':8,'groups':3,'background_regions':16,'proxy_neighbor_pairs':2,'code_sha256':sha(__file__),'exit_code':0})
    print(json.dumps(read(folder/'freeze_receipt.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--old',required=True);a=p.parse_args();freeze(a.run,a.old)
