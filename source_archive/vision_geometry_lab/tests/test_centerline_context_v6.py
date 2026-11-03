from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT=Path(__file__).resolve().parents[2]
TRAIN_PY=ROOT/'vision_training/.venv/Scripts/python.exe'
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def _torch_available():
    if not TRAIN_PY.is_file():return False
    try:
        result=subprocess.run([str(TRAIN_PY),'-c','import torch'],cwd=ROOT,
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                              timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return result.returncode==0
    except (OSError,subprocess.TimeoutExpired):return False


def _run_training_python(arguments,timeout=180):
    environment=dict(os.environ);environment['PYTHONUTF8']='1';environment['CUDA_VISIBLE_DEVICES']=''
    environment['OMP_NUM_THREADS']='1';environment['MKL_NUM_THREADS']='1'
    result=subprocess.run([str(TRAIN_PY),*arguments],cwd=ROOT,env=environment,
                          text=True,capture_output=True,timeout=timeout,
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:
        raise AssertionError(f'command failed ({result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}')
    return result


def _add_sample(folder,samples,sample_id,group_id,split,with_quality=False):
    folder=Path(folder);image=(folder/f'{sample_id}.png').resolve()
    height,width=80,160
    yy,xx=np.mgrid[:height,:width]
    pixels=np.stack([(70+xx//3)%255,(90+yy*2)%255,np.full_like(xx,120)],axis=2).astype(np.uint8)
    pixels[36:45,8:152]=np.array([190,175,145],np.uint8)
    Image.fromarray(pixels,'RGB').save(image)
    image_hash=hashlib.sha256(image.read_bytes()).hexdigest()
    annotation=(folder/f'{sample_id}.json').resolve()
    label={'schema_version':1,'sample_id':sample_id,'image_sha256':image_hash,
           'coordinate_system':'full_image_xy','status':'partial','reviewed':True,
           'polylines':{'centerline':[[12,40],[80,40],[145,44]],
                        'edge_flange':[[12,36],[80,36],[145,40]],
                        'edge_surface':[[12,44],[80,44],[145,48]],
                        'free_seed':[[12,40],[30,40]]},
           'ignore_polygons':[],'notes':'V6 deterministic fixture'}
    annotation.write_text(json.dumps(label),'utf-8')
    sample={'sample_id':sample_id,'group_id':group_id,'split':split,
            'image_path':str(image),'image_sha256':image_hash,
            'image_shape_hw':[height,width],'roi_xyxy':[8,8,152,72],
            'annotation_path':str(annotation),
            'source':{'video_path':f'{group_id}.avi','frame_index':1,
                      'time_s':.1,'camera':'fixture'}}
    if with_quality:
        quality={'schema_version':1,'sample_id':sample_id,'image_sha256':image_hash,
                 'annotation_sha256':hashlib.sha256(annotation.read_bytes()).hexdigest(),
                 'coordinate_system':'full_image_xy','regions':[{
                     'region_id':'uncertain_tail','kind':'context_inferred',
                     'exclude_from_precision':True,
                     'polygon_xy':[[100,20],[140,20],[140,60],[100,60]]}]}
        annotation.with_suffix('.quality.json').write_text(json.dumps(quality),'utf-8')
    samples.append(sample);return sample,label


def _write_manifest(folder,train_quality=True):
    folder=Path(folder);samples=[]
    train,_=_add_sample(folder,samples,'train_case','train_clip','train',train_quality)
    holdout,holdout_label=_add_sample(folder,samples,'holdout_case','holdout_clip','holdout',False)
    manifest=folder/'manifest.json'
    manifest.write_text(json.dumps({'schema_version':1,
        'task':'seed_conditioned_new_winding_centerline','samples':samples}),'utf-8')
    return manifest,train,holdout,holdout_label


@unittest.skipUnless(_torch_available(),'vision_training Python with torch is unavailable')
class CenterlineContextV6Tests(unittest.TestCase):
    def test_letterbox_half_pixel_roundtrip_and_unpadded_crop(self):
        code=r'''
import numpy as np
from vision_geometry_lab.centerline_context_v6 import (letterbox_geometry,
    full_image_xy_to_model_xy,model_xy_to_full_image_xy,crop_letterbox_probabilities,
    _resize_exclusion_conservative,build_parser)
roi=[8,8,152,72];geometry=letterbox_geometry([64,144],[64,128])
assert geometry['model_input_hw']==[64,128]
assert geometry['content_hw']==[57,128]
assert geometry['content_xyxy']==[0,3,128,60]
points=np.array([[8,8],[8.5,9.25],[80,40],[151,71]],float)
model=full_image_xy_to_model_xy(points,roi,geometry)
restored=model_xy_to_full_image_xy(model,roi,geometry)
assert np.allclose(restored,points,atol=1e-10)
content_grid=(points-np.array(roi[:2])+.5)*np.array(geometry['scale_xy'])-.5
assert np.allclose(model-np.array(geometry['content_xyxy'][:2]),content_grid,atol=1e-10)
canvas=np.arange(64*128,dtype=np.float32).reshape(64,128)
cropped=crop_letterbox_probabilities(canvas,geometry)
assert cropped.shape==(57,128)
assert np.array_equal(cropped,canvas[3:60,:])
thin=np.zeros((64,144),np.float32);thin[:,72]=1
conservative=_resize_exclusion_conservative(thin,[57,128])
assert conservative.any() and conservative.shape==(57,128)
args=build_parser().parse_args(['train-manifest','--manifest','fixture.json'])
assert args.architecture=='context' and args.target_height==512 and args.target_width==1024
assert args.steps==60 and args.seed==20260926 and args.cache_samples is False
'''
        _run_training_python(['-c',code])

    def test_native_raster_letterbox_input_is_shared_and_quality_only_masks_validity(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,_,_,_=_write_manifest(folder,train_quality=True)
            code=r'''
import sys,numpy as np,torch
from vision_geometry_lab.centerline_model_v5 import inspect_manifest
from vision_geometry_lab.centerline_context_v6 import (ManifestLetterboxDataset,
    letterbox_manifest_input,rasterize_letterbox_record)
record=inspect_manifest(sys.argv[1])['usable']['train'][0]
raster=rasterize_letterbox_record(record,[64,128])
inference,geometry,_,_=letterbox_manifest_input(record,[64,128])
assert torch.equal(raster['input'],inference)
left,top,right,bottom=geometry['content_xyxy']
assert top>0 or left>0
padding=torch.ones(64,128,dtype=torch.bool);padding[top:bottom,left:right]=False
assert raster['body_valid_mask'][0][padding].sum().item()==0
assert raster['center_valid_mask'][0][padding].sum().item()==0
assert raster['input'][:,padding].abs().sum().item()==0

control=dict(record);control['parsed_quality_exclude_polygons']=[]
unmasked=rasterize_letterbox_record(control,[64,128])
for key in ('input','body_target','center_target'):
    assert torch.equal(raster[key],unmasked[key]),key
for key in ('body_valid_mask','center_valid_mask'):
    assert raster[key].sum().item()<unmasked[key].sum().item(),key
    assert raster[key].sum().item()>0,key

altered=dict(record);altered['parsed_polylines']=dict(record['parsed_polylines'])
altered['parsed_polylines']['centerline']=np.array([[12.,60.],[145.,60.]])
altered['parsed_polylines']['edge_flange']=np.array([[12.,56.],[145.,56.]])
altered['parsed_polylines']['edge_surface']=np.array([[12.,64.],[145.,64.]])
altered_input,_,_,_=letterbox_manifest_input(altered,[64,128])
assert torch.equal(inference,altered_input)

uncached=ManifestLetterboxDataset([record],[64,128],False)[0]
cached_dataset=ManifestLetterboxDataset([record],[64,128],True);cached=cached_dataset[0]
for key in ('input','body_target','center_target','body_valid_mask','center_valid_mask'):
    assert torch.equal(uncached[key],cached[key]),key
assert cached_dataset[0] is cached and cached_dataset.cache_bytes()>0
'''
            _run_training_python(['-c',code,str(manifest)])

    def test_unknown_quality_and_padding_pixels_have_zero_gradient(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,_,_,_=_write_manifest(folder,train_quality=True)
            code=r'''
import sys,torch
from vision_geometry_lab.centerline_model_v5 import inspect_manifest
from vision_geometry_lab.centerline_context_v6 import rasterize_letterbox_record,masked_two_head_loss
record=inspect_manifest(sys.argv[1])['usable']['train'][0]
sample=rasterize_letterbox_record(record,[64,128])
body_logits=torch.zeros_like(sample['body_target'][None],requires_grad=True)
center_logits=torch.zeros_like(sample['center_target'][None],requires_grad=True)
loss=masked_two_head_loss({'body_logits':body_logits,'center_logits':center_logits},
    sample['body_target'][None],sample['center_target'][None],
    body_valid_mask=sample['body_valid_mask'][None],
    center_valid_mask=sample['center_valid_mask'][None])['loss']
assert torch.isfinite(loss);loss.backward()
body_valid=sample['body_valid_mask'][None]>0;center_valid=sample['center_valid_mask'][None]>0
assert body_logits.grad[~body_valid].abs().sum().item()==0
assert center_logits.grad[~center_valid].abs().sum().item()==0
assert body_logits.grad[body_valid].abs().sum().item()>0
assert center_logits.grad[center_valid].abs().sum().item()>0
assert torch.isfinite(body_logits.grad).all() and torch.isfinite(center_logits.grad).all()
'''
            _run_training_python(['-c',code,str(manifest)])

    def test_shallow_and_context_share_shapes_but_report_distinct_receptive_fields(self):
        code=r'''
import torch
from vision_geometry_lab.centerline_context_v6 import build_model,architecture_metadata
torch.set_num_threads(1);value=torch.rand(1,4,65,129)
shallow=build_model('shallow',4,16);context=build_model('context',4,16)
for model in (shallow,context):
    result=model(value)
    assert result['body_logits'].shape==result['center_logits'].shape==(1,1,65,129)
    try:model(torch.rand(1,3,65,129))
    except ValueError:pass
    else:raise AssertionError('three-channel input was accepted')
assert sum(p.numel() for p in context.parameters())>sum(p.numel() for p in shallow.parameters())
assert architecture_metadata('shallow',4,16)['theoretical_structured_receptive_field_px']==50
meta=architecture_metadata('context',4,16)
assert meta['pool_depth']==6 and meta['theoretical_structured_receptive_field_px']==1078
assert 'not evidence' in meta['receptive_field_claim']
'''
        _run_training_python(['-c',code])

    def test_v6_checkpoint_loads_and_old_square_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);valid=root/'valid.pt';old=root/'old.pt'
            code=r'''
import sys,torch
from vision_geometry_lab.centerline_context_v6 import (CHECKPOINT_FORMAT,PREPROCESS_VERSION,
    build_model,load_model)
model=build_model('shallow',4,16)
payload={'format':CHECKPOINT_FORMAT,'preprocess_version':PREPROCESS_VERSION,
 'raster_encoding_version':2,'model_config':{'architecture':'shallow','input_channels':4,
 'base_channels':4,'channel_cap':16,'heads':['body_logits','center_logits'],
 'input_size_hw':[32,64]},'state_dict':model.state_dict()}
torch.save(payload,sys.argv[1]);loaded,checkpoint,size=load_model(sys.argv[1])
assert size==(32,64) and checkpoint['format']==CHECKPOINT_FORMAT
old=dict(payload);old['format']='centerline_v5_seed_conditioned_two_head';torch.save(old,sys.argv[2])
try:load_model(sys.argv[2])
except ValueError as error:
 assert 'V5 square-resize checkpoints are rejected' in str(error)
else:raise AssertionError('V5 checkpoint was accepted by V6')
'''
            _run_training_python(['-c',code,str(valid),str(old)])

    def test_cli_trains_reports_both_splits_and_exports_benchmark_compatible_crop(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest,_,holdout_sample,holdout_label=_write_manifest(root,True)
            model_dir=root/'model'
            _run_training_python(['-m','vision_geometry_lab.centerline_context_v6','train-manifest',
                                  '--manifest',str(manifest),'--output-dir',str(model_dir),
                                  '--architecture','shallow','--steps','1',
                                  '--target-height','32','--target-width','64',
                                  '--base-channels','4','--channel-cap','16','--batch-size','1',
                                  '--threads','1','--seed','23','--cache-samples'])
            report=json.loads((model_dir/'report.json').read_text('utf-8'))
            self.assertEqual(report['status'],'manifest_training_complete_unvalidated')
            self.assertEqual(report['architecture']['name'],'shallow')
            self.assertEqual(report['training']['model_input_hw'],[32,64])
            self.assertTrue(report['training']['cache_samples'])
            self.assertGreater(report['training']['cache_bytes'],0)
            self.assertEqual(report['data']['train_sample_ids'],['train_case'])
            self.assertEqual(report['data']['holdout_sample_ids'],['holdout_case'])
            for key in ('train_pixel_diagnostic','holdout_pixel_diagnostic'):
                self.assertGreater(report[key]['body_valid_pixels'],0)
                self.assertGreater(report[key]['center_valid_pixels'],0)
                self.assertTrue(math.isfinite(report[key]['masked_two_head_loss']))
            checkpoint=model_dir/'checkpoint.pt';self.assertTrue(checkpoint.is_file())

            exports=root/'exports'
            _run_training_python(['-m','vision_geometry_lab.centerline_context_v6','infer-manifest',
                                  '--manifest',str(manifest),'--checkpoint',str(checkpoint),
                                  '--output-dir',str(exports),'--split','holdout','--threads','1'])
            summary=json.loads((exports/'summary.json').read_text('utf-8'))
            self.assertEqual(summary['status'],'complete')
            self.assertEqual(len(summary['exported']),1)
            metadata=json.loads((exports/'predictions/holdout_case.json').read_text('utf-8'))
            self.assertEqual(metadata['native_roi_hw'],[64,144])
            self.assertEqual(metadata['model_input_hw'],[32,64])
            self.assertEqual(metadata['probability_map_hw'],[28,64])
            self.assertEqual(metadata['letterbox']['content_xyxy'],[0,2,64,30])
            self.assertFalse(metadata['ground_truth_center_or_edges_used_as_input'])
            self.assertTrue(metadata['human_free_seed_used'])
            self.assertTrue(metadata['real_annotation_trained'])
            with np.load(exports/'inputs/holdout_case.npz',allow_pickle=False) as native:
                self.assertEqual(set(native.files),{'rgb','seed_heatmap'})
                self.assertEqual(native['rgb'].shape,(64,144,3))
            with np.load(exports/'predictions/holdout_case.npz',allow_pickle=False) as prediction:
                self.assertEqual(prediction['body_probability'].shape,(28,64))
                self.assertEqual(prediction['center_probability'].shape,(28,64))
                self.assertTrue(np.isfinite(prediction['body_probability']).all())
                self.assertTrue(np.isfinite(prediction['center_probability']).all())

            from vision_geometry_lab.centerline_benchmark_v5 import load_prediction
            from vision_geometry_lab.centerline_review_v5 import validate_annotation
            loaded=load_prediction(exports/'predictions',holdout_sample,
                                   validate_annotation(holdout_label,holdout_sample))
            self.assertIsNotNone(loaded)
            loaded_meta,body,center=loaded
            self.assertEqual(loaded_meta['sample_id'],'holdout_case')
            self.assertEqual(body.shape,center.shape)


if __name__=='__main__':
    unittest.main()
