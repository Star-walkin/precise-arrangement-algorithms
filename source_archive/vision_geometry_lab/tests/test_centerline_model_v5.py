from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT=Path(__file__).resolve().parents[2]
TRAIN_PY=ROOT/'vision_training/.venv/Scripts/python.exe'


def _torch_available():
    if not TRAIN_PY.is_file():
        return False
    try:
        result=subprocess.run([str(TRAIN_PY),'-c','import torch'],cwd=ROOT,
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                              timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return result.returncode==0
    except (OSError,subprocess.TimeoutExpired):
        return False


def _run_training_python(arguments,timeout=120):
    environment=dict(os.environ);environment['PYTHONUTF8']='1';environment['CUDA_VISIBLE_DEVICES']=''
    environment['OMP_NUM_THREADS']='1';environment['MKL_NUM_THREADS']='1'
    result=subprocess.run([str(TRAIN_PY),*arguments],cwd=ROOT,env=environment,
                          text=True,capture_output=True,timeout=timeout,
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:
        raise AssertionError(f'command failed ({result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}')
    return result


@unittest.skipUnless(_torch_available(),'vision_training Python with torch is unavailable')
class CenterlineModelV5Tests(unittest.TestCase):
    def test_ignore_pixels_have_zero_loss_gradient_and_no_nan(self):
        code=r'''
import torch
from vision_geometry_lab.centerline_model_v5 import masked_two_head_loss
body_logits=torch.zeros(1,1,8,8,requires_grad=True)
center_logits=torch.zeros(1,1,8,8,requires_grad=True)
body_target=torch.ones_like(body_logits);center_target=torch.ones_like(center_logits)
valid=torch.ones_like(body_logits);valid[:,:,2:6,2:6]=0
body_target=body_target.detach();center_target=center_target.detach()
body_target[valid==0]=float('nan');center_target[valid==0]=float('nan')
loss=masked_two_head_loss({'body_logits':body_logits,'center_logits':center_logits},
                          body_target,center_target,valid)['loss']
assert torch.isfinite(loss);loss.backward()
assert body_logits.grad[valid==0].abs().sum().item()==0
assert center_logits.grad[valid==0].abs().sum().item()==0
assert body_logits.grad[valid>0].abs().sum().item()>0
assert center_logits.grad[valid>0].abs().sum().item()>0
assert torch.isfinite(body_logits.grad).all() and torch.isfinite(center_logits.grad).all()
'''
        _run_training_python(['-c',code])

    def test_four_channel_two_head_loss_masks_unknown_pixels_and_backpropagates(self):
        code=r'''
import torch
from vision_geometry_lab.centerline_model_v5 import TinyCenterlineUNet,masked_two_head_loss
torch.set_num_threads(1);torch.manual_seed(7)
model=TinyCenterlineUNet(base_channels=4)
assert model.body_head is not model.center_head
assert sum(parameter.numel() for parameter in model.parameters())<100000
try: model(torch.rand(1,3,17,19))
except ValueError: pass
else: raise AssertionError('three-channel input was not rejected')
outputs=model(torch.rand(2,4,31,29))
assert outputs['body_logits'].shape==(2,1,31,29)
assert outputs['center_logits'].shape==(2,1,31,29)
outputs['body_logits'].retain_grad();outputs['center_logits'].retain_grad()
body=torch.zeros_like(outputs['body_logits']);center=torch.zeros_like(outputs['center_logits'])
body[:,:,8:14,6:20]=1;center[:,:,10:12,6:20]=1
valid=torch.zeros_like(body);valid[:,:,4:24,3:25]=1
first=masked_two_head_loss(outputs,body,center,valid)
body_changed=body.clone();center_changed=center.clone()
body_changed[valid==0]=float('nan');center_changed[valid==0]=1
second=masked_two_head_loss(outputs,body_changed,center_changed,valid)
assert torch.allclose(first['loss'],second['loss'])
assert torch.isfinite(first['loss']) and first['valid_weight'].item()>0
first['loss'].backward()
assert outputs['body_logits'].grad[valid==0].abs().sum().item()==0
assert outputs['center_logits'].grad[valid==0].abs().sum().item()==0
assert outputs['body_logits'].grad[valid>0].abs().sum().item()>0
separate=model(torch.rand(1,4,17,19));body_valid=torch.zeros_like(separate['body_logits'])
center_valid=torch.zeros_like(separate['center_logits']);body_valid[:,:,2:5,2:5]=1
separate['body_logits'].retain_grad();separate['center_logits'].retain_grad()
head_loss=masked_two_head_loss(separate,torch.zeros_like(body_valid),torch.zeros_like(center_valid),
    body_valid_mask=body_valid,center_valid_mask=center_valid)['loss']
head_loss.backward()
assert separate['body_logits'].grad.abs().sum().item()>0
assert separate['center_logits'].grad.abs().sum().item()==0
model.zero_grad(set_to_none=True)
empty_outputs=model(torch.rand(1,4,17,19))
empty=torch.zeros_like(empty_outputs['body_logits'])
empty_loss=masked_two_head_loss(empty_outputs,empty,empty,empty)['loss']
assert empty_loss.item()==0
empty_loss.backward()
assert all(parameter.grad is None or parameter.grad.abs().sum().item()==0 for parameter in model.parameters())
'''
        _run_training_python(['-c',code])

    def test_synthetic_generator_varies_nuisances_and_splits_whole_groups(self):
        code=r'''
import torch
from vision_geometry_lab.centerline_model_v5 import SyntheticCurveDataset,synthetic_curve_sample,split_scene_groups
train,validation=split_scene_groups(9,.33,41)
assert train and validation and not (set(train)&set(validation))
assert set(train)|set(validation)=={f'scene_{index:03d}' for index in range(9)}
assert (train,validation)==split_scene_groups(9,.33,41)
train_dataset=SyntheticCurveDataset(train,3,size=17,seed=41)
validation_dataset=SyntheticCurveDataset(validation,3,size=17,seed=41)
assert all(train_dataset[index]['group_id'] in train for index in range(len(train_dataset)))
assert all(validation_dataset[index]['group_id'] in validation for index in range(len(validation_dataset)))
one=synthetic_curve_sample(2,0,size=33,seed=41)
two=synthetic_curve_sample(2,1,size=33,seed=41)
assert one['input'].shape==(4,33,33)
assert one['body_target'].shape==one['center_target'].shape==one['valid_mask'].shape==(1,33,33)
assert one['input'][3].max().item()>.5
assert not torch.equal(one['input'],two['input'])
assert two['valid_mask'].sum().item()<two['valid_mask'].numel()
'''
        _run_training_python(['-c',code])

    def test_synthetic_cli_trains_saves_loads_and_infers_on_cpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model_dir=root/'model'
            _run_training_python(['-m','vision_geometry_lab.centerline_model_v5','train-synthetic',
                                  '--output-dir',str(model_dir),'--steps','3','--groups','6',
                                  '--variants','2','--size','32','--base-channels','4',
                                  '--batch-size','2','--threads','1','--seed','17'])
            report=json.loads((model_dir/'report.json').read_text('utf-8'))
            self.assertEqual(report['status'],'synthetic_mechanical_smoke_only')
            self.assertEqual(report['raster_encoding_version'],2)
            self.assertEqual(report['training']['steps_completed'],3)
            self.assertGreater(report['training']['elapsed_seconds'],0)
            self.assertEqual(len(report['training']['loss_history']),3)
            self.assertTrue(all(math.isfinite(item['loss']) for item in report['training']['loss_history']))
            self.assertEqual(report['data']['real_annotation_count'],0)
            self.assertTrue(report['data']['groups_disjoint'])
            self.assertFalse(set(report['data']['train_groups'])&set(report['data']['validation_groups']))
            self.assertFalse(report['claim_limits']['field_accuracy_established'])
            self.assertFalse(report['claim_limits']['production_ready'])
            checkpoint=model_dir/'checkpoint.pt';self.assertTrue(checkpoint.is_file())

            input_path=root/'input.npz';output_path=root/'prediction.npz'
            rgb=np.full((35,37,3),128,np.uint8)
            seed=np.zeros((35,37),np.float32);seed[15:20,2:15]=1
            np.savez_compressed(input_path,rgb=rgb,seed_heatmap=seed)
            _run_training_python(['-m','vision_geometry_lab.centerline_model_v5','infer',
                                  '--checkpoint',str(checkpoint),'--input-npz',str(input_path),
                                  '--output-npz',str(output_path),'--threads','1',
                                  '--sample-id','synthetic_cli_probe','--image-sha256','a'*64,
                                  '--roi-xyxy','0','0','37','35',
                                  '--free-seed-source','external_detector'])
            with np.load(output_path,allow_pickle=False) as prediction:
                body=prediction['body_probability'];center=prediction['center_probability']
            self.assertEqual(body.shape,(32,32));self.assertEqual(center.shape,(32,32))
            self.assertTrue(np.isfinite(body).all() and np.isfinite(center).all())
            self.assertTrue(0<=body.min()<=body.max()<=1)
            self.assertTrue(0<=center.min()<=center.max()<=1)
            inference=json.loads(output_path.with_suffix('.json').read_text('utf-8'))
            self.assertEqual(inference['status'],'unvalidated_model_inference')
            self.assertGreater(inference['inference_ms'],0)
            self.assertEqual(inference['roi_xyxy'],[0,0,37,35])
            self.assertEqual(inference['native_roi_hw'],[35,37])
            self.assertEqual(inference['model_input_hw'],[32,32])
            self.assertEqual(inference['probability_map_hw'],[32,32])
            self.assertAlmostEqual(inference['scale_xy'][0],32/37)
            self.assertAlmostEqual(inference['scale_xy'][1],32/35)
            self.assertEqual(inference['checkpoint_training_source'],'synthetic_mechanical_smoke_only')
            self.assertFalse(inference['real_annotation_trained'])
            self.assertTrue(inference['body_head_supervised'])
            self.assertTrue(inference['center_head_supervised'])
            self.assertTrue(inference['supervision_complete'])
            self.assertFalse(inference['human_free_seed_used'])
            self.assertFalse(inference['ground_truth_center_or_edges_used_as_input'])
            self.assertFalse(inference['field_accuracy_established'])
            self.assertFalse(inference['hardware_access'])
            self.assertEqual(inference['raster_encoding_version'],2)

            legacy=root/'legacy_checkpoint.pt'
            code=r'''
import sys,torch
from vision_geometry_lab.centerline_model_v5 import load_model
checkpoint=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
checkpoint.pop('raster_encoding_version',None);torch.save(checkpoint,sys.argv[2])
try: load_model(sys.argv[2])
except ValueError as error: assert 'Legacy checkpoint' in str(error)
else: raise AssertionError('legacy line-map encoding checkpoint was accepted')
'''
            _run_training_python(['-c',code,str(checkpoint),str(legacy)])

    def test_fixed_manifest_schema_trains_only_reviewed_eligible_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);samples=[]

            def add_sample(sample_id,group_id,split,status,with_label=True,quality_polygon=None):
                image_path=(root/f'{sample_id}.png').resolve()
                pixels=np.full((32,32,3),90,np.uint8);pixels[14:20,:,0]=180
                Image.fromarray(pixels,'RGB').save(image_path)
                digest=hashlib.sha256(image_path.read_bytes()).hexdigest()
                annotation=(root/f'{sample_id}.json').resolve()
                if status=='absent':
                    lines={key:[] for key in ('centerline','edge_flange','edge_surface','free_seed')}
                else:
                    lines={'centerline':[[3,16],[12,16],[28,18]],
                           'edge_flange':[[3,14],[12,14],[28,16]],
                           'edge_surface':[[3,18],[12,18],[28,20]],
                           'free_seed':[[3,16],[9,16]]}
                label={'schema_version':1,'sample_id':sample_id,'image_sha256':digest,
                       'coordinate_system':'full_image_xy','status':status,'reviewed':True,
                       'polylines':lines,'ignore_polygons':[[[0,0],[2,0],[2,2],[0,2]]],
                       'notes':'test fixture, not field data'}
                if with_label: annotation.write_text(json.dumps(label),'utf-8')
                sample={'sample_id':sample_id,'group_id':group_id,'split':split,
                        'image_path':str(image_path),'image_sha256':digest,
                        'image_shape_hw':[32,32],'roi_xyxy':[0,0,32,32],
                        'annotation_path':str(annotation),
                        'source':{'video_path':'fixture.avi','frame_index':0,
                                  'time_s':0.0,'camera':'fixture'}}
                samples.append(sample)
                if with_label and quality_polygon is not None:
                    quality={'schema_version':1,'sample_id':sample_id,'image_sha256':digest,
                             'annotation_sha256':hashlib.sha256(annotation.read_bytes()).hexdigest(),
                             'coordinate_system':'full_image_xy','regions':[{
                                 'region_id':'fixture_uncertain','kind':'coarse_direction_only',
                                 'exclude_from_precision':True,'polygon_xy':quality_polygon}]}
                    quality_path=annotation.with_suffix('.quality.json')
                    quality_path.write_text(json.dumps(quality),'utf-8')
                    return hashlib.sha256(quality_path.read_bytes()).hexdigest()
                return None

            train_quality=add_sample('train_clear','train_clip_a','train','clear',
                                     quality_polygon=[[0,0],[2,0],[2,2],[0,2]])
            add_sample('train_absent','train_clip_b','train','absent')
            add_sample('train_missing','train_clip_c','train','clear',with_label=False)
            holdout_quality=add_sample('holdout_partial','holdout_clip_a','holdout','partial',
                                       quality_polygon=[[0,0],[2,0],[2,2],[0,2]])
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'schema_version':1,
                'task':'seed_conditioned_new_winding_centerline','samples':samples}),'utf-8')
            code=r'''
import sys,numpy as np
from vision_geometry_lab.centerline_model_v5 import (inspect_manifest,manifest_native_input,
    rasterize_manifest_record,_resize_native_model_input,_polyline_image,
    ManifestCurveDataset,build_parser)
info=inspect_manifest(sys.argv[1]);records={r['sample_id']:r for s in ('train','holdout') for r in info['usable'][s]}
clear=records['train_clear'];absent=records['train_absent']
uncached=ManifestCurveDataset([clear],32,cache_samples=False)[0]
cached_dataset=ManifestCurveDataset([clear],32,cache_samples=True)
cached=cached_dataset[0]
for key in ('input','body_target','center_target','body_valid_mask','center_valid_mask'):
    assert np.array_equal(uncached[key].numpy(),cached[key].numpy()),key
assert uncached['sample_id']==cached['sample_id'] and uncached['group_id']==cached['group_id']
assert cached_dataset[0] is cached
base_args=build_parser().parse_args(['train-manifest','--manifest',sys.argv[1]])
cache_args=build_parser().parse_args(['train-manifest','--manifest',sys.argv[1],'--cache-samples'])
assert base_args.cache_samples is False and cache_args.cache_samples is True
training=rasterize_manifest_record(clear,32);rgb,native_seed=manifest_native_input(clear)
_,inference_seed=_resize_native_model_input(rgb,native_seed,[32,32])
assert np.isfinite(training['center_target'].numpy()).all()
assert np.isfinite(training['input'].numpy()[3]).all()
assert abs(float(training['center_target'].max())-1.)<1e-6
assert abs(float(training['input'][3].max())-1.)<1e-6
assert np.allclose(training['input'].numpy()[3],inference_seed,atol=1e-7)
empty=rasterize_manifest_record(absent,32)
assert float(empty['center_target'].max())==0 and float(empty['input'][3].max())==0
empty_map=_polyline_image((20,20),[],blur=3,normalize_peak=True)
assert np.isfinite(empty_map).all() and not empty_map.any()
'''
            _run_training_python(['-c',code,str(manifest)])
            output=root/'manifest_model'
            _run_training_python(['-m','vision_geometry_lab.centerline_model_v5','train-manifest',
                                  '--manifest',str(manifest),'--output-dir',str(output),
                                  '--steps','2','--size','32','--base-channels','4',
                                  '--batch-size','2','--threads','1','--cache-samples'])
            report=json.loads((output/'report.json').read_text('utf-8'))
            self.assertEqual(report['status'],'manifest_training_complete_unvalidated')
            self.assertEqual(set(report['data']['train_sample_ids']),{'train_clear','train_absent'})
            self.assertEqual(report['data']['holdout_sample_ids'],['holdout_partial'])
            self.assertEqual(report['data']['skipped_samples'][0]['sample_id'],'train_missing')
            self.assertEqual(set(report['data']['train_annotation_sha256_by_sample']),
                             {'train_clear','train_absent'})
            self.assertEqual(set(report['data']['holdout_annotation_sha256_by_sample']),
                             {'holdout_partial'})
            self.assertEqual(report['data']['train_quality_sha256_by_sample'],
                             {'train_clear':train_quality,'train_absent':None})
            self.assertEqual(report['data']['holdout_quality_sha256_by_sample'],
                             {'holdout_partial':holdout_quality})
            self.assertTrue(report['supervision']['body_head_supervised'])
            self.assertTrue(report['supervision']['center_head_supervised'])
            self.assertEqual(report['supervision']['body_valid_sample_count'],2)
            self.assertEqual(report['supervision']['center_valid_sample_count'],2)
            self.assertTrue(report['training']['cache_samples'])
            self.assertGreaterEqual(report['training']['cache_preparation_seconds'],0)
            self.assertGreater(report['training']['model_forward_seconds'],0)
            self.assertGreaterEqual(report['training']['batch_fetch_seconds'],0)
            self.assertGreater(report['training']['command_elapsed_before_report_seconds'],
                               report['training']['elapsed_seconds'])
            self.assertFalse(report['data']['holdout_used_for_training'])
            self.assertFalse(report['data']['holdout_used_for_tuning_by_code'])
            first_step=report['training']['loss_history'][0]
            self.assertTrue(1024<first_step['body_valid_pixels']<2048)
            self.assertTrue(1024<first_step['center_valid_pixels']<2048)
            self.assertIsNotNone(report['holdout_diagnostic'])
            self.assertFalse(report['claim_limits']['field_accuracy_established'])
            checkpoint=output/'checkpoint.pt';self.assertTrue(checkpoint.is_file())

            exports=root/'exports'
            _run_training_python(['-m','vision_geometry_lab.centerline_model_v5','infer-manifest',
                                  '--manifest',str(manifest),'--checkpoint',str(checkpoint),
                                  '--output-dir',str(exports),'--split','all','--threads','1'])
            summary=json.loads((exports/'summary.json').read_text('utf-8'))
            self.assertEqual(summary['status'],'complete')
            self.assertEqual({item['sample_id'] for item in summary['exported']},
                             {'train_clear','holdout_partial'})
            self.assertEqual(summary['model_input_hw'],[32,32])
            self.assertEqual(summary['train_quality_sha256_by_sample'],
                             {'train_clear':train_quality,'train_absent':None})
            self.assertFalse(summary['ground_truth_center_or_edges_used_as_input'])
            metadata=json.loads((exports/'predictions/holdout_partial.json').read_text('utf-8'))
            self.assertEqual(metadata['manifest_split'],'holdout')
            self.assertEqual(len(metadata['annotation_sha256']),64)
            self.assertEqual(len(metadata['free_seed_sha256']),64)
            self.assertEqual(metadata['quality_sha256'],holdout_quality)
            self.assertEqual(metadata['train_quality_sha256_by_sample'],
                             {'train_clear':train_quality,'train_absent':None})
            self.assertTrue(metadata['human_free_seed_used'])
            self.assertFalse(metadata['ground_truth_center_or_edges_used_as_input'])
            self.assertEqual(metadata['train_group_ids'],['train_clip_a','train_clip_b'])
            self.assertTrue(metadata['body_head_supervised'])
            self.assertTrue(metadata['center_head_supervised'])
            self.assertTrue(metadata['supervision_complete'])
            with np.load(exports/'inputs/holdout_partial.npz',allow_pickle=False) as native_input:
                self.assertEqual(native_input['rgb'].shape,(32,32,3))
                self.assertEqual(native_input['seed_heatmap'].shape,(32,32))
                self.assertNotIn('centerline',native_input.files)
                self.assertNotIn('edge_flange',native_input.files)
            with np.load(exports/'predictions/holdout_partial.npz',allow_pickle=False) as prediction:
                self.assertEqual(prediction['body_probability'].shape,(32,32))
                self.assertEqual(prediction['center_probability'].shape,(32,32))

    def test_manifest_with_no_reviewed_labels_is_blocked_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);image=(root/'waiting.png').resolve()
            Image.fromarray(np.zeros((24,40,3),np.uint8),'RGB').save(image)
            digest=hashlib.sha256(image.read_bytes()).hexdigest()
            manifest=root/'manifest.json';output=root/'output'
            manifest.write_text(json.dumps({'schema_version':1,
                'task':'seed_conditioned_new_winding_centerline','samples':[{
                    'sample_id':'waiting','group_id':'train_clip','split':'train',
                    'image_path':str(image),'image_sha256':digest,'image_shape_hw':[24,40],
                    'roi_xyxy':[0,0,40,24],'annotation_path':str((root/'missing.json').resolve()),
                    'source':{'video_path':'fixture.avi','frame_index':0,'time_s':0.0,'camera':'left'}}]}),'utf-8')
            _run_training_python(['-m','vision_geometry_lab.centerline_model_v5','train-manifest',
                                  '--manifest',str(manifest),'--output-dir',str(output),
                                  '--steps','2','--threads','1'])
            report=json.loads((output/'report.json').read_text('utf-8'))
            self.assertEqual(report['status'],'blocked_no_reviewed_labels')
            self.assertEqual(report['training']['steps_completed'],0)
            self.assertEqual(report['artifacts']['checkpoint'],None)
            self.assertFalse(report['supervision']['body_head_supervised'])
            self.assertFalse(report['supervision']['center_head_supervised'])
            self.assertEqual(report['data']['resize_audit'][0]['native_roi_hw'],[24,40])
            self.assertEqual(report['data']['resize_audit'][0]['model_input_hw'],[512,512])
            self.assertEqual(report['data']['resize_audit'][0]['scale_xy'],[512/40,512/24])
            self.assertFalse((output/'checkpoint.pt').exists())
            self.assertEqual(report['data']['skipped_samples'][0]['reason'],'missing_annotation')

    def test_quality_sidecar_excludes_precision_only_and_stale_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);samples=[]

            def add_sample(sample_id,polygon,annotation_hash_override=None):
                image=(root/f'{sample_id}.png').resolve()
                pixels=np.full((64,64,3),110,np.uint8);pixels[28:37,:,1]=190
                Image.fromarray(pixels,'RGB').save(image)
                image_hash=hashlib.sha256(image.read_bytes()).hexdigest()
                annotation=(root/f'{sample_id}.json').resolve()
                label={'schema_version':1,'sample_id':sample_id,'image_sha256':image_hash,
                       'coordinate_system':'full_image_xy','status':'partial','reviewed':True,
                       'polylines':{'centerline':[[4,32],[32,32],[59,34]],
                                    'edge_flange':[[4,28],[32,28],[59,30]],
                                    'edge_surface':[[4,36],[32,36],[59,38]],
                                    'free_seed':[[4,32],[16,32]]},
                       'ignore_polygons':[],'notes':'quality-mask fixture'}
                annotation.write_text(json.dumps(label),'utf-8')
                annotation_hash=hashlib.sha256(annotation.read_bytes()).hexdigest()
                sample={'sample_id':sample_id,'group_id':f'clip_{sample_id}','split':'train',
                        'image_path':str(image),'image_sha256':image_hash,
                        'image_shape_hw':[64,64],'roi_xyxy':[0,0,64,64],
                        'annotation_path':str(annotation),
                        'source':{'video_path':'fixture.avi','frame_index':0,
                                  'time_s':0.0,'camera':'fixture'}}
                quality={'schema_version':1,'sample_id':sample_id,'image_sha256':image_hash,
                         'annotation_sha256':annotation_hash_override or annotation_hash,
                         'coordinate_system':'full_image_xy','regions':[{
                             'region_id':'uncertain_terminal','kind':'context_inferred',
                             'exclude_from_precision':True,'polygon_xy':polygon}]}
                quality_path=annotation.with_suffix('.quality.json')
                quality_path.write_text(json.dumps(quality),'utf-8')
                samples.append(sample)
                return hashlib.sha256(quality_path.read_bytes()).hexdigest()

            partial_hash=add_sample('partial_exclusion',[[32,0],[63,0],[63,63],[32,63]])
            full_hash=add_sample('full_exclusion',[[0,0],[63,0],[63,63],[0,63]])
            add_sample('stale_exclusion',[[32,0],[63,0],[63,63],[32,63]],'0'*64)
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'schema_version':1,
                'task':'seed_conditioned_new_winding_centerline','samples':samples}),'utf-8')
            code=r'''
import sys,numpy as np,torch
from vision_geometry_lab.centerline_model_v5 import (inspect_manifest,
    manifest_supervision_summary,masked_two_head_loss,rasterize_manifest_record)
info=inspect_manifest(sys.argv[1])
records={record['sample_id']:record for record in info['usable']['train']}
assert set(records)=={'partial_exclusion','full_exclusion'}
assert records['partial_exclusion']['quality_sha256']==sys.argv[2]
assert records['full_exclusion']['quality_sha256']==sys.argv[3]
assert any(item['sample_id']=='stale_exclusion' and
           'Quality annotation_sha256 mismatch' in item['reason'] for item in info['skipped'])

record=records['partial_exclusion'];masked=rasterize_manifest_record(record,16)
control=dict(record);control['parsed_quality_exclude_polygons']=[]
unmasked=rasterize_manifest_record(control,16)
for key in ('input','body_target','center_target'):
    assert torch.equal(masked[key],unmasked[key]),key
for key in ('body_valid_mask','center_valid_mask'):
    assert masked[key][:,:,8:].sum().item()==0,key
    assert masked[key][:,:,:8].sum().item()>0,key
    assert masked[key].sum().item()<unmasked[key].sum().item(),key
assert masked['body_target'][:,:,8:].sum().item()>0
assert masked['center_target'][:,:,8:].sum().item()>0

body_logits=torch.zeros_like(masked['body_target'][None],requires_grad=True)
center_logits=torch.zeros_like(masked['center_target'][None],requires_grad=True)
loss=masked_two_head_loss({'body_logits':body_logits,'center_logits':center_logits},
    masked['body_target'][None],masked['center_target'][None],
    body_valid_mask=masked['body_valid_mask'][None],
    center_valid_mask=masked['center_valid_mask'][None])['loss']
assert torch.isfinite(loss);loss.backward()
assert body_logits.grad[:,:,:,8:].abs().sum().item()==0
assert center_logits.grad[:,:,:,8:].abs().sum().item()==0
assert body_logits.grad[:,:,:,:8].abs().sum().item()>0
assert center_logits.grad[:,:,:,:8].abs().sum().item()>0

full=rasterize_manifest_record(records['full_exclusion'],16)
assert full['body_target'].sum().item()>0 and full['center_target'].sum().item()>0
assert full['body_valid_mask'].sum().item()==0
assert full['center_valid_mask'].sum().item()==0
partial_summary=manifest_supervision_summary([record],16)
assert partial_summary['body_head_supervised'] and partial_summary['center_head_supervised']
full_summary=manifest_supervision_summary([records['full_exclusion']],16)
assert not full_summary['body_head_supervised'] and not full_summary['center_head_supervised']
assert full_summary['body_valid_sample_count']==0
assert full_summary['center_valid_sample_count']==0
'''
            _run_training_python(['-c',code,str(manifest),partial_hash,full_hash])

            output=root/'blocked_model'
            full_manifest=root/'full_manifest.json'
            full_manifest.write_text(json.dumps({'schema_version':1,
                'task':'seed_conditioned_new_winding_centerline','samples':[samples[1]]}),'utf-8')
            _run_training_python(['-m','vision_geometry_lab.centerline_model_v5','train-manifest',
                                  '--manifest',str(full_manifest),'--output-dir',str(output),
                                  '--steps','1','--size','16','--threads','1'])
            report=json.loads((output/'report.json').read_text('utf-8'))
            self.assertEqual(report['status'],'blocked_no_positive_target_supervision')
            self.assertFalse(report['supervision']['body_head_supervised'])
            self.assertFalse(report['supervision']['center_head_supervised'])
            self.assertIsNone(report['artifacts']['checkpoint'])

    def test_center_only_manifest_is_explicit_partial_supervision(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);image=(root/'center_only.png').resolve()
            Image.fromarray(np.full((64,128,3),120,np.uint8),'RGB').save(image)
            digest=hashlib.sha256(image.read_bytes()).hexdigest();label_path=(root/'label.json').resolve()
            label={'schema_version':1,'sample_id':'center_only','image_sha256':digest,
                   'coordinate_system':'full_image_xy','status':'partial','reviewed':True,
                   'polylines':{'centerline':[[4,32],[112,36]],'edge_flange':[],
                                'edge_surface':[],'free_seed':[[4,32],[28,32]]},
                   'ignore_polygons':[],'notes':'center-only test fixture'}
            label_path.write_text(json.dumps(label),'utf-8')
            manifest=root/'manifest.json';output=root/'model'
            manifest.write_text(json.dumps({'schema_version':1,
                'task':'seed_conditioned_new_winding_centerline','samples':[{
                    'sample_id':'center_only','group_id':'train_clip','split':'train',
                    'image_path':str(image),'image_sha256':digest,'image_shape_hw':[64,128],
                    'roi_xyxy':[0,0,128,64],'annotation_path':str(label_path),
                    'source':{'video_path':'fixture.avi','frame_index':0,'time_s':0.0,'camera':'left'}}]}),'utf-8')
            code=r'''
import sys,numpy as np
from vision_geometry_lab.centerline_model_v5 import (inspect_manifest,manifest_native_input,
    rasterize_manifest_record,_resize_native_model_input,_polyline_image)
record=inspect_manifest(sys.argv[1])['usable']['train'][0]
training=rasterize_manifest_record(record,32)['input'].numpy()
native_rgb,native_seed=manifest_native_input(record)
infer_rgb,infer_seed=_resize_native_model_input(native_rgb,native_seed,[32,32])
inference=np.concatenate([infer_rgb.transpose(2,0,1),infer_seed[None]],axis=0)
assert training.shape==inference.shape==(4,32,32)
assert np.allclose(training,inference,atol=1e-7)
assert np.isfinite(training).all() and float(training[3].max())==1.
empty=_polyline_image((128,64),[],blur=4,normalize_peak=True)
assert empty.shape==(64,128) and np.isfinite(empty).all() and float(empty.max())==0.
'''
            _run_training_python(['-c',code,str(manifest)])
            _run_training_python(['-m','vision_geometry_lab.centerline_model_v5','train-manifest',
                                  '--manifest',str(manifest),'--output-dir',str(output),
                                  '--steps','1','--size','32','--base-channels','4',
                                  '--batch-size','1','--threads','1'])
            report=json.loads((output/'report.json').read_text('utf-8'))
            self.assertEqual(report['status'],'partial_supervision')
            self.assertFalse(report['supervision']['body_head_supervised'])
            self.assertTrue(report['supervision']['center_head_supervised'])
            self.assertEqual(report['supervision']['body_valid_sample_count'],0)
            self.assertEqual(report['supervision']['center_valid_sample_count'],1)

            input_path=root/'input.npz';prediction=root/'prediction.npz'
            np.savez_compressed(input_path,rgb=np.full((64,128,3),.5,np.float32),
                                seed_heatmap=np.zeros((64,128),np.float32))
            _run_training_python(['-m','vision_geometry_lab.centerline_model_v5','infer',
                                  '--checkpoint',str(output/'checkpoint.pt'),'--input-npz',str(input_path),
                                  '--output-npz',str(prediction),'--sample-id','center_only',
                                  '--image-sha256',digest,'--roi-xyxy','0','0','128','64',
                                  '--free-seed-source','human_polyline','--threads','1'])
            metadata=json.loads(prediction.with_suffix('.json').read_text('utf-8'))
            self.assertFalse(metadata['body_head_supervised'])
            self.assertTrue(metadata['center_head_supervised'])
            self.assertFalse(metadata['supervision_complete'])
            self.assertEqual(metadata['native_roi_hw'],[64,128])
            self.assertEqual(metadata['model_input_hw'],[32,32])
            self.assertEqual(metadata['scale_xy'],[.25,.5])

    def test_manifest_reuses_gui_validation_and_rejects_line_crossing_ignore(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);image=(root/'crossing.png').resolve()
            Image.fromarray(np.full((32,32,3),100,np.uint8),'RGB').save(image)
            digest=hashlib.sha256(image.read_bytes()).hexdigest();label_path=(root/'label.json').resolve()
            label={'schema_version':1,'sample_id':'crossing','image_sha256':digest,
                   'coordinate_system':'full_image_xy','status':'partial','reviewed':True,
                   'polylines':{'centerline':[[2,16],[30,16]],'edge_flange':[],
                                'edge_surface':[],'free_seed':[[2,16],[8,16]]},
                   'ignore_polygons':[[[14,10],[18,10],[18,20],[14,20]]],
                   'notes':'line intentionally crosses ignore for validation test'}
            label_path.write_text(json.dumps(label),'utf-8')
            manifest=root/'manifest.json';output=root/'output'
            manifest.write_text(json.dumps({'schema_version':1,
                'task':'seed_conditioned_new_winding_centerline','samples':[{
                    'sample_id':'crossing','group_id':'train_clip','split':'train',
                    'image_path':str(image),'image_sha256':digest,'image_shape_hw':[32,32],
                    'roi_xyxy':[0,0,32,32],'annotation_path':str(label_path),
                    'source':{'video_path':'fixture.avi','frame_index':0,'time_s':0.0,'camera':'left'}}]}),'utf-8')
            _run_training_python(['-m','vision_geometry_lab.centerline_model_v5','train-manifest',
                                  '--manifest',str(manifest),'--output-dir',str(output),
                                  '--steps','1','--threads','1'])
            report=json.loads((output/'report.json').read_text('utf-8'))
            self.assertEqual(report['status'],'blocked_no_reviewed_labels')
            self.assertEqual(report['training']['steps_completed'],0)
            self.assertIsNone(report['artifacts']['checkpoint'])
            self.assertIn('crosses the ignore region',report['data']['skipped_samples'][0]['reason'])


if __name__=='__main__':
    unittest.main()
