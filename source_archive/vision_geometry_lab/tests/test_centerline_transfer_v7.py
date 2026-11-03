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


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = ROOT / 'vision_training/.venv/Scripts/python.exe'
SAM_CHECKPOINT = ROOT / 'vision_training/checkpoints/sam2.1_hiera_tiny.pt'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _runtime_available():
    if not TRAIN_PY.is_file() or not SAM_CHECKPOINT.is_file():
        return False
    try:
        result = subprocess.run(
            [str(TRAIN_PY), '-c', 'import torch,sam2'], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _run(arguments, timeout=240):
    environment = dict(os.environ)
    environment['PYTHONUTF8'] = '1'
    environment['CUDA_VISIBLE_DEVICES'] = ''
    environment['OMP_NUM_THREADS'] = '1'
    environment['MKL_NUM_THREADS'] = '1'
    result = subprocess.run(
        [str(TRAIN_PY), *arguments], cwd=ROOT, env=environment,
        text=True, capture_output=True, timeout=timeout,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise AssertionError(
            f'command failed ({result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}')
    return result


def _add_sample(folder, samples, sample_id, group_id, split, with_quality=False):
    folder = Path(folder)
    image = (folder / f'{sample_id}.png').resolve()
    height, width = 80, 160
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack([(70 + xx // 3) % 255, (90 + yy * 2) % 255,
                       np.full_like(xx, 120)], axis=2).astype(np.uint8)
    pixels[36:45, 8:152] = np.array([190, 175, 145], np.uint8)
    Image.fromarray(pixels, 'RGB').save(image)
    image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    annotation = (folder / f'{sample_id}.json').resolve()
    label = {
        'schema_version': 1, 'sample_id': sample_id, 'image_sha256': image_hash,
        'coordinate_system': 'full_image_xy', 'status': 'partial', 'reviewed': True,
        'polylines': {
            'centerline': [[12, 40], [80, 40], [145, 44]],
            'edge_flange': [[12, 36], [80, 36], [145, 40]],
            'edge_surface': [[12, 44], [80, 44], [145, 48]],
            'free_seed': [[12, 40], [30, 40]],
        },
        'ignore_polygons': [], 'notes': 'V7 deterministic fixture',
    }
    annotation.write_text(json.dumps(label), 'utf-8')
    sample = {
        'sample_id': sample_id, 'group_id': group_id, 'split': split,
        'image_path': str(image), 'image_sha256': image_hash,
        'image_shape_hw': [height, width], 'roi_xyxy': [8, 8, 152, 72],
        'annotation_path': str(annotation),
        'source': {'video_path': f'{group_id}.avi', 'frame_index': 1,
                   'time_s': .1, 'camera': 'fixture'},
    }
    if with_quality:
        quality = {
            'schema_version': 1, 'sample_id': sample_id, 'image_sha256': image_hash,
            'annotation_sha256': hashlib.sha256(annotation.read_bytes()).hexdigest(),
            'coordinate_system': 'full_image_xy',
            'regions': [{'region_id': 'uncertain_tail', 'kind': 'context_inferred',
                         'exclude_from_precision': True,
                         'polygon_xy': [[100, 20], [140, 20], [140, 60], [100, 60]]}],
        }
        annotation.with_suffix('.quality.json').write_text(json.dumps(quality), 'utf-8')
    samples.append(sample)
    return sample, label


def _write_manifest(folder):
    folder = Path(folder)
    samples = []
    train, _ = _add_sample(folder, samples, 'train_case', 'train_clip', 'train', True)
    holdout, holdout_label = _add_sample(
        folder, samples, 'holdout_case', 'holdout_clip', 'holdout', False)
    manifest = folder / 'manifest.json'
    manifest.write_text(json.dumps({
        'schema_version': 1, 'task': 'seed_conditioned_new_winding_centerline',
        'samples': samples}), 'utf-8')
    return manifest, train, holdout, holdout_label


@unittest.skipUnless(_runtime_available(), 'vision_training torch/SAM2 runtime is unavailable')
class CenterlineTransferV7Tests(unittest.TestCase):
    def test_embedding_and_all_three_feature_crops_are_exact_integer_slices(self):
        code = r'''
import torch
from vision_geometry_lab.centerline_transfer_v7 import (sam_embedding_geometry,
    sam_normalized_square,crop_sam_feature_pyramid,expected_cropped_feature_shapes)
geometry=sam_embedding_geometry([512,1024])
assert geometry['canvas_xyxy_in_sam']==[0,256,1024,768]
assert geometry['feature_crop_xyxy']==[[0,64,256,192],[0,32,128,96],[0,16,64,48]]
assert expected_cropped_feature_shapes([512,1024])==(
    (32,128,256),(64,64,128),(256,32,64))
first=torch.zeros(4,512,1024);first[:3]=.25;first[3]=1
second=first.clone();second[3]=0
square1,mapping1=sam_normalized_square(first)
square2,mapping2=sam_normalized_square(second)
assert torch.equal(square1,square2) and mapping1==mapping2
fpn=[]
for channels,size in ((32,256),(64,128),(256,64)):
    rows=torch.arange(size,dtype=torch.float32).view(1,1,size,1)
    fpn.append(rows.expand(1,channels,size,size).clone())
cropped=crop_sam_feature_pyramid(fpn,geometry)
assert [list(value.shape) for value in cropped]==[[32,128,256],[64,64,128],[256,32,64]]
assert cropped[0][0,0,0].item()==64 and cropped[0][0,-1,0].item()==191
assert cropped[1][0,0,0].item()==32 and cropped[2][0,0,0].item()==16
'''
        _run(['-c', code])

    def test_identical_head_shapes_seed_and_zero_control_with_masked_gradients(self):
        code = r'''
import torch
from torch.utils.data import DataLoader
from vision_geometry_lab.centerline_transfer_v7 import (build_head,expected_cropped_feature_shapes,
    initialize_head,training_order_generator)
from vision_geometry_lab.centerline_model_v5 import masked_two_head_loss
torch.set_num_threads(1);torch.manual_seed(29);one=build_head(4,3)
torch.manual_seed(29);two=build_head(4,3)
assert list(one.state_dict())==list(two.state_dict())
for key in one.state_dict():assert torch.equal(one.state_dict()[key],two.state_dict()[key])
value=torch.rand(1,4,32,64)
features=tuple(torch.zeros((1,*shape)) for shape in expected_cropped_feature_shapes([32,64]))
assert all(feature.count_nonzero().item()==0 for feature in features)
result=one(value,features)
assert result['body_logits'].shape==result['center_logits'].shape==(1,1,32,64)
body=torch.zeros(1,1,32,64);center=torch.zeros_like(body)
valid=torch.zeros_like(body);valid[:,:,:16,:32]=1
loss=masked_two_head_loss(result,body,center,body_valid_mask=valid,
                          center_valid_mask=valid)['loss']
assert torch.isfinite(loss);loss.backward()
assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
           for parameter in one.parameters())
# Simulate arbitrary RNG consumption by SAM construction.  Both paired modes
# still reset immediately before the common head and use an isolated sampler.
control,control_hash=initialize_head(20260926,4,3)
torch.manual_seed(999);torch.rand(10000)
transfer,transfer_hash=initialize_head(20260926,4,3)
assert control_hash==transfer_hash
for key in control.state_dict():assert torch.equal(control.state_dict()[key],transfer.state_dict()[key])
dataset=list(range(11))
order1=[int(x) for batch in DataLoader(dataset,batch_size=3,shuffle=True,
        generator=training_order_generator(20260926)) for x in batch]
torch.rand(1000)
order2=[int(x) for batch in DataLoader(dataset,batch_size=3,shuffle=True,
        generator=training_order_generator(20260926)) for x in batch]
assert order1==order2
try:one(torch.rand(1,3,32,64),features)
except ValueError:pass
else:raise AssertionError('three-channel task input was accepted')
'''
        _run(['-c', code])

    def test_rgb_only_feature_cache_ignores_seed_and_target_annotations(self):
        with tempfile.TemporaryDirectory() as folder:
            code = r'''
import json,sys,torch
from pathlib import Path
import vision_geometry_lab.centerline_transfer_v7 as v7
class FakeExtractor:
 def __init__(self):
  self.checkpoint=Path(v7.DEFAULT_SAM_CHECKPOINT).resolve()
  self.checkpoint_sha256=v7.EXPECTED_SAM_CHECKPOINT_SHA256
  self.config_sha256=v7.EXPECTED_SAM_CONFIG_SHA256
  self.extractor_code_sha256=v7._module_sha256()
  self.vendor_code_sha256=v7._vendor_code_sha256();self.calls=0
 def extract(self,input_tensor):
  self.calls+=1
  features=tuple(torch.full(shape,float(i+1))
                 for i,shape in enumerate(v7.expected_cropped_feature_shapes([32,64])))
  return features,{'forward_seconds':.01,'feature_bytes_float32':sum(x.numel()*4 for x in features),
                   'feature_shapes_chw':[list(x.shape) for x in features],
                   'sam_embedding':v7.sam_embedding_geometry([32,64])}
 def release(self):pass
fake=FakeExtractor();provider=v7.FeatureProvider('sam2_frozen',[32,64],sys.argv[1],
    v7.DEFAULT_SAM_CHECKPOINT,fake,1)
record={'sample_id':'same_rgb','image_sha256':'a'*64,'roi_xyxy':[0,0,64,32],
        'parsed_polylines':{'centerline':[[1,1],[20,20]],'free_seed':[[1,1],[3,3]]}}
first=torch.rand(4,32,64);second=first.clone();second[3]=1-first[3]
features1,provenance1=provider.get(record,first)
altered=dict(record);altered['parsed_polylines']={'centerline':[[2,2],[50,20]],
                                                  'free_seed':[[20,20],[30,20]]}
features2,provenance2=provider.get(altered,second)
assert fake.calls==1 and provider.cache_misses==1 and provider.cache_hits==1
assert all(torch.equal(a,b) for a,b in zip(features1,features2))
assert provenance1['cache_npz']==provenance2['cache_npz']
with __import__('numpy').load(provenance1['cache_npz'],allow_pickle=False) as arrays:
 meta=json.loads(str(arrays['metadata_json'].item()))
 assert meta['ground_truth_used'] is False and meta['free_seed_used'] is False
 assert 'annotation_sha256' not in meta and 'free_seed_sha256' not in meta
assert provenance1['input_semantics']=='rgb_only_no_seed_no_target'
'''
            _run(['-c', code, folder])

    def test_v6_raster_quality_padding_and_unknown_gradient_are_reused_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, _, _, _ = _write_manifest(folder)
            code = r'''
import sys,torch
from vision_geometry_lab.centerline_model_v5 import inspect_manifest,masked_two_head_loss
from vision_geometry_lab.centerline_context_v6 import rasterize_letterbox_record
from vision_geometry_lab.centerline_transfer_v7 import FeatureProvider,TransferManifestDataset
record=inspect_manifest(sys.argv[1])['usable']['train'][0]
provider=FeatureProvider('zero_control',[32,64],None,None)
sample=TransferManifestDataset([record],provider,True)[0]
reference=rasterize_letterbox_record(record,[32,64])
for key in ('input','body_target','center_target','body_valid_mask','center_valid_mask'):
 assert torch.equal(sample[key],reference[key]),key
assert all(sample[f'feature_{i}'].count_nonzero().item()==0 for i in range(3))
body=torch.zeros_like(sample['body_target'][None],requires_grad=True)
center=torch.zeros_like(sample['center_target'][None],requires_grad=True)
loss=masked_two_head_loss({'body_logits':body,'center_logits':center},
 sample['body_target'][None],sample['center_target'][None],
 body_valid_mask=sample['body_valid_mask'][None],
 center_valid_mask=sample['center_valid_mask'][None])['loss']
loss.backward();bv=sample['body_valid_mask'][None]>0;cv=sample['center_valid_mask'][None]>0
assert body.grad[~bv].abs().sum().item()==0 and center.grad[~cv].abs().sum().item()==0
assert body.grad[bv].abs().sum().item()>0 and center.grad[cv].abs().sum().item()>0
'''
            _run(['-c', code, str(manifest)])

    def test_zero_control_cli_trains_exports_and_rejects_v6_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, _, holdout_sample, holdout_label = _write_manifest(root)
            model_dir = root / 'model'
            _run(['-m', 'vision_geometry_lab.centerline_transfer_v7', 'train-manifest',
                  '--manifest', str(manifest), '--output-dir', str(model_dir),
                  '--feature-mode', 'zero_control', '--steps', '1',
                  '--target-height', '32', '--target-width', '64',
                  '--base-channels', '4', '--projected-feature-channels', '3',
                  '--batch-size', '1', '--threads', '1', '--seed', '37',
                  '--cache-samples'])
            report = json.loads((model_dir / 'report.json').read_text('utf-8'))
            self.assertEqual(report['experiment_version'], 7)
            self.assertEqual(report['feature_mode'], 'zero_control')
            self.assertIsNone(report['sam2'])
            self.assertTrue(report['data']['groups_disjoint'])
            self.assertFalse(report['data']['holdout_used_for_model_selection'])
            checkpoint = model_dir / 'checkpoint.pt'
            exports = root / 'exports'
            _run(['-m', 'vision_geometry_lab.centerline_transfer_v7', 'infer-manifest',
                  '--manifest', str(manifest), '--checkpoint', str(checkpoint),
                  '--output-dir', str(exports), '--split', 'holdout', '--threads', '1'])
            metadata = json.loads(
                (exports / 'predictions/holdout_case.json').read_text('utf-8'))
            self.assertEqual(metadata['experiment_version'], 7)
            self.assertEqual(metadata['feature_mode'], 'zero_control')
            self.assertEqual(metadata['model_input_hw'], [32, 64])
            self.assertEqual(metadata['probability_map_hw'], [28, 64])
            self.assertIsNone(metadata['feature_provenance']['source_checkpoint'])
            self.assertIsNone(metadata['feature_provenance']['cache_npz'])
            self.assertTrue(metadata['feature_provenance']['strict_zero_features'])
            self.assertEqual(metadata['feature_provenance']['input_semantics'],
                             'rgb_only_no_seed_no_target')
            self.assertEqual(metadata['trained_checkpoint_sha256'],
                             metadata['checkpoint_sha256'])
            with np.load(exports / 'inputs/holdout_case.npz', allow_pickle=False) as native:
                self.assertEqual(set(native.files), {'rgb', 'seed_heatmap'})
            from vision_geometry_lab.centerline_benchmark_v5 import load_prediction
            from vision_geometry_lab.centerline_review_v5 import validate_annotation
            loaded = load_prediction(exports / 'predictions', holdout_sample,
                                     validate_annotation(holdout_label, holdout_sample))
            self.assertIsNotNone(loaded)

            code = r'''
import sys,torch
from vision_geometry_lab.centerline_transfer_v7 import load_head_checkpoint
payload=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
assert payload['sam2_state_included'] is False
assert not any(key.startswith('sam2') for key in payload['head_state_dict'])
payload['format']='centerline_context_v6_letterbox_two_head'
torch.save(payload,sys.argv[2])
try:load_head_checkpoint(sys.argv[2])
except ValueError as error:assert 'rejects V5/V6' in str(error)
else:raise AssertionError('V6 checkpoint format was accepted')
'''
            _run(['-c', code, str(checkpoint), str(root / 'wrong.pt')])

    def test_factory_sam2_smoke_has_frozen_projected_levels_and_expected_crop(self):
        code = r'''
import torch
from vision_geometry_lab.centerline_transfer_v7 import FrozenSam2Extractor
torch.set_num_threads(2)
extractor=FrozenSam2Extractor(threads=2)
value=torch.zeros(4,512,1024);value[:3]=.25;value[3]=1
features,stats=extractor.extract(value)
assert extractor.checkpoint_sha256=='7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'
assert extractor.parameter_count==38962498 and extractor.trainable_parameter_count==0
assert [list(x.shape) for x in features]==[[32,128,256],[64,64,128],[256,32,64]]
assert stats['feature_bytes_float32']==8*1024*1024
assert stats['forward_seconds']>0 and math.isfinite(stats['forward_seconds'])
assert all(not x.requires_grad and torch.isfinite(x).all() for x in features)
assert all(not parameter.requires_grad for parameter in extractor.model.parameters())
'''
        # Keep this one real: it detects local checkpoint/config/API drift.
        _run(['-c', 'import math\n' + code], timeout=180)


if __name__ == '__main__':
    unittest.main()
