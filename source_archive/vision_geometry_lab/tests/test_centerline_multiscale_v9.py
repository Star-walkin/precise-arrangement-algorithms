from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = ROOT / "vision_training/.venv/Scripts/python.exe"
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from vision_geometry_lab.centerline_multiscale_contract_v9 import (
    LOCAL_MODEL_SIZE,
    array_sha256,
    blend_weight_map,
    build_tile_layout,
    expected_coverage_mask,
    rebuild_input_contract,
    tile_in_frame_mask,
)


def _torch_available():
    if not TRAIN_PY.is_file():
        return False
    try:
        result = subprocess.run(
            [str(TRAIN_PY), "-c", "import torch"], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _run_training_python(arguments, timeout=180):
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    result = subprocess.run(
        [str(TRAIN_PY), *arguments], cwd=ROOT, env=environment,
        text=True, capture_output=True, timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise AssertionError(
            f"command failed ({result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def _high_frequency_image(height=768, width=1024):
    yy, xx = np.mgrid[:height, :width]
    return np.stack([
        40+180*((xx+yy) % 2),
        50+150*((xx//3+yy//5) % 2),
        60+120*((xx//7) % 2),
    ], axis=2).astype(np.uint8)


class CenterlineMultiscaleContractV9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = _high_frequency_image()
        cls.seed = [[60, 390], [120, 390], [220, 390]]

    def test_contract_changes_only_local_rgb_and_is_rebuildable_without_torch(self):
        native = rebuild_input_contract(self.image, self.seed, "native_detail")
        control = rebuild_input_contract(self.image, self.seed, "lowpass_control")
        self.assertEqual(native["global_input"].shape, (4, 256, 384))
        self.assertEqual(native["local_inputs"].shape, (9, 4, 384, 384))
        self.assertTrue(np.array_equal(native["global_input"], control["global_input"]))
        self.assertTrue(np.array_equal(native["local_inputs"][:, 3],
                                       control["local_inputs"][:, 3]))
        self.assertFalse(np.array_equal(native["local_inputs"][:, :3],
                                        control["local_inputs"][:, :3]))
        self.assertEqual(native["tile_layout"], control["tile_layout"])
        self.assertEqual(native["input_contract"]["global_input_sha256"],
                         control["input_contract"]["global_input_sha256"])
        self.assertNotEqual(native["input_contract"]["local_inputs_sha256"],
                            control["input_contract"]["local_inputs_sha256"])
        for result in (native, control):
            self.assertEqual(result["input_contract"]["expected_coverage_mask_sha256"],
                             array_sha256(result["expected_coverage_mask"]))
            coverage_values = set(np.unique(result["expected_coverage_mask"]))
            self.assertTrue(coverage_values.issubset({0, 1}))
            self.assertIn(1, coverage_values)
            self.assertEqual(result["expected_coverage_mask"].shape, (384, 512))

    def test_seed_only_layout_is_even_ordered_and_padding_coverage_is_exact(self):
        layout = build_tile_layout(self.seed, self.image.shape[:2])
        self.assertEqual(len(layout["tiles"]), 9)
        self.assertEqual([(tile["u_px"], tile["v_px"]) for tile in layout["tiles"]],
                         [(u, v) for u in (0, 512, 1024) for v in (-384, 0, 384)])
        for tile in layout["tiles"]:
            left, top, right, bottom = tile["native_xyxy"]
            self.assertEqual(left % 2, 0)
            self.assertEqual(top % 2, 0)
            self.assertEqual((right-left, bottom-top), (768, 768))
            local = tile_in_frame_mask(tile, self.image.shape[:2])
            self.assertEqual(local.shape, (LOCAL_MODEL_SIZE, LOCAL_MODEL_SIZE))
            self.assertTrue(set(np.unique(local)).issubset({0, 1}))
        coverage = expected_coverage_mask(layout, self.image.shape[:2])
        rebuilt = np.zeros_like(coverage)
        for tile in layout["tiles"]:
            left, top, _, _ = tile["native_xyxy"]
            mask = tile_in_frame_mask(tile, self.image.shape[:2])
            gx0, gy0 = left//2, top//2
            for y, x in zip(*np.nonzero(mask)):
                gx, gy = gx0+x, gy0+y
                if 0 <= gx < rebuilt.shape[1] and 0 <= gy < rebuilt.shape[0]:
                    rebuilt[gy, gx] = 1
        self.assertTrue(np.array_equal(coverage, rebuilt))
        weight = blend_weight_map()
        self.assertGreater(float(weight.min()), 0.)
        self.assertEqual(weight.shape, (384, 384))

    def test_contract_rejects_non_seed_inputs_or_invalid_geometry(self):
        with self.assertRaises(ValueError):
            rebuild_input_contract(self.image, self.seed, "unknown")
        with self.assertRaises(ValueError):
            rebuild_input_contract(self.image, [[1, 1], [1, 1]], "native_detail")
        with self.assertRaises(ValueError):
            rebuild_input_contract(self.image[:-1], self.seed, "native_detail")


@unittest.skipUnless(_torch_available(), "vision_training Python with torch is unavailable")
class CenterlineMultiscaleTorchV9Tests(unittest.TestCase):
    def test_preclipped_targets_and_loss_are_invariant_to_unknown_raw_tail_in_both_arms(self):
        code = r'''
import hashlib,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from vision_geometry_lab.centerline_multiscale_v9 import (PreparedRecordCache,prepare_record,
 masked_two_head_loss)

H,W=768,1024
yy,xx=np.mgrid[:H,:W]
rgb=np.stack([50+180*((xx+yy)%2),60+150*((xx//3+yy//5)%2),np.full_like(xx,100)],2).astype(np.uint8)
trusted={'centerline':np.array([[120.,390.],[220.,390.],[400.,380.],[650.,330.]]),
 'edge_flange':np.array([[120.,384.],[220.,384.],[400.,374.],[650.,324.]]),
 'edge_surface':np.array([[120.,396.],[220.,396.],[400.,386.],[650.,336.]])}
def make(path,tail):
 raw={k:np.vstack([v,tail+np.array([0.,(-6 if k=='edge_flange' else 6 if k=='edge_surface' else 0)])]) for k,v in trusted.items()}
 raw['free_seed']=np.array([[60.,390.],[120.,390.],[220.,390.]])
 return dict(sample_id='s',group_id='g',split='train',image_path=str(path),
  image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),image_shape_hw=[H,W],roi_xyxy=[0,0,W,H],
  label={'status':'partial'},parsed_polylines=raw,
  parsed_supervision_polylines={k:v.copy() for k,v in trusted.items()},parsed_ignore_polygons=[],
  parsed_quality_exclude_polygons=[np.array([[700.,0.],[1023.,0.],[1023.,767.],[700.,767.]])])
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'im.png';Image.fromarray(rgb,'RGB').save(p)
 first=make(p,np.array([900.,250.]));second=make(p,np.array([900.,650.]))
 prepared={}
 for mode in ('native_detail','lowpass_control'):
  a=prepare_record(first,mode);b=prepare_record(second,mode)
  assert torch.equal(a['global_input'],b['global_input'])
  assert torch.equal(a['local_inputs'],b['local_inputs'])
  for index in range(9):
   for key in ('body_target','center_target','body_valid_mask','center_valid_mask'):
    assert torch.equal(a['supervision'][index][key],b['supervision'][index][key]),(mode,index,key)
  prepared[mode]=a
 # Optional caching is a performance-only switch and cannot change tensors.
 uncached=PreparedRecordCache([first],'native_detail',cache_samples=False).get(0)
 cached_loader=PreparedRecordCache([first],'native_detail',cache_samples=True)
 cached=cached_loader.get(0)
 assert torch.equal(uncached['global_input'],cached['global_input'])
 assert torch.equal(uncached['local_inputs'],cached['local_inputs'])
 assert cached_loader.get(0) is cached and cached_loader.cache_bytes()>0
 for index in range(9):
  for key in ('body_target','center_target','body_valid_mask','center_valid_mask'):
   assert torch.equal(prepared['native_detail']['supervision'][index][key],
                      prepared['lowpass_control']['supervision'][index][key])
 # A quality/padding-invalid pixel contributes exactly zero gradient.
 tile=prepared['native_detail']['supervision'][0]
 body=torch.zeros_like(tile['body_target'][None],requires_grad=True)
 center=torch.zeros_like(tile['center_target'][None],requires_grad=True)
 loss=masked_two_head_loss({'body_logits':body,'center_logits':center},
  tile['body_target'][None],tile['center_target'][None],
  body_valid_mask=tile['body_valid_mask'][None],center_valid_mask=tile['center_valid_mask'][None])['loss']
 assert torch.isfinite(loss);loss.backward()
 assert body.grad[tile['body_valid_mask'][None]==0].abs().sum().item()==0
 assert center.grad[tile['center_valid_mask'][None]==0].abs().sum().item()==0
 assert any(float(t['body_valid_mask'].sum()+t['center_valid_mask'].sum())==0
            for t in prepared['native_detail']['supervision'])
'''
        _run_training_python(["-c", code])

    def test_model_initialization_sampler_forward_and_stitch_are_paired(self):
        code = r'''
import numpy as np,torch
from vision_geometry_lab.centerline_multiscale_contract_v9 import build_tile_layout,expected_coverage_mask
from vision_geometry_lab.centerline_multiscale_v9 import (build_model,_state_dict_sha256,
 build_training_sequence,_stitch_tiles,build_parser)
torch.set_num_threads(1)
torch.manual_seed(20260927);first=build_model(4);a=_state_dict_sha256(first.state_dict())
torch.manual_seed(20260927);second=build_model(4);b=_state_dict_sha256(second.state_dict())
assert a==b
records=[{'sample_id':'a'},{'sample_id':'b'}]
assert build_training_sequence(records,5,2,20260927)==build_training_sequence(records,5,2,20260927)
sequence=build_training_sequence(records,9,2,20260927)
assert {item['tile_index'] for batch in sequence for item in batch}==set(range(9))
out=first(torch.rand(2,4,64,96),torch.rand(2,4,65,67))
assert out['body_logits'].shape==out['center_logits'].shape==(2,1,65,67)
layout=build_tile_layout([[60,390],[220,390]],[768,1024])
pred=[(np.full((384,384),index/10,np.float32),np.full((384,384),(9-index)/10,np.float32))
      for index in range(9)]
body,center,coverage=_stitch_tiles(pred,layout,[768,1024])
assert body.shape==center.shape==coverage.shape==(384,512)
assert np.array_equal(coverage,expected_coverage_mask(layout,[768,1024]))
assert np.all(body[coverage==0]==0) and np.all(center[coverage==0]==0)
args=build_parser().parse_args(['train-manifest','--manifest','m.json','--output-dir','o',
 '--mode','native_detail','--cache-samples'])
assert args.steps==300 and args.seed==20260927 and args.threads==2 and args.batch_size==2 and args.base_channels==8
assert args.cache_samples is True
'''
        _run_training_python(["-c", code])

    def test_synthetic_cli_runs_both_arms_without_saving_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"smoke"
            _run_training_python([
                "-m", "vision_geometry_lab.centerline_multiscale_v9", "synthetic-smoke",
                "--output-dir", str(output), "--steps", "1", "--threads", "1",
                "--batch-size", "1", "--base-channels", "2", "--seed", "20260927",
            ], timeout=240)
            report = json.loads((output/"report.json").read_text("utf-8"))
            self.assertEqual(report["status"], "synthetic_mechanical_smoke_complete")
            self.assertTrue(report["synthetic_only"])
            self.assertTrue(all(report["paired_checks"].values()))
            self.assertEqual(report["arms"]["native_detail"]["initial_state_sha256"],
                             report["arms"]["lowpass_control"]["initial_state_sha256"])
            self.assertEqual(report["arms"]["native_detail"]["global_input_sha256"],
                             report["arms"]["lowpass_control"]["global_input_sha256"])
            self.assertNotEqual(report["arms"]["native_detail"]["local_inputs_sha256"],
                                report["arms"]["lowpass_control"]["local_inputs_sha256"])
            self.assertFalse((output/"checkpoint.pt").exists())
            self.assertFalse(report["claims"]["field_accuracy_established"])


if __name__ == "__main__":
    unittest.main()
