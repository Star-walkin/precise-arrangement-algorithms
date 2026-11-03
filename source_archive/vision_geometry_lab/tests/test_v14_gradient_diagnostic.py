"""Synthetic gradient arithmetic only; never load checkpoint or field data."""
import unittest

from vision_geometry_lab.tests.test_v11_path_model import run_torch


class GradientDiagnosticV14Tests(unittest.TestCase):
    def test_positive_opposed_orthogonal_zero_and_unused_gradients(self):
        run_torch('''
import math
import torch
from vision_geometry_lab.v14_gradient_diagnostic import _measure
a=torch.tensor([1.,2.])
r=_measure([a],[2*a])
assert abs(r['cosine']-1.)<1e-12
assert abs(r['primary_gradient_l2']-math.sqrt(5))<1e-12
assert abs(r['weighted_edge_to_primary_norm_ratio']-1.)<1e-12
assert r['shared_gradient_dot_product']==10.
r=_measure([a],[-a]);assert abs(r['cosine']+1.)<1e-12
assert r['weighted_edge_to_primary_norm_ratio']==.5
r=_measure([torch.tensor([1.,0.])],[torch.tensor([0.,1.])])
assert r['cosine']==0. and r['shared_gradient_dot_product']==0.
r=_measure([torch.zeros(2)],[a])
assert r['cosine'] is None and r['weighted_edge_to_primary_norm_ratio'] is None
assert r['primary_zero_norm'] and not r['edge_zero_norm']
r=_measure([a],[torch.zeros(2)])
assert r['cosine'] is None and r['weighted_edge_to_primary_norm_ratio']==0.
assert not r['primary_zero_norm'] and r['edge_zero_norm']
r=_measure([None],[None])
assert r['cosine'] is None and r['primary_zero_norm'] and r['edge_zero_norm']
assert r['primary_unused_parameter_tensors']==1 and r['edge_unused_parameter_tensors']==1
r=_measure([None,torch.tensor([3.])],[torch.tensor([4.]),None])
assert r['cosine']==0. and r['primary_gradient_l2']==3. and r['raw_edge_ce_gradient_l2']==4.
assert abs(r['weighted_edge_to_primary_norm_ratio']-2/3)<1e-12
assert r['primary_unused_parameter_tensors']==1 and r['edge_unused_parameter_tensors']==1
for left,right in (([torch.tensor([float('nan')])],[torch.ones(1)]),
                   ([torch.ones(1)],[torch.tensor([float('inf')])])):
 try: _measure(left,right)
 except ValueError: pass
 else: raise AssertionError('nonfinite gradient accepted')
''')


if __name__ == '__main__':
    unittest.main()
