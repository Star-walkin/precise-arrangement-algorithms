import unittest
import numpy as np

from vision_geometry_lab.v10_input_interventions import intervene_inputs, difference_summary


class InputInterventionV10Tests(unittest.TestCase):
    def test_seed_zero_preserves_rgb_and_does_not_mutate_original(self):
        global_input = np.arange(4*3*5).reshape(4, 3, 5).astype(np.float32)
        local = np.arange(9*4*3*5).reshape(9, 4, 3, 5).astype(np.float32)
        original_global, original_local = global_input.copy(), local.copy()
        changed_global, changed_local, descriptor_zero = intervene_inputs(global_input, local, 'seed_channels_zero')
        self.assertFalse(descriptor_zero)
        self.assertTrue(np.array_equal(changed_global[:3], original_global[:3]))
        self.assertTrue(np.array_equal(changed_local[:, :3], original_local[:, :3]))
        self.assertFalse(changed_global[3].any())
        self.assertFalse(changed_local[:, 3].any())
        self.assertTrue(np.array_equal(global_input, original_global))
        self.assertTrue(np.array_equal(local, original_local))

    def test_descriptor_intervention_leaves_seed_inputs_unchanged(self):
        global_input = np.ones((4, 2, 3), np.float32)
        local = np.ones((9, 4, 2, 3), np.float32)
        for condition, flag in (('baseline', False), ('global_descriptor_zero', True)):
            g, l, zero = intervene_inputs(global_input, local, condition)
            self.assertEqual(zero, flag)
            self.assertTrue(np.array_equal(g, global_input))
            self.assertTrue(np.array_equal(l, local))
        with self.assertRaises(ValueError):
            intervene_inputs(global_input, local, 'unknown')

    def test_differences_report_full_and_supported_grid_separately(self):
        result = difference_summary(np.array([[0., 1.], [0., 0.]]), np.zeros((2, 2)),
                                    np.array([[0, 1], [0, 0]]))
        self.assertEqual(result['full_grid_mean_abs'], .25)
        self.assertEqual(result['full_grid_max_abs'], 1.)
        self.assertEqual(result['covered_grid_mean_abs'], 1.)


if __name__ == '__main__':
    unittest.main()
