"""Analytic geometry checks for the OPTIONAL local/multi-model audit."""
import unittest
import numpy as np

from lc1_geometry import estimate_turn
from lc1_turn_audit import audit_turn_local


def geometry():
    phi=np.arctan(-.2)
    tangent=np.array([np.cos(phi),np.sin(phi)])
    normal=np.array([-np.sin(phi),np.cos(phi)])
    origin=np.array([250.,400.])
    abc=np.r_[normal,-normal@origin]
    return origin,tangent,normal,abc


def path_from_normal(u,v):
    origin,tangent,normal,abc=geometry()
    return origin+u[:,None]*tangent+v[:,None]*normal,abc


class LocalAuditTests(unittest.TestCase):
    def test_strong_C1_parabola_two_models_two_windows(self):
        u=np.r_[np.linspace(-140,-1,180),np.linspace(0,180,181)]
        points,abc=path_from_normal(u,.008*np.minimum(u,0)**2)
        result=audit_turn_local(points,abc)
        self.assertTrue(result['valid'],result['reason'])
        self.assertLess(abs(result['x']-250.),1.)
        self.assertLessEqual(result['model_disagreement_px'],8.)
        self.assertFalse(result['physical_contact_validated'])
        self.assertFalse(result['anchor_uncertainty_included'])
        self.assertFalse(result['default_estimator_changed'])

    def test_left_free_orientation(self):
        u=np.r_[np.linspace(-180,0,181),np.linspace(1,140,180)]
        points,abc=path_from_normal(u,.008*np.maximum(u,0)**2)
        result=audit_turn_local(points,abc,free_side='left')
        self.assertTrue(result['valid'],result['reason'])
        self.assertLess(abs(result['x']-250.),1.)

    def test_long_circular_arc_point_bias_reduced_but_uncertainty_still_rejects(self):
        u=np.r_[np.linspace(-190,-1,200),np.linspace(0,180,181)]
        radius=250.
        v=np.where(u<0,radius-np.sqrt(radius**2-u**2),0.)
        points,abc=path_from_normal(u,v)
        global_result=estimate_turn(points,pitch_px=20.)
        local=audit_turn_local(points,abc)
        # This is the documented misspecification counterexample: a global
        # quadratic can be self-confident while estimating the wrong turn.
        self.assertTrue(global_result['valid'])
        self.assertGreater(abs(global_result['x']-250.),global_result['uncertainty_px'])
        quadratic=[m for m in local['models'] if m['degree']==2 and m.get('x') is not None]
        self.assertTrue(quadratic)
        self.assertLess(max(abs(m['x']-250.) for m in quadratic),4.)
        # Do not promote accurate point estimates by dropping the cubic model's
        # unresolved transition or widening the existing half-pitch limit.
        self.assertFalse(local['valid'])
        self.assertEqual(local['reason'],'insufficient_two_model_two_window_support')

    def test_straight_and_bad_anchor_are_rejected(self):
        u=np.linspace(-150,180,250)
        points,abc=path_from_normal(u,np.zeros_like(u))
        self.assertFalse(audit_turn_local(points,abc)['valid'])
        abc[-1]+=20.
        self.assertEqual(audit_turn_local(points,abc)['reason'],'far_free_anchor_not_supported')

    def test_configured_model_disagreement_gate_is_not_silently_ignored(self):
        u=np.r_[np.linspace(-140,-1,180),np.linspace(0,180,181)]
        h=np.minimum(u,0)
        points,abc=path_from_normal(u,.008*h*h-.00004*h*h*h)
        result=audit_turn_local(points,abc,max_disagreement_pitch=.10)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'],'local_model_or_window_disagreement')
        self.assertGreater(result['model_disagreement_px'],result['max_model_disagreement_px'])

    def test_spatial_duplicate_density_cannot_create_precision(self):
        u=np.r_[np.linspace(-140,-1,180),np.linspace(0,180,181)]
        points,abc=path_from_normal(u,.008*np.minimum(u,0)**2)
        original=audit_turn_local(points,abc)
        repeated=audit_turn_local(np.repeat(points,5,axis=0),abc)
        self.assertEqual(original['valid'],repeated['valid'])
        self.assertAlmostEqual(original['x'],repeated['x'])
        self.assertAlmostEqual(original['uncertainty_px'],repeated['uncertainty_px'])


if __name__=='__main__':
    unittest.main()
