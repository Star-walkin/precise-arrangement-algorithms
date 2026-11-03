import unittest
import numpy as np
from vision_geometry_lab.v11_path_contract import _geometry,strip_to_image
from vision_geometry_lab.v24_candidate_support_audit import audit,relaxed_prefix,decoded_column_rows


class CandidateAuditTests(unittest.TestCase):
    def setUp(self):
        self.geometry=_geometry((3000,4096),[[50,100],[100,100]],16.)
        self.valid=np.ones((256,512),bool)
        self.target=np.full(512,127.5)
        self.trusted=np.zeros(512,bool);self.trusted[40:80]=True
        yy=np.arange(256)[:,None]
        self.prob=np.repeat(np.exp(-.5*((yy-127.5)/1.)**2),512,axis=1)
        self.prob/=self.prob.sum(axis=0)

    def test_original_probabilities_unchanged(self):
        before=self.prob.copy()
        out=audit(self.prob,self.valid,self.geometry,self.target,self.trusted,[])
        np.testing.assert_array_equal(before,self.prob)
        self.assertEqual(out['summary']['all_trusted']['original_node_hit'],40)
        self.assertEqual(out['summary']['all_trusted']['saved_decoder_hit'],0)
        self.assertFalse(out['nodes_injected'])

    def test_gt_cannot_inject_missing_probability_node(self):
        target=np.full(512,155.)
        out=audit(self.prob,self.valid,self.geometry,target,self.trusted,[])
        self.assertEqual(out['summary']['all_trusted']['original_node_hit'],0)

    def test_unknown_gt_not_negative(self):
        target=self.target.copy();target[~self.trusted]=np.nan
        out=audit(self.prob,self.valid,self.geometry,target,self.trusted,[])
        self.assertEqual(out['summary']['all_trusted']['trusted_columns'],40)

    def test_saved_segments_not_bridged(self):
        seg1=strip_to_image(np.array([[40,127.5],[50,127.5]]),self.geometry).tolist()
        seg2=strip_to_image(np.array([[60,127.5],[70,127.5]]),self.geometry).tolist()
        rows=decoded_column_rows([seg1,seg2],self.geometry)
        self.assertFalse(rows[55]);self.assertTrue(rows[45])

    def test_relaxed_graph_two_missing_allowed_three_not(self):
        nodes=[[{'row':127.5}],[],[],[{'row':127.5}],[],[],[],[{'row':127.5}]]
        ok,_=relaxed_prefix(nodes,np.full(8,127.5),np.ones(8,bool),3.,0)
        self.assertTrue(ok[3]);self.assertFalse(ok[7])

    def test_gt_wrong_branch_disallowed_without_rewriting_nodes(self):
        nodes=[[{'row':127.5}],[{'row':128.5}],[{'row':129.5}],[{'row':130.5}]]
        target=np.array([127.5,127.5,127.5,127.5])
        ok,_=relaxed_prefix(nodes,target,np.ones(4,bool),3.,0)
        self.assertTrue(ok[1]);self.assertFalse(ok[2]);self.assertFalse(ok[3])
        self.assertEqual(nodes[3][0]['row'],130.5)

    def test_graph_unknown_gap_is_not_scored_identity(self):
        nodes=[[{'row':127.5}],[{'row':128.5}],[{'row':129.5}]]
        target=np.array([127.5,np.nan,129.5]);trusted=np.array([1,0,1],bool)
        ok,_=relaxed_prefix(nodes,target,trusted,3.,0)
        self.assertTrue(ok[2])

    def test_no_anchor_never_reachable(self):
        nodes=[[{'row':170.}],[{'row':170.}]]
        ok,_=relaxed_prefix(nodes,np.array([170.,170.]),np.ones(2,bool),3.,0)
        self.assertFalse(ok.any())

    def test_no_scored_columns_null_denominator(self):
        out=audit(self.prob,self.valid,self.geometry,np.full(512,np.nan),np.zeros(512,bool),[])
        self.assertIsNone(out['summary']['all_trusted']['fractions']['original_node_hit'])

    def test_shape_invalid_distribution_rejected(self):
        with self.assertRaises(ValueError):audit(self.prob*2,self.valid,self.geometry,self.target,self.trusted,[])


if __name__=='__main__':unittest.main()
