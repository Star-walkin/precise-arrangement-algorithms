import tempfile
import unittest
from pathlib import Path

import numpy as np

from vision_geometry_lab import v22_pair_continuity_audit as audit
from vision_geometry_lab import v22_local_edges as v22


def fixture(flags,missing=(),changed_edge_at=None):
    times=np.arange(len(flags))/10.;frames=np.arange(len(flags),dtype=np.int64)
    columns={n:i for i,n in enumerate(v22.PAIR_COLUMNS)};rows=[];old_id=0;prev=None
    for i,flag in enumerate(flags):
        if i in missing:prev=None;continue
        edge=20 if changed_edge_at is not None and i>=changed_edge_at else 10
        identity=(edge,flag)
        if identity!=prev:old_id+=1
        prev=identity
        row=np.zeros(len(columns),float)
        for key,value in dict(time_index=i,original_row_index=66,row_pair_track_id=old_id,
            left_edge_track_id=edge,right_edge_track_id=edge+1,bright_1_dark_minus1=1,
            width_px=12.,band_candidate=int(flag)).items():row[columns[key]]=value
        rows.append(row)
    return np.asarray(rows,float).reshape((-1,len(columns))),times,frames


class ContinuityTests(unittest.TestCase):
    def test_qualification_flips_do_not_split_edge_pair_identity(self):
        r=audit.audit_pairs(*fixture([False,True]*5))
        self.assertEqual(len(r['chains']),1);self.assertEqual(r['chains'][0]['observations'],10)
        self.assertEqual(r['chains'][0]['first_available_s'],.4)
        self.assertEqual(r['summary']['restarts_at_qualification_flip'],9)
        self.assertEqual(len(r['qualified_segments']),5)
        self.assertTrue(all(x['observations']==1 and x['first_available_s'] is None for x in r['qualified_segments']))
        self.assertEqual(r['summary']['original_short_fragments_in_flip_affected_now_persistent_chains'],10)

    def test_qualification_continuity_separate_confirmation(self):
        r=audit.audit_pairs(*fixture([False]*4+[True]*6))
        self.assertEqual(len(r['chains']),1);self.assertEqual(len(r['qualified_segments']),1)
        self.assertEqual(r['chains'][0]['first_available_s'],.4)
        self.assertEqual(r['qualified_segments'][0]['first_observed_s'],.4)
        self.assertEqual(r['qualified_segments'][0]['first_available_s'],.8)

    def test_missing_pair_frame_splits_chain_no_interpolation(self):
        r=audit.audit_pairs(*fixture([True]*9,missing=(4,)))
        self.assertEqual(len(r['chains']),2)
        self.assertEqual([x['observations'] for x in r['chains']],[4,4])
        self.assertTrue(all(x['first_available_s'] is None for x in r['chains']))

    def test_missing_source_frame_splits_even_if_old_id_unchanged(self):
        table,times,frames=fixture([True]*6);frames[3:]+=1;times[3:]+=.1
        r=audit.audit_pairs(table,times,frames)
        self.assertEqual(len(r['chains']),2);self.assertEqual([x['observations'] for x in r['chains']],[3,3])

    def test_changed_edge_identity_splits_pair(self):
        r=audit.audit_pairs(*fixture([True]*8,changed_edge_at=4))
        self.assertEqual(len(r['chains']),2)
        self.assertEqual([x['left_edge_track_id'] for x in r['chains']],[10,20])

    def test_changed_polarity_splits(self):
        table,times,frames=fixture([True]*8)
        col={n:i for i,n in enumerate(v22.PAIR_COLUMNS)}
        table[4:,col['bright_1_dark_minus1']]=-1;table[4:,col['row_pair_track_id']]=2
        r=audit.audit_pairs(table,times,frames)
        self.assertEqual(len(r['chains']),2)

    def test_two_rows_do_not_merge_same_edge_ids(self):
        table,times,frames=fixture([True]*6);other=table.copy()
        other[:,v22.PAIR_COLUMNS.index('original_row_index')]=67
        r=audit.audit_pairs(np.vstack([table,other]),times,frames)
        self.assertEqual(len(r['chains']),2);self.assertEqual(r['summary']['edge_pair_chains']['persistent'],2)

    def test_source_prefix_outputs_identical(self):
        args=fixture([False,True,True,False,True,True,True,True,True,False])
        full=audit.audit_pairs(*args)
        for cut in (1,4,6,9):
            partial_table=args[0][args[0][:,0]<cut]
            part=audit.audit_pairs(partial_table,args[1][:cut],args[2][:cut])
            self.assertEqual(full['records'][:cut],part['records'])
            np.testing.assert_equal(full['table'][full['table'][:,0]<cut],part['table'])

    def test_confirm_time_is_observed_not_first_plus_point4(self):
        table,times,frames=fixture([True]*5);times=np.arange(5)*.2
        r=audit.audit_pairs(table,times,frames)
        self.assertEqual(r['chains'][0]['first_available_s'],.8)
        self.assertEqual(r['qualified_segments'][0]['first_available_s'],.8)

    def test_empty_table_reports_zero_not_failure(self):
        table,times,frames=fixture([True]*6,missing=tuple(range(6)))
        r=audit.audit_pairs(table,times,frames)
        self.assertEqual(r['summary']['edge_pair_chains']['total'],0)
        self.assertEqual(r['table'].shape,(0,len(audit.COLUMNS)))

    def test_duplicate_identity_same_frame_rejected(self):
        table,times,frames=fixture([True]*6)
        with self.assertRaises(ValueError):audit.audit_pairs(np.vstack([table,table[0]]),times,frames)

    def test_rewind_and_noninteger_frame_rejected(self):
        table,times,frames=fixture([True]*6);times[2]=times[1]
        with self.assertRaises(ValueError):audit.audit_pairs(table,times,frames)
        table,times,frames=fixture([True]*6)
        with self.assertRaises(ValueError):audit.audit_pairs(table,times,frames.astype(float))

    def test_no_freeze_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileExistsError):audit.freeze(Path(td))


if __name__=='__main__':unittest.main()
