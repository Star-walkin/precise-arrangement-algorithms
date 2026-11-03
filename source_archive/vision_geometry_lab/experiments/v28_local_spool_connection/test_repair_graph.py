"""Synthetic graph regressions; source sections are fixtures, never field truth."""
from __future__ import annotations

import unittest

import numpy as np

from . import connections_repair as connection


def source_node(u, v, body=True):
    xy = np.array([40.+u, 100.+v])
    return dict(v_px=float(v), width_px=10., quality=.99,
                kind='body' if body else 'groove', strict_body=bool(body),
                raw_pair_supported=True,
                left_xy=(xy+[0., -5.]).tolist(), right_xy=(xy+[0., 5.]).tolist())


def graph_fixture():
    image = np.full((320, 600, 3), 128, np.uint8)
    search = np.zeros(image.shape[:2], bool)
    search[80:300, :201] = True
    search[120:300, 201:] = True
    regions = dict(search_mask=search, explicit_background=np.zeros_like(search), spool_mask=search)
    sections = []
    for u in np.arange(4., 404., 4.):
        nodes = []
        for slope in (0., .5, 1., 1.5):
            if slope == 0 and u > 160:
                continue
            v = min(80., max(0., u-20.))*slope
            nodes.append(source_node(u, v))
        sections.append(dict(u_px=float(u), nodes=nodes))
    return image, dict(sections=sections, structures=[]), regions


class RepairGraphTests(unittest.TestCase):
    def test_finite_pool_preserves_short_distinct_terminal_not_only_longest_routes(self):
        image, bands, regions = graph_fixture()
        pool = connection.generate_pool(image, [[10., 100.], [40., 100.]], 10., bands, regions,
            dict(maximum_candidates=3, terminal_every_px=80., beam_size=40))
        self.assertLessEqual(len(pool['candidates']), 3)
        self.assertTrue(any(c['original_section_u_px'][-1] <= 200. and
                            abs(c['points_xy'][-1][1]-100.) < 12.
                            for c in pool['candidates']),
                        'The distinct visible route ending at u=160 must not be discarded just because three rivals extend to u=400')

    def test_join_is_measured_connection_location_not_identical_free_origin(self):
        image, bands, regions = graph_fixture()
        pool = connection.generate_pool(image, [[10., 100.], [40., 100.]], 10., bands, regions,
            dict(maximum_candidates=1, terminal_every_px=80., beam_size=20))
        self.assertTrue(pool['candidates'])
        for candidate in pool['candidates']:
            self.assertGreater(np.linalg.norm(np.asarray(candidate['join_xy'])-[40., 100.]), 0.,
                               'Old G0 distance would be identically zero for all entry-as-join candidates')

    def test_grooves_and_missing_sections_never_acquire_strict_body_support(self):
        image = np.full((220, 260, 3), 128, np.uint8)
        region = np.ones(image.shape[:2], bool)
        regions = dict(search_mask=region, spool_mask=region, explicit_background=~region)
        items = []
        for index, kind in enumerate(('missing', 'body', 'body', 'missing', 'body', 'body', 'groove', 'groove', 'groove')):
            u = index*4.
            node = None if kind == 'missing' else source_node(u, 0., body=kind == 'body')
            items.append(dict(u=u, v=0., width=10., node=node, unavailable=kind == 'missing'))
        candidate = connection._candidate(image, dict(items=items, cost=1.), np.array([40., 100.]),
            np.array([1., 0.]), 10., regions, 0, dict(structures=[]), connection.DEFAULT_CONFIG)
        self.assertFalse(any(candidate['strict_body_supported']))
        for supported, inferred, unavailable in zip(candidate['supported'], candidate['inferred'], candidate['unavailable']):
            self.assertEqual(int(supported)+int(inferred)+int(unavailable), 1)
        self.assertIsNone(candidate['left_xy'][3])
        self.assertIsNone(candidate['right_xy'][3])
        self.assertTrue(candidate['raw_pair_supported'][6])


if __name__ == '__main__':
    unittest.main()
