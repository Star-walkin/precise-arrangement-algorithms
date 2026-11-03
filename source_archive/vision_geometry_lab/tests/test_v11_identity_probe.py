import numpy as np
from vision_geometry_lab.v11_identity_probe import make_pair, assess_pair


def test_identity_pair_changes_only_prompt_not_rgb_or_crop():
    pair = make_pair(19)
    assert np.array_equal(pair['inputs'][0, [0,1,2,3,5,6,7]], pair['inputs'][1, [0,1,2,3,5,6,7]])
    assert not np.array_equal(pair['inputs'][0,4], pair['inputs'][1,4])
    assert np.min(pair['rows'][1]-pair['rows'][0]) > 10


def test_oracle_two_queries_pass_but_same_path_for_both_fails():
    pair = make_pair(120, test=True)
    oracle = np.stack([t['target_distribution'] for t in pair['targets']])
    assert assess_pair(oracle, pair)['both_queries_pass']
    failed = assess_pair(np.repeat(oracle[:1], 2, axis=0), pair)
    assert not failed['both_queries_pass']
    assert failed['wrong_instance_fraction'][1] == 1


def test_probe_is_deterministic_and_not_field_data():
    a, b = make_pair(80), make_pair(80)
    assert np.array_equal(a['inputs'], b['inputs'])
    assert a['kind'] == 'generated_identity_probe_not_field_geometry'
