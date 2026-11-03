import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from vision_geometry_lab.centerline_quality_cutoff_v8 import cutoff_geometry, build_quality_cutoff_payload, validate_trusted_supervision
from vision_geometry_lab.centerline_quality_v5 import load_label_quality, quality_path


_assertions=unittest.TestCase()


def _close_xy(actual,expected):
    return len(actual)==len(expected) and all(math.isclose(a,b,rel_tol=0.,abs_tol=1e-7) for a,b in zip(actual,expected))


def fixture_lines():
    return dict(centerline=[[10+10*i,50] for i in range(8)],
                edge_flange=[[5,45],[30,45],[60,45],[90,45]],
                edge_surface=[[5,55],[40,55],[70,55],[90,55]],
                free_seed=[[10,50],[20,50]])


def test_horizontal_cut_exact_segment_and_vertex_intersections():
    lines=fixture_lines(); before=copy.deepcopy(lines); result=cutoff_geometry(lines,[101,101],7)
    assert lines==before
    assert result['origin_xy']==[70.,50.]
    assert result['unit_tangent_xy']==[1.,0.]
    assert set(map(tuple,result['polygon_xy']))=={(70.,0.),(100.,0.),(100.,100.),(70.,100.)}
    flange=result['edge_intersections']['edge_flange']
    assert flange['xy']==[70.,45.] and math.isclose(flange['segment_fraction'],1/3)
    assert flange['source_segment_index_0based']==2 and not flange['extrapolated']
    assert result['edge_intersections']['edge_surface']['xy']==[70.,55.]
    assert result['retained_polylines']['centerline']==lines['centerline'][:7]
    assert result['retained_polylines']['edge_flange'][-1]==[70.,45.]


def test_rotation_and_mirror_keep_exact_cross_section():
    transforms=[
        lambda p:[100-p[0],p[1]],
        lambda p:[p[1],100-p[0]],
        lambda p:[100-p[0],100-p[1]],
        lambda p:[50+(p[0]-50)*math.sqrt(.5)-(p[1]-50)*math.sqrt(.5),
                  50+(p[0]-50)*math.sqrt(.5)+(p[1]-50)*math.sqrt(.5)],
    ]
    for transform in transforms: _check_rotation_and_mirror(transform)


def _check_rotation_and_mirror(transform):
    base=cutoff_geometry(fixture_lines(),[101,101],7)
    lines={key:[transform(p) for p in values] for key,values in fixture_lines().items()}
    result=cutoff_geometry(lines,[101,101],7)
    assert _close_xy(result['origin_xy'],transform(base['origin_xy']))
    for key in ('edge_flange','edge_surface'):
        assert _close_xy(result['edge_intersections'][key]['xy'],transform(base['edge_intersections'][key]['xy']))
    for p in result['polygon_xy']:
        assert all(0<=v<=100 for v in p)
        assert sum((p[i]-result['origin_xy'][i])*result['unit_tangent_xy'][i] for i in range(2))>=-1e-7


def test_unknown_last_point_does_not_determine_tangent_or_polygon():
    lines=fixture_lines(); first=cutoff_geometry(lines,[101,101],7)
    lines['centerline'][7]=[99,2]
    second=cutoff_geometry(lines,[101,101],7)
    assert first==second  # Later CENTER points are safety-checked, not used for geometry.


def test_reversed_edges_and_terminal_touch_are_not_extrapolated():
    lines=fixture_lines(); lines['edge_flange'].reverse(); lines['edge_surface']=[[5,55],[70,55]]
    result=cutoff_geometry(lines,[101,101],7)
    assert result['edge_intersections']['edge_flange']['input_order_reversed_for_retained_prefix']
    assert result['retained_polylines']['edge_flange']==[[5.,45.],[30.,45.],[60.,45.],[70.,45.]]
    assert result['edge_intersections']['edge_surface']['segment_fraction']==1.


def test_rejects_ambiguous_or_unsafe_geometry():
    cases=[
        ('edge_ends_early','no extrapolation'),('multiple_crossings','multiple crossings'),
        ('collinear_edge','lying in'),('trusted_prefix_crosses','Trusted center prefix'),
        ('tail_returns','Inferred center tail'),('seed_crosses','Free seed'),
        ('degenerate_trusted_segment','degenerate'),('edges_same_side','bracket'),
        ('nonfinite','finite'),('outside','outside'),
    ]
    for kind,fragment in cases: _check_unsafe_geometry(kind,fragment)


def _check_unsafe_geometry(kind,fragment):
    lines=fixture_lines()
    if kind=='edge_ends_early': lines['edge_flange']=[[5,45],[69,45]]
    elif kind=='multiple_crossings': lines['edge_flange']=[[5,45],[80,45],[60,40],[90,40]]
    elif kind=='collinear_edge': lines['edge_flange']=[[5,45],[70,45],[70,46],[90,46]]
    elif kind=='trusted_prefix_crosses': lines['centerline'][2]=[75,50]
    elif kind=='tail_returns': lines['centerline'][7]=[69,90]
    elif kind=='seed_crosses': lines['free_seed'][-1]=[80,50]
    elif kind=='degenerate_trusted_segment': lines['centerline'][6]=lines['centerline'][5][:]
    elif kind=='edges_same_side': lines['edge_surface']=[[5,40],[90,40]]
    elif kind=='nonfinite': lines['edge_flange'][0][0]=float('nan')
    elif kind=='outside': lines['edge_flange'][0][0]=101
    with _assertions.assertRaisesRegex(ValueError,fragment): cutoff_geometry(lines,[101,101],7)


def test_index_is_one_based_trusted_prefix_with_tail():
    for index in (0,1,8,9,True,7.0):
        with _assertions.assertRaisesRegex(ValueError,'one-based'): cutoff_geometry(fixture_lines(),[101,101],index)


def test_hash_bound_payload_is_deterministic_and_old_loader_compatible():
    with tempfile.TemporaryDirectory() as directory:
        _check_hash_bound_payload(Path(directory))


def _check_hash_bound_payload(tmp_path):
    sample=dict(sample_id='example',image_sha256='a'*64,image_shape_hw=[101,101],
                annotation_path=str(tmp_path/'label.json'))
    label=dict(schema_version=1,sample_id='example',image_sha256='a'*64,coordinate_system='full_image_xy',
               status='partial',reviewed=True,polylines=fixture_lines(),ignore_polygons=[],notes='inferred tail')
    raw=(json.dumps(label,indent=3)+'\n').encode('utf-8')
    result=build_quality_cutoff_payload(sample,raw)
    assert result==build_quality_cutoff_payload(sample,raw)
    assert result['quality_payload']['annotation_sha256']==hashlib.sha256(raw).hexdigest()
    assert not list(tmp_path.iterdir())  # Builder itself performs no file IO/writes.
    (tmp_path/'label.json').write_bytes(raw)
    quality_path(sample).write_text(json.dumps(result['quality_payload']),encoding='utf-8')
    assert load_label_quality(sample)['precision_exclude_polygons']==[result['diagnostic']['polygon_xy']]
    assert (tmp_path/'label.json').read_bytes()==raw
    reformatted=json.dumps(label,separators=(',',':')).encode('utf-8')
    assert build_quality_cutoff_payload(sample,reformatted)['quality_payload']['annotation_sha256']!=result['quality_payload']['annotation_sha256']
    label['image_sha256']='b'*64
    with _assertions.assertRaisesRegex(ValueError,'binding'): build_quality_cutoff_payload(sample,json.dumps(label).encode())


def payload_fixture():
    sample=dict(sample_id='example',image_sha256='a'*64,image_shape_hw=[101,101])
    annotation=dict(schema_version=1,sample_id='example',image_sha256='a'*64,
                    coordinate_system='full_image_xy',status='partial',reviewed=True,
                    polylines=fixture_lines(),ignore_polygons=[],notes='confirmed inferred tail')
    payload=build_quality_cutoff_payload(sample,json.dumps(annotation).encode())['quality_payload']
    return sample,annotation,payload


def test_trusted_supervision_returns_targets_only_and_legacy_none():
    sample,annotation,payload=payload_fixture()
    result=validate_trusted_supervision(sample,annotation,payload)
    assert set(result)=={'centerline','edge_flange','edge_surface'}
    assert len(result['centerline'])==7
    assert result['edge_surface'][-1]==[70.,55.]
    result['centerline'][0][0]=999
    assert payload['trusted_supervision_polylines']['centerline'][0][0]==10
    del payload['trusted_supervision_polylines']
    del payload['trusted_supervision_rule']
    assert validate_trusted_supervision(sample,annotation,payload) is None


def test_trusted_supervision_rejects_tampered_payload_or_rule():
    cases=[
        'target','polygon','origin','tangent','tangent_indices','include_seed','missing_target',
        'rule_type','rule_version','rule_extra','rule_region','missing_rule','duplicate_region',
        'missing_region','nonfinite','bool_coordinate','binding','trusted_label_changed',
    ]
    for kind in cases: _check_tampered_payload(kind)


def _check_tampered_payload(kind):
    sample,annotation,payload=payload_fixture(); region=payload['regions'][0]
    if kind=='target': payload['trusted_supervision_polylines']['centerline'][-1][0]+=.01
    elif kind=='polygon': region['polygon_xy'][0][0]+=.01
    elif kind=='origin': region['cutoff_origin_xy'][0]+=.01
    elif kind=='tangent': region['cutoff_unit_tangent_xy']=[0.,1.]
    elif kind=='tangent_indices': region['tangent_source_indices_1based']=[7,8]
    elif kind=='include_seed': payload['trusted_supervision_polylines']['free_seed']=[[10,50],[20,50]]
    elif kind=='missing_target': del payload['trusted_supervision_polylines']['edge_flange']
    elif kind=='rule_type': payload['trusted_supervision_rule']['type']='tail_fitted_line'
    elif kind=='rule_version': payload['trusted_supervision_rule']['version']=True
    elif kind=='rule_extra': payload['trusted_supervision_rule']['use_c8']=True
    elif kind=='rule_region': payload['trusted_supervision_rule']['region_id']='unrelated'
    elif kind=='missing_rule': del payload['trusted_supervision_rule']
    elif kind=='duplicate_region': payload['regions'].append(copy.deepcopy(region))
    elif kind=='missing_region': payload['regions']=[]
    elif kind=='nonfinite': payload['trusted_supervision_polylines']['centerline'][0][0]=float('nan')
    elif kind=='bool_coordinate': payload['trusted_supervision_polylines']['centerline'][0][0]=True
    elif kind=='binding': payload['image_sha256']='b'*64
    elif kind=='trusted_label_changed': annotation['polylines']['centerline'][5][1]+=1
    with _assertions.assertRaises(ValueError): validate_trusted_supervision(sample,annotation,payload)


def test_recomputed_trusted_supervision_never_uses_unknown_tail_geometry():
    sample,annotation,payload=payload_fixture()
    before=validate_trusted_supervision(sample,annotation,payload)
    annotation['polylines']['centerline'][-1]=[99,2]
    assert validate_trusted_supervision(sample,annotation,payload)==before
    # Raw-byte changes must separately fail the parent quality-loader SHA check.


def test_rule_without_trusted_lines_fails_closed():
    sample,annotation,payload=payload_fixture()
    del payload['trusted_supervision_polylines']
    with _assertions.assertRaisesRegex(ValueError,'requires trusted_supervision_polylines'):
        validate_trusted_supervision(sample,annotation,payload)


def test_boundary_policy_uses_requested_center_index():
    result=cutoff_geometry(fixture_lines(),[101,101],6)
    assert result['boundary_policy'].startswith('C6 is the geometric cut;')
    assert result['tangent_from_center_indices_1based']==[5,6]
