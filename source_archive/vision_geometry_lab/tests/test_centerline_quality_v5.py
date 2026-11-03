import hashlib
import json
from pathlib import Path
import tempfile

from vision_geometry_lab.centerline_quality_v5 import load_label_quality, quality_path


def _fixture(directory):
    annotation=Path(directory)/'sample.json'
    annotation.write_text('{"original":"do not edit"}',encoding='utf-8')
    sample=dict(annotation_path=str(annotation),sample_id='sample',
                image_sha256='a'*64,image_shape_hw=[50,100])
    policy=dict(schema_version=1,sample_id='sample',image_sha256='a'*64,
                annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
                coordinate_system='full_image_xy',regions=[dict(
                    region_id='tail',kind='coarse_direction_only',exclude_from_precision=True,
                    polygon_xy=[[60,10],[90,10],[90,40],[60,40]])])
    return annotation,sample,policy


def _write(sample,policy):
    quality_path(sample).write_text(json.dumps(policy),encoding='utf-8')


def _reject(sample,fragment):
    try: load_label_quality(sample)
    except ValueError as error: assert fragment in str(error)
    else: raise AssertionError('Bad quality sidecar was accepted')


def test_quality_missing_sidecar_has_no_exclusion_or_hash():
    with tempfile.TemporaryDirectory() as directory:
        annotation,sample,policy=_fixture(directory)
        assert load_label_quality(sample)==dict(quality_sha256=None,
                                                precision_exclude_polygons=[],regions=[])


def test_quality_preserves_label_and_hash_binds_regions():
    with tempfile.TemporaryDirectory() as directory:
        annotation,sample,policy=_fixture(directory)
        before=annotation.read_bytes()
        _write(sample,policy)
        result=load_label_quality(sample)
        assert result['precision_exclude_polygons']==[policy['regions'][0]['polygon_xy']]
        assert result['quality_sha256']==hashlib.sha256(quality_path(sample).read_bytes()).hexdigest()
        assert annotation.read_bytes()==before


def test_quality_changed_annotation_requires_review_again():
    with tempfile.TemporaryDirectory() as directory:
        annotation,sample,policy=_fixture(directory);_write(sample,policy)
        annotation.write_text('{"original":"changed"}',encoding='utf-8')
        _reject(sample,'annotation_sha256')


def test_quality_rejects_source_mismatch_and_unknown_semantics():
    with tempfile.TemporaryDirectory() as directory:
        annotation,sample,policy=_fixture(directory)
        _write(sample,dict(policy,image_sha256='b'*64));_reject(sample,'image_sha256')
        _write(sample,dict(policy,sample_id='other'));_reject(sample,'sample_id')
        _write(sample,dict(policy,coordinate_system='roi_xy'));_reject(sample,'coordinates')
        policy['regions'][0]['kind']='precise_truth'
        _write(sample,policy);_reject(sample,'quality kind')


def test_quality_rejects_nonfinite_outside_or_degenerate_regions():
    with tempfile.TemporaryDirectory() as directory:
        annotation,sample,policy=_fixture(directory)
        for polygon,fragment in (([[0,0],[float('nan'),0],[4,4]],'finite'),
                                 ([[0,0],[100,0],[4,4]],'outside'),
                                 ([[1,1],[2,2],[3,3]],'zero area')):
            policy['regions'][0]['polygon_xy']=polygon
            _write(sample,policy);_reject(sample,fragment)


def test_quality_cannot_promote_uncertain_regions_to_precision():
    with tempfile.TemporaryDirectory() as directory:
        annotation,sample,policy=_fixture(directory)
        policy['regions'][0]['exclude_from_precision']=False
        _write(sample,policy);_reject(sample,'excluded from precision')
