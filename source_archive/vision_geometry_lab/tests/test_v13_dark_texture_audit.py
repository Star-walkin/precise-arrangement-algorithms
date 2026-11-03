import numpy as np

from vision_geometry_lab.v13_dark_texture_audit import (
    arc_geometry, geometry_at, patches_at, paired_summary)


def scene():
    curves=dict(centerline=[[20,70],[480,70]],edge_flange=[[20,60],[480,60]],
                edge_surface=[[20,80],[480,80]])
    points,positions=arc_geometry(curves['centerline'])
    return curves,points,positions


def test_real_intersections_and_no_extrapolation():
    curves,points,positions=scene()
    g,reason=geometry_at(curves,points,positions,200,10)
    assert reason is None and g['width']==20
    assert geometry_at(curves,points,positions,-1,10)[1]=='unannotated_center_extension'
    curves['edge_surface']=[[20,80],[40,80]]
    assert geometry_at(curves,points,positions,200,10)[0] is None


def test_curvilinear_patches_sample_equal_trusted_support():
    curves,points,positions=scene()
    image=np.tile(np.arange(160,dtype=np.uint8)[:,None],(1,520))
    unknown=np.zeros_like(image)
    data,reason=patches_at(image,unknown,curves,points,positions,200,20)
    assert reason is None
    assert len(data['patches'])==5 and data['patches'][0].shape==(129,33)
    assert [int(p[64,16]) for p in data['patches']]==[60,65,70,75,80]
    assert patches_at(image,unknown,curves,points,positions,10,20)[0] is None
    unknown[60,220]=255
    assert patches_at(image,unknown,curves,points,positions,200,20)[1]=='patch_intersects_annotated_unknown'


def test_paired_summary_retains_unknown_not_false():
    rows=[dict(sample_id='a',available=True,candidates=[dict(alpha=a,metrics={'x':v})
        for a,v in [(0.,9),(.25,1),(.5,2),(.75,None),(1.,2)]])]
    result=paired_summary(rows)[0]
    assert result['paired_comparisons']==2 and result['missing_feature_pairs']==1
    assert result['pooled_win_fraction']==.75
    assert result['macro_image_win_fraction']==.75
