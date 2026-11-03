import numpy as np
import cv2
import pytest

from vision_geometry_lab.v22_diagnostics import table_count,validate_axis,crop_with_outer_label,bucket_summary
from vision_geometry_lab.v22_diagnostics import array_pixel_hash,verify_lossless_replay


def test_ragged_table_uses_columns_and_original_rows():
    columns=['persistent','original_row_index','time_index']
    table=np.array([[1,66,0],[0,89,0],[1,89,1]],float)
    np.testing.assert_array_equal(table_count(table,columns,[0,.1],[66,89]),[[1,1],[0,1]])
    np.testing.assert_array_equal(table_count(table,columns,[0,.1],[66,89],'persistent'),[[1,0],[0,1]])
    with pytest.raises(ValueError,match='outside'):
        table_count(np.array([[1,67,0]]),columns,[0,.1],[66,89])


def test_axis_missing_mismatch_and_pixel_hash_fail():
    source=[dict(frame_index=0,time_s=0.,frame_sha256='a'),dict(frame_index=1,time_s=.1,frame_sha256='b')]
    validate_axis(source,source,hash_key='frame_sha256')
    with pytest.raises(ValueError,match='incomplete'):validate_axis(source[:1],source)
    with pytest.raises(ValueError,match='frame/time'):validate_axis(source[::-1],source)
    bad=[dict(source[0],frame_sha256='wrong'),source[1]]
    with pytest.raises(ValueError,match='pixel hash'):validate_axis(bad,source,hash_key='frame_sha256')


def test_native_crop_label_is_outside_no_flip_or_enhancement():
    raw=np.arange(100*100*3,dtype=np.uint8).reshape(100,100,3)
    crop,labelled=crop_with_outer_label(raw,[3,4,53,43],273,27.3)
    assert crop.size==(51,40)
    assert labelled.width%2==labelled.height%2==0
    np.testing.assert_array_equal(np.asarray(crop),raw[4:44,3:54,::-1])
    np.testing.assert_array_equal(np.asarray(labelled)[36:76,:51],np.asarray(crop))


def test_final_bucket_keeps_60_and_unknown_not_edge_absence():
    times=np.arange(601)/10
    counts={key:np.ones((601,24),int) for key in ('all_edges','bands','geometric_persistent','qualified_persistent')}
    data=dict(arrays=dict(times_s=times,usable=np.zeros((601,24),bool)),counts=counts)
    result=bucket_summary(data)
    assert result[-1]['frames']==51 and sum(r['frames'] for r in result)==601
    assert all(r['usable_row_cells']==0 for r in result)
    assert result[0]['multiple_edge_cells']==0
    assert result[-1]['band_cells']==51*24


def test_ffv1_roundtrip_checks_every_pixel_and_rejects_wrong_hash(tmp_path):
    path=tmp_path/'test.avi'
    frames=[np.random.default_rng(i).integers(0,256,(40,52,3),dtype=np.uint8) for i in range(3)]
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),10.,(52,40))
    assert writer.isOpened(), 'FFV1 required for lossless export'
    try:
        for frame in frames:writer.write(frame)
    finally:writer.release()
    hashes=[array_pixel_hash(frame) for frame in frames]
    verify_lossless_replay(path,hashes,10.)
    with pytest.raises(ValueError,match='pixel mismatch'):
        verify_lossless_replay(path,[hashes[0],'wrong',hashes[2]],10.)
