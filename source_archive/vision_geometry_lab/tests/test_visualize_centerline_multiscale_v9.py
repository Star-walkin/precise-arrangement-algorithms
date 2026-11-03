import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from vision_geometry_lab import visualize_centerline_multiscale_v9 as viz


def _row(identifier='fixture',split='train'):
    return dict(sample_id=identifier,split=split,prediction_status='loaded',
        decoded=dict(segments=[[[10,20],[25,20]],[[40,20],[60,20]]],reason='fixture_stop'),
        metrics=dict(downstream=dict(covered_fraction=.4,prediction_precision_at_5px=.8),
                     score_support=dict(full_reference_segments=[[[10,20],[60,20]]])))


def _arm():
    yy,xx=np.indices((40,60))
    return dict(row=_row(),arrays=dict(body=(xx/59).astype(np.float32),
        center=(yy/39).astype(np.float32),coverage=np.ones((40,60),np.uint8)))


def test_representative_selection_is_fixed_and_cannot_replace_missing_row():
    rows=[_row(identifier,split) for _,identifier,split in viz.REPRESENTATIVES]
    rows.append(_row('extra_prettier_sample'))
    selected=viz.fixed_rows(dict(rows=rows))
    assert list(selected)==[item[1] for item in viz.REPRESENTATIVES]
    try:viz.fixed_rows(dict(rows=rows[1:]))
    except ValueError as exc:assert 'fixed representative' in str(exc)
    else:raise AssertionError('Missing representative silently replaced')


def test_detail_crop_depends_only_on_manual_seed_and_reference():
    seed=[[250,240],[300,240]];reference=[[[300,240],[750,380]]]
    assert viz.detail_roi([1000,1200],seed,reference)==[154,144,847,477]
    assert viz.detail_roi([80,120],[[0,0],[10,10]],[])==[0,0,107,80]


def test_probability_crop_uses_exact_half_pixel_mapping():
    ramp=np.tile(np.arange(8,dtype=np.float32),(6,1))
    crop=viz.probability_crop(ramp,[2,2,8,5])
    assert crop.shape==(3,6)
    assert np.allclose(crop[1],[.75,1.25,1.75,2.25,2.75,3.25])


def test_color_scale_is_fixed_not_normalized_per_arm():
    first=viz.probability_rgb(np.array([[0.,.25,.5,1.]],np.float32))
    second=viz.probability_rgb(np.array([[.25,.5]],np.float32))
    assert np.array_equal(first[:,1:3],second)


def test_render_retains_saved_gaps_no_redecode_and_no_input_mutation():
    rgb=np.full((80,120,3),90,np.uint8);control=_arm();native=_arm()
    before=copy.deepcopy(control);references=[[[10,20],[60,20]]]
    image=viz.render_figure(rgb,[[10,20],[20,20]],references,control,native,[0,0,120,80],
                            sample_id='synthetic',split='train',view_name='overview')
    assert image.width==3048 and image.height>1200
    assert control['row']==before['row']
    assert all(np.array_equal(control['arrays'][key],before['arrays'][key]) for key in control['arrays'])
    assert len(control['row']['decoded']['segments'])==2
    assert image.getextrema()!=((25,25),(28,28),(34,34))


def test_missing_export_keeps_explicit_empty_panel_without_hiding_sample():
    control=_arm();native=_arm();native['arrays']=None
    native['row']['prediction_status']='missing_prediction';native['row']['decoded']['segments']=[]
    image=viz.render_figure(np.zeros((80,120,3),np.uint8),[[10,20],[20,20]],[[[10,20],[60,20]]],
        control,native,[0,0,120,80],sample_id='synthetic_missing',split='holdout',view_name='detail')
    assert image.width==3048 and not native['row']['decoded']['segments']


def test_existing_output_refuses_before_any_source_read():
    with tempfile.TemporaryDirectory() as folder:
        with patch.object(viz,'_dataset',side_effect=AssertionError('Must reject before reading')):
            try:viz.render_results('unused','unused','unused','unused','unused',folder)
            except FileExistsError:pass
            else:raise AssertionError('Existing output overwritten')


def test_summary_must_match_fixed_protocol_manifest_and_mode():
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'summary.json'
        value=dict(experiment_version=9,mode='native_detail',manifest_sha256='fixture',
            protocol=viz.V9_PROTOCOL,protocol_sha256=viz.canonical_hash(viz.V9_PROTOCOL))
        path.write_text(json.dumps(value),'utf-8')
        assert viz._read_summary(path,'native_detail','fixture')==value
        try:viz._read_summary(path,'lowpass_control','fixture')
        except ValueError:pass
        else:raise AssertionError('Arm mismatch accepted')


def test_full_synthetic_renderer_writes_all_fixed_views_and_keeps_missing_arm():
    records=[];rows=[]
    for _,identifier,split in viz.REPRESENTATIVES:
        records.append(dict(sample=dict(sample_id=identifier),label=dict(polylines=dict(free_seed=[[10,20],[20,20]]))))
        rows.append(_row(identifier,split))
    def arm(folder,record,row,rgb,snapshot,manifest_sha,mode):
        result=_arm();result['row']=row
        if mode=='native_detail':result['arrays']=None
        return result
    with tempfile.TemporaryDirectory() as folder:
        output=Path(folder)/'new_visuals'
        with (patch.object(viz,'_dataset',return_value=({},records,{})),
              patch.object(viz,'_read_summary',return_value=dict(rows=rows)),
              patch.object(viz,'_load_arm',side_effect=arm),
              patch.object(viz,'read_clean_image',return_value=Image.new('RGB',(120,80))),
              patch.object(viz,'file_hash',return_value='synthetic_fixture')):
            report=viz.render_results('m','c','n','cp','np',output)
        assert len(report['images'])==10 and len(list(output.glob('*.png')))==10
        assert not report['model_or_decoder_executed'] and not report['force_validation']
        assert (output/'summary.json').exists() and (output/'README.md').exists()
        assert [entry['view'] for entry in report['images']]==['overview','detail']*5
