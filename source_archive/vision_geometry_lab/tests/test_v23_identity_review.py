from PIL import Image
from vision_geometry_lab.v23_identity_review import schedules, sheet, ROIS, SAVE_NATIVE


def test_schedules_keep_uniform_and_development_separate():
    for side in ('left','right'):
        s=schedules(side)
        assert s['uniform']==tuple(range(0,61,5))
        assert all(t in set(s['uniform'])|set(s['development_near_edge']) for t in SAVE_NATIVE[side])
    assert schedules('left')['development_near_edge']==tuple(range(20,36))
    assert schedules('right')['development_near_edge']==tuple(range(8,21))


def test_fixed_raw_rois_valid_for_original_frames():
    for x0,y0,x1,y1 in ROIS.values():
        assert 0<=x0<x1<=4096 and 0<=y0<y1<=3000


def test_contact_sheet_empty_slots_not_filled_by_repeated_frames():
    picture=Image.new('RGB',(100,100),'red')
    result=sheet([dict(time_s=0,source_frame=0,full=picture,crop=picture)],'left','uniform',1,1)
    assert result.size==(2000,1450)
    assert result.getpixel((1500,1000))==(21,28,37)
