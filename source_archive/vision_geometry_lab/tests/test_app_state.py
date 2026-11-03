"""State tests without opening a desktop window or any hardware."""
from pathlib import Path
from unittest.mock import patch, call
import queue
import hashlib
import json
import numpy as np
from vision_geometry_lab.app import (App, INITIAL_STATUS_TEXT, OVERLAY_LEGEND_TEXT,
    INVALIDATED_STATUS_TEXT, WOUND_GUIDANCE_SEMANTICS, WOUND_REASON_TRANSLATIONS,
    compose_display_image)


class Field:
    def __init__(self):
        self.value = None
    def set(self,value):
        self.value = value


class Toggle:
    def __init__(self,value):
        self.value = value
    def get(self):
        return self.value


class Canvas:
    source = 'old'
    zoom = 3
    def delete(self,*args):
        pass


class BadVideo:
    def isOpened(self):
        return True
    def get(self,prop):
        return 10
    def set(self,*args):
        pass
    def read(self):
        return False,None
    def release(self):
        pass


def app_without_window():
    app = App.__new__(App)
    app.busy = False
    app.playing = False
    app.version = 0
    app.result = {'old':True}
    app.cap = None
    app.frame = np.ones((40,40,3),np.uint8)
    app.index,app.fps,app.total = 37,10,100
    app.path = Path('old.bmp')
    app.canvas = Canvas()
    app.source_text,app.status = Field(),Field()
    app.arc_text,app.rope_text,app.contact_text,app.wound_text = Field(),Field(),Field(),Field()
    app.arc_roi = app.rope_roi = None
    app.prompts,app.labels,app.wound_prompts = [],[],[]
    app.overlay,app.show_arc,app.show_rope = Toggle(True),Toggle(True),Toggle(True)
    app.show_raw,app.show_exclusion,app.show_conic = Toggle(False),Toggle(False),Toggle(False)
    app.show_wound = Toggle(True)
    return app


def test_failed_new_video_never_relabels_old_pixels():
    app = app_without_window()
    with patch('vision_geometry_lab.app.cv2.VideoCapture',return_value=BadVideo()):
        app.open('new_broken.mp4')
    assert app.frame is None and app.result is None
    assert app.canvas.source is None
    assert app.path == Path('new_broken.mp4') and app.index == 0


def test_failed_frame_never_keeps_previous_measurement():
    app = app_without_window()
    app.cap = BadVideo()
    app.read_frame(45)
    assert app.frame is None and app.result is None
    assert app.canvas.source is None and app.index == 45


def test_reanalysis_discards_previous_result_before_worker_runs():
    app = app_without_window()
    app.arc_roi = app.rope_roi = None
    app.prompts,app.labels = [],[]
    app.render = lambda: None
    class Executor:
        def submit(self,callback):
            pass
    app.executor = Executor()
    app.analyze()
    assert app.result is None and app.busy


def test_worker_error_cannot_make_old_result_saveable():
    app = app_without_window()
    app.arc_roi = app.rope_roi = None
    app.prompts,app.labels = [],[]
    app.render = lambda: None
    app.queue = queue.Queue()
    app.closed = False
    class Root:
        def after(self,*args):
            pass
    class Executor:
        def submit(self,callback):
            callback()
    app.root,app.executor = Root(),Executor()
    with patch('vision_geometry_lab.app.analyze_frame',side_effect=RuntimeError('test_error')):
        app.analyze()
    app.poll()
    assert app.result is None and not app.busy


def test_parameter_change_invalidates_pending_and_saved_results():
    app=app_without_window()
    app.render=lambda:None
    app.playing=True
    previous=app.version
    app.parameter_changed()
    assert app.result is None and not app.playing and app.version>previous
    assert app.status.value.startswith('屏蔽余量已变化')


def test_invalidate_relabels_all_stale_fields_and_status():
    app=app_without_window()
    app.invalidate()
    assert app.result is None
    assert '失效' in app.arc_text.value and '失效' in app.rope_text.value
    assert '尚未生成' in app.contact_text.value and '失效' in app.wound_text.value
    assert app.status.value == INVALIDATED_STATUS_TEXT


def test_save_failure_does_not_claim_complete_bundle():
    app=app_without_window(); app.root=None
    with patch('vision_geometry_lab.app.draw_overlay',return_value=app.frame), \
         patch('vision_geometry_lab.app.write_image',side_effect=OSError('disk_full')), \
         patch('vision_geometry_lab.app.dump_json') as manifest, \
         patch('vision_geometry_lab.app.messagebox.showerror') as error:
        app.save()
    assert not manifest.called and error.called
    assert '保存失败' in app.status.value


def test_save_commits_manifest_after_images_and_then_renames():
    app=app_without_window()
    app.wound_prompts=[[12,13]]
    app.result={'wound_prompts':[[12,13]],'wound_guidance':dict(
        used=True,kind='manual_search_points',points=[[12.,13.]],
        semantics=WOUND_GUIDANCE_SEMANTICS)}
    order=[]
    def image(path,*args):
        assert path.parent.name.endswith('.incomplete')
        order.append(path.name)
    def manifest(path,data):
        assert data['bundle_complete'] is True
        assert data['display_settings']==dict(overlay=True,show_arc=True,show_rope=True,
            show_raw=False,show_exclusion=False,show_conic=False,show_wound=True)
        assert data['wound_guidance']['kind']=='manual_search_points'
        assert data['wound_guidance']['semantics']==WOUND_GUIDANCE_SEMANTICS
        assert set(data['artifact_semantics'])=={'display.png','overlay.png','diagnostics.png','source.png'}
        order.append(path.name)
    with patch('vision_geometry_lab.app.draw_overlay',return_value=app.frame) as draw, \
         patch('vision_geometry_lab.app.write_image',side_effect=image), \
         patch('vision_geometry_lab.app.dump_json',side_effect=manifest), \
         patch('pathlib.Path.rename',side_effect=lambda path:order.append('commit')):
        app.save()
    assert order==['display.png','overlay.png','diagnostics.png','source.png','measurement.json','commit']
    common=dict(show_arc=True,show_rope=True,show_raw=False,show_exclusion=False,
                show_conic=False,show_wound=True)
    diagnostic=dict(show_arc=True,show_rope=True,show_raw=True,show_exclusion=True,
                    show_conic=True,show_wound=True)
    assert draw.call_args_list==[
        call(app.frame,app.result,**common),
        call(app.frame,app.result,**common),
        call(app.frame,app.result,**diagnostic)]
    assert 'bundle_complete' not in app.result and 'display_settings' not in app.result
    assert '已保存' in app.status.value


def test_reference_import_requires_current_clean_image_fingerprint():
    app=app_without_window(); app.root=None
    app.wound_prompts=[]
    app.render=lambda:None
    calls=[]; app.analyze=lambda:calls.append('analyze')
    digest=hashlib.sha256(np.ascontiguousarray(app.frame).tobytes()).hexdigest()
    for key,expected in [('different_image',False),(digest,True)]:
        data=dict(original_frame_sha256=key,points_original_px=[[10,10],[20,20]])
        with patch('pathlib.Path.read_text',return_value=json.dumps(data)), \
             patch('vision_geometry_lab.app.messagebox.showerror') as error:
            app.import_wound_reference('reference.json')
        assert bool(error.called) is not expected
    assert app.wound_prompts==[[10,10],[20,20]] and calls==['analyze']


def test_reference_import_rejects_non_object_json_without_mutating_prompts():
    app=app_without_window(); app.root=None
    app.wound_prompts=[[7,8]]
    calls=[]; app.analyze=lambda:calls.append('analyze')
    with patch('pathlib.Path.read_text',return_value='[]'), \
         patch('vision_geometry_lab.app.messagebox.showerror') as error:
        app.import_wound_reference('reference.json')
    assert error.called and not calls
    assert app.wound_prompts==[[7,8]]


def test_video_frame_change_clears_old_wound_prompts():
    app=app_without_window()
    class GoodVideo(BadVideo):
        def read(self):
            return True,np.zeros((40,40,3),np.uint8)
    app.cap=GoodVideo(); app.wound_prompts=[[12,13]]
    app.render=lambda:None; app.analyze=lambda:None
    app.read_frame(38)
    assert app.wound_prompts==[]


def test_display_snapshot_uses_ui_switches_and_keeps_interaction_marks():
    frame=np.zeros((80,100,3),np.uint8)
    settings=dict(overlay=True,show_arc=False,show_rope=True,show_raw=True,
                  show_exclusion=False,show_conic=True,show_wound=False)
    base=np.zeros_like(frame); base[0,0]=(1,2,3)
    with patch('vision_geometry_lab.app.draw_overlay',return_value=base.copy()) as draw:
        shown=compose_display_image(frame,{'result':True},settings,
            arc_roi=[5,5,30,30],rope_roi=[40,5,70,30],
            prompts=[[15,45],[35,45]],prompt_labels=[1,0],wound_prompts=[[60,50]])
    draw.assert_called_once_with(frame,{'result':True},show_arc=False,show_rope=True,
        show_raw=True,show_exclusion=False,show_conic=True,show_wound=False)
    assert tuple(shown[0,0])==(1,2,3)
    assert shown[5,5].any() and shown[5,40].any()
    assert shown[45,15].any() and shown[45,35].any() and shown[50,60].any()


def test_display_snapshot_with_geometry_hidden_still_shows_guidance():
    frame=np.zeros((60,80,3),np.uint8)
    with patch('vision_geometry_lab.app.draw_overlay') as draw:
        shown=compose_display_image(frame,{},dict(overlay=False),
                                    arc_roi=[3,3,20,20],wound_prompts=[[30,30]])
    assert not draw.called
    assert shown[3,3].any() and shown[30,30].any()


def test_guidance_wording_and_new_reason_translation_are_explicit():
    assert '自动生成候选' in INITIAL_STATUS_TEXT and '不自动确认绳身份' in INITIAL_STATUS_TEXT
    assert '人工搜索提示' in OVERLAY_LEGEND_TEXT
    assert '不是绳体中心线或实体边真值' in OVERLAY_LEGEND_TEXT
    translation=WOUND_REASON_TRANSLATIONS['guided_body_center_vs_edge_unresolved']
    assert '绳体中心' in translation and '实体边缘仍未区分' in translation
