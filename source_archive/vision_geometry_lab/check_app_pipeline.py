"""Hidden-window smoke test of real Tk, decode, worker, labels and parameter state.

Only opens the project test video; no camera, light or motion hardware access.
"""
from pathlib import Path
import time
import tkinter as tk
import cv2
from .app import App


def main():
    cv2.setNumThreads(2)
    root=tk.Tk()
    root.withdraw()
    app=App(root)
    def finish():
        deadline=time.monotonic()+60
        while app.busy and time.monotonic()<deadline:
            root.update()
            time.sleep(.02)
        if app.busy:
            raise TimeoutError('analysis worker exceeded 60s')
        assert app.result is not None,app.status.get()
    try:
        path=Path(__file__).resolve().parents[1]/'现场视频/训练采集/双相机同步/dual_20260924_111500/left_DA9627065.avi'
        app.open(path)
        finish()
        app.read_frame(round(26.5*app.fps))
        finish()
        assert app.result['frame_index']==265
        assert app.result['rope']['valid'] and app.result['conic']['valid']
        assert app.result['exclusion']['valid']
        assert app.result['arc']['local_boundary_valid']
        assert app.result['wound_curve']['identity_confirmed'] is False
        assert '交点' in app.contact_text.get()
        print(app.arc_text.get())
        print(app.rope_text.get())
        print(app.contact_text.get())
        app.show_raw.set(True)
        app.show_exclusion.set(True)
        app.show_conic.set(True)
        app.show_wound.set(True)
        app.render()
        app.exclusion_margin.set('9')
        assert app.result is None and not app.playing
        app.analyze()
        finish()
        assert app.result['exclusion']['extra_margin_px']==9
        app.read_frame(246)
        finish()
        reference=Path(__file__).parent/'outputs/wound_curve_v3/manual_247/reference.json'
        app.import_wound_reference(reference)
        finish()
        assert len(app.wound_prompts)==6
        assert app.result['wound_guidance']['used']
        assert len(app.result['wound_guidance']['points'])==6
        assert '人工提示' in app.wound_text.get(),app.wound_text.get()
        app.read_frame(247)
        finish()
        assert app.wound_prompts==[]
        assert app.result['wound_guidance']['used'] is False
        print('PASS real Tk decode/worker/labels/render/parameter invalidation; hardware access FALSE')
    finally:
        app.close()


if __name__=='__main__':
    main()
