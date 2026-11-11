"""LC1 adapter for the unchanged RB-F3.0.0 traditional observation core.

This is not the complete historical F3 fusion pipeline and invokes no model.
All inputs and outputs use the caller's same full-resolution image coordinates.
The adapter does not mirror, crop, resize, or invent a surface/pitch estimate.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CORE_PATH = (ROOT / "source_archive" / "数据资产" / "05_派生与实验结果"
             / "camera_condition_full_surface_joint_20261028_v3" / "surface_observation.py")


@lru_cache(maxsize=1)
def load_core():
    name = "_lc1_frozen_f3_surface_observation"
    spec = importlib.util.spec_from_file_location(name, CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load restored F3 observation core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def detect_surface_core(original_bgr, surface_mask, pitch_px, *,
                        exclusion_mask=None, c3_record=None):
    """Return image-supported D4 observations on the supplied full surface.

    original_bgr: uint8 H x W x 3 BGR.
    surface_mask/exclusion_mask: same-size uint8 masks with values 0/255.
    pitch_px: observed image pitch in original pixels, 8 <= pitch <= 70.
    c3_record: optional unchanged C3 record with its shape reference; None
        requests the original core's no-reference branch.
    exclusion_mask=None means an explicitly empty exclusion mask, not a
        claim that the real scene has no occlusion.

    Coordinates are full-image xy in the input orientation. This function
    returns the archived core's dictionary unchanged. Full F3 fusion needs
    additional historical C3/D4/F2 vectors and masks; see F3_RESTORE_NOTES.md.
    """
    import numpy as np

    if exclusion_mask is None:
        exclusion_mask = np.zeros(surface_mask.shape, dtype=np.uint8)
    data = dict(original=original_bgr, surface=surface_mask, pitch=float(pitch_px),
                c3_record=c3_record, c3_exclusion=exclusion_mask)
    return load_core().detect_surface(data)
