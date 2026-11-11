# Restored F3 traditional observation core for LC1

Repository: Star-walkin/precise-arrangement-algorithms  
Commit: 8d30826fbef2df423ccfc56047995e6194f12b6c  
Tree: d91e066e3c9b3f681585f71fcb7f2b0863d1869a  
Archived version: RB-F3.0.0, 2026-10-28

The 11 restored vendor files retain their original repository paths under
vendor_archive/source_archive/. Nine are Python sources; two are the original
F3 protocol/report. The code and text are unchanged, including original line
endings. f3_source_manifest.json records Git blob IDs and the P1 archive's
SHA-256 values when available. f3_restore_receipt.json independently verifies
the restored file bytes, SHA-256, and Python syntax.

## What can run independently

f3_core_adapter.detect_surface_core() calls the unchanged archived
surface_observation.detect_surface(data). The complete runtime dependency
closure of this call is six source modules:

- camera_condition_full_surface_joint_20261028_v3/surface_observation.py
- camera_condition_seam_joint_20261028_v2/seam_observation.py, only its
  _normalize_surface() function is used by this core
- camera_condition_dark_width_20261027_v4/shadow_width.py
- camera_condition_dark_width_20261027_v4/period_guard.py
- camera_condition_dark_width_20261027_v4/shadow_tracks.py
- camera_condition_dark_width_20261027_v4/ribbon_curves.py

External packages are NumPy and OpenCV; the core calls no neural model.
The seam module's other dynamic dependencies are not imported by
_normalize_surface(). No original D-drive dataset loader is called.

Input: uint8 H×W×3 BGR original image; same-size uint8 surface and exclusion
masks with only 0/255; observed image pitch in original pixels from 8 through
70; optional C3 record carrying its existing shape reference. None selects
the original no-reference branch. The adapter's missing exclusion default
is an explicitly empty mask and does not establish absence of occlusion.

Output: dictionary with normalized_bgr, points, tracks, fit, fields,
exclusion_mask, and diagnostics. All xy are full-resolution coordinates in
the input image's orientation. No mirror, crop, or resize is performed by
the adapter. The six core modules remain in their archived relative layout,
so the core's dynamic source loading needs no absolute-path modification.

Example, after putting vendor_archive on sys.path:

```python
import numpy as np
from f3_core_adapter import detect_surface_core

# image_bgr, surface_mask, pitch_px must be measured/provided by LC1.
observations = detect_surface_core(
    image_bgr, surface_mask, pitch_px,
    exclusion_mask=np.zeros(surface_mask.shape, np.uint8),
    c3_record=None,
)
curves = observations["fit"]["curves"]
```

For this machine, use the new .venv_lc1/Scripts/python.exe in the workspace
root. It supplies OpenCV/NumPy and avoids the old broken virtual environment.
Vendor files contain no machine-specific runtime or path changes.

## What full historical F3 additionally requires

The restored surface_fusion.fuse_surface(data, detected) is a pure traditional
fusion function, but it is not equivalent to the standalone observation
core. Its input contracts include:

- data.pitch and data.surface in the same original-image coordinate system.
- data.f2_merged: previously accepted full-image F2 boundary raster.
- data.f2_inferred: F2 inferred raster, with source status retained.
- data.f2_transition: frozen transition mask; new final additions are limited
  to this region intersected with surface.
- data.c3_record.fit.curves and data.d4_record.fit.curves, with accepted
  sampled_points_xy and curve_id, for original reference correspondence.
- data.f2_case.joins, with source_curve_ids, sampled_points_xy and candidate_id,
  for accepted F2 source lineage/reference joins.
- detected.fields, detected.tracks and detected.fit from the current
  observation stage, including source points and inferred status.

surface_fusion.py:66-91 constructs accepted C3/D4/F2 reference lineage;
:255 onward performs the actual fusion. The original run_surface.py additionally
requires data.f2_image for rendering, verifies unchanged outside-transition
pixels, and loads hash-locked historical inputs through surface_io.py.

run_surface.py and surface_io.py are restored for traceability only. The full
IO chain, source datasets, old C3/D4/F2/SAM2/C4 predictions and caches have
not been restored here. Full historical run_surface.py is therefore not
advertised as independently runnable. Its indirect dataset loaders retain
their original paths/protocols; any future relocation belongs in a new
adapter, never in the archived source.

The historical surface and exclusion masks include upstream model-derived
priors. Supplying a newly measured/manual/traditional ROI and pitch permits
a new model-free observation experiment; that change must be recorded as
an LC1 adaptation. It must not be called an exact full-F3 reproduction.
Neither D4 candidate curves nor fused visual boundaries establish physical
rope identity, contact truth, or a validated remaining-rope count.

## Verification scope

verify_f3_restore.py checks the exact restored bytes and source syntax.
With --smoke it invokes the core on an empty surface and a synthetic
periodic image, checks array contracts/coordinate range/input immutability,
and records counts and runtime versions. This reads no research image,
reserved media, model weight or device. Synthetic execution is not an
accuracy evaluation.
