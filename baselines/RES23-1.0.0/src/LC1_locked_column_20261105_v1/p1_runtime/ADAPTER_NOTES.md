# LC1 real B750/H6 adapter

This directory restores the minimum original P1 neural network, A7 entry detector
and H6 postprocessing sources from private repository
Star-walkin/precise-arrangement-algorithms at commit
8d30826fbef2df423ccfc56047995e6194f12b6c. source_manifest.json records the
repository-relative path and Git blob SHA. source_restore_receipt.json verifies
the restored bytes. Original source_archive files are unchanged.

The public adapter is ../lc1_p1_adapter.py:
    P1H6Adapter(surface_mask_original, pitch_px, model_path, threads=2,
                camera=None).step(bgr_original)

Inputs are original unmirrored uint8 BGR images, exactly 4096x3000. The supplied
surface mask is a static scene search domain and does not contain an active-rope
target. Left input is not flipped inside P1. All public xy outputs are original
image pixels. H6 internal crop-half points map by original=2*xy+[0,1050].

B750 weights: step_0750.pt, 562941 bytes; SHA256
ed4bfa12c5b3081a19d056265b053d516bdba5ec05dce658e1e32339b47fb882.
The adapter rejects any different SHA. It loads the real EntryUNet checkpoint,
runs the original two-channel preprocessing (normalized grayscale original
rows1050:3000 at half size plus Gaussian current A7 seed), pads 17 bottom rows,
and reads the sigmoid response over 2048x975. CPU inference uses two threads by
default; no historical response file is read.

Current image-only A7 supplies each automatic entry. A7's finite outside anchor
is sorted far-end to spool using distance to the static scene domain. The
adapter retains at most two raw-supported conditions and does not merge heatmaps.
seed_xy and seed_source are suitable for the original-orientation S1 wrapper.
line_points_xy contains current joint raw/heat or explicit current raw A7
observations; neither the input seed nor unsupported gaps count as observations.

The upstream scene chain is an explicit LC1 adaptation: run the unchanged F3
traditional observation core on the current full frame and accept its D4 fits
as current boundary vectors. Then the genuine unchanged H6 build_tracks,
describe_free, prepare_task_free, connect, and task_direction_feasibility run on
the current image and real B750 response. Historical C3/F2/F3 fusion caches are
not loaded. This is not replay of the complete frozen historical P1 scene chain,
and is not complete historical F3 fusion. Raw inherited H6 descriptions use the
word saved/F3; their current adapter scope is explicitly marked in the output.

step() returns compact JSON-safe geometry and diagnostics. path_xy can retain a
partial/ambiguous candidate for diagnostics. Only diagnostics.path_segment_support
authorizes solid rendering. whole_path_candidate_supported is a separate strict
LC1 current-evidence gate checking eligible H6 attachment, free/transition/curve
support, supported lengths, unsupported gaps, A7 uniqueness/direction and H6
branch margin. It never verifies physical contact or same-rope material identity.
No old target path, manual target seed, label or reserved frame is read.

smoke_current.py reads only the two authorized development source videos, frame0,
in original orientation. smoke_current_frame/ contains exact JSON records, actual
B750 float responses in NPZ, heatmap visualizations, and original-frame overlays.
The receipt distinguishes successful runtime execution from whole-path validity.

Observed smoke outcomes (2026-11-05):
- right113305 f0: one real B750 forward; automatic A7 accepted; 225 current
  midpoint tracks; H6 selects only a free fragment (204 points), rejecting weak
  attachment. Whole-path gate false. Current free observations remain available.
- left111500 f0: one real B750 forward; automatic A7 accepted; 146 current
  midpoint tracks; H6 selects a legal curve connection (400 points), but current
  free support has a 426px gap. Whole-path gate false.
These are evidence failures, not a missing-weight claim. Overlay orange dashed
segments are diagnostic unsupported geometry; they cannot calibrate d.
Full current D4 observation is the dominant startup/frame cost (right25.1s).
No model or H6 parameters were tuned to force these two frames to pass.

