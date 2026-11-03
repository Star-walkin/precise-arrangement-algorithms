Left automatic weak labels 2026-11-02

12 real images from frozen newly allowed L_run and L_run_stop_run sources.
Raw whole contexts and both native overlay windows for every image were
actually viewed. No U-Net heat was consulted as an acceptance threshold.

JSON schema: id,image_path,image_sha256,source_group,seed_xy,segments,review.
accepted_weak segments contain original xy center points, stage, 6px local
valid_radius_px, initial anchors and original-gray refinement diagnostics.
Stage1 is outside the independent saved spool mask; stage2 is projected on
that mask. The mask only classifies visual position, not rope identity or
physical contact. All confirmed bent/edge identity remains absent/unknown.
Unknown tail and insufficient bilateral or ±4px-bound-hitting fits are ignored.

annotation_refine.py uses bounded normal opposing-edge fitting after tangent
median integration. Narrow body and broad cast shadow were visually separated.
A7 provides a coarse raw-image axis only, not its acceptance or saved endpoints.
All new labels remain automatic weak labels, not independent human truth.
No neighbor negatives were fabricated from unlabeled areas. Label endpoint is
not a physical rope endpoint. Parent dataset builder retains sigma3 and ignores
local endpoint caps. Native crops use scale1 and include original-coordinate
bboxes for exact mapping. review_receipt.json binds actual accepted JSON hashes.
