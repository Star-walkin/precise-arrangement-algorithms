Video FrameEngine: original B750 + current A7 + history scene prior + frozen H6

Run with D:/precise_arrangement/vision_training/.venv/Scripts/python.exe.
engine.py appends the existing vision_prototype site-packages for OpenCV;
no packages were installed. No acquisition/hardware module is imported.

API:
    from frame_engine.engine import FrameEngine, anchor_job
    engine = FrameEngine(anchor_job(video), anchor_time=anchor_actual_time, threads=2)
    result = engine.step(original_bgr_4096x3000, actual_time_s)
Do not call before the earliest permitted scene anchor; render UNKNOWN then.
Timestamps must strictly increase; use the source's recorded time basis.

Output:
 heat: float32 975x2048 [0,1] or None if no validated automatic short entry.
       Raw B forward; original_xy = 2*heat_xy + [0,1050]. No painted paths.
 heatmaps_by_entry: separate float responses for each condition (<=2). If a
       temporal selector chooses another entry, display that entry's response.
 normalization_by_entry: corresponding actual gray input statistics.
 seeds: original-pixel short automatic entries, source and age.
 raw_candidates: <=3 independent original-pixel H6 candidate paths, stage,
       point_status, supported, ranking metrics and entry_index.
 surface_tracks: original-pixel history F3 center geometry, inferred flags,
       current_image_supported, age. They are not material identities.
 pitch_original_px: saved pixel spacing; not calibrated physical diameter.
 diagnostics: per-stage timing, forward count, normalization, current scene
       registration and entry checks, anchor provenance, no future records.

Initialization reads only the selected earliest open400 image's SAM2/C4/F3
scene prior and its source dependencies. Loaded old A6/A7 entries are discarded.
No saved B response, H6 target path, later dataset observation or target label
is used. Fresh current-frame A7 supplies the short entry; repeat every 3 s,
or immediately on entry loss. Between refreshes forward/back LK must pass
both displacement/round-trip gates and current original-image dual profiles.
No entry means heat=None, not a zero-seed pseudo target heatmap.

F3 is NOT rerun each video frame. Static upper machinery LK (original y50–1000)
estimates bounded scene translation, preserving lane ordering. Rotation of
spool material is not used as scene motion. Per-track original gray and B
profiles are recomputed on every current frame; saved observed/inferred
boundary flags remain historical provenance, not fresh boundary observations.
Failed scene registration forbids new attached continuation; historical gray
tracks display inferred. Raw supported free observations remain available.
The scene range is not expanded, new lanes are not invented, and existing
scene prior cannot validate unobserved changed geometry. Identity remains
unknown even when geometry/profile gates pass.

B remains the original two-channel EntryUNet step0750, no H7 C/D, no training.
Image -> original gray -> crop[0,1050,4096,3000] -> 2x area reduction -> crop
mean/std (std floor .08) -> concatenate Gaussian short seed sigma2 half px ->
neutral bottom padding17 -> B -> remove padding. Heat display fixed0–1.

Executed development runs:
 benchmark_upstream.py: two open originals, raw-only A7 and full-D4 timings.
 pilot.py: fixed source101517right and L_runleft,10 frames each,0.5s spacing.
 pilot_001: first sceneLK used rotating spool; weakness retained for diagnosis.
 pilot.py --run pilot_002: static machinery registration;20 frames exit0,
 19 forwards,18 candidate frames,2 UNKNOWN,2.401s/frame mean on CPU threads2.
 pilot.py --run no_entry_smoke_001 --no-entry-smoke: original101517left,
 10 frames exit0,10 UNKNOWN,0 forwards,1.299s/frame. Failure is retained.
 The final scene-invalid attachment guard is an adapter policy; no frozen
 H6/B/A7 code or model changed. Full-video runtime/protocol belongs to parent.

No reserved data, S1 labels, target truth, devices or machine-control used.
