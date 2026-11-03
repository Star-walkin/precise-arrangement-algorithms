Independent video-output audit (IR-V1.0.0)

Use the existing vision_prototype virtual environment. No production engine,
temporal gate, or renderer module is imported by these checks.

Completed whole run:
vision_prototype/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v1/audit/audit_run.py --run all_001

If partial decode receipts already exist, append --reuse-partial-decodes.
Complete frame decoding is reused only for identical video SHA256, frame
count, frozen protocol and production-source hashes. Logs, numeric arrays,
current hashes and full source coverage are checked afresh. Final receipts
link every reused decode receipt and preserve its hash.

Completed individual videos while other videos are running:
vision_prototype/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v1/audit/audit_run.py --run all_001 --video VIDEO_ID

Whole-run descriptive comparison:
vision_prototype/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v1/audit/temporal_comparison.py --run all_001

Authoritative paired-distance comparison (common domain for all four paths):
vision_prototype/.venv/Scripts/python.exe -B experiments/active_rope_video_20261101_v1/audit/common_domain_comparison.py --run all_001

The common-domain result supersedes only the paired-distance statistics in
temporal_comparison.json. That first diagnostic matched observation pairs
but permitted each method its own shared path range. The refined result
uses identical arclength sample positions and a shared normal field for
both methods, and stores every evaluated frame pair. The first diagnostic
remains preserved; its per-state counting is unchanged.

audit_run.py checks source allowlist/protocol/code hashes; observation/media
time mapping; accepted-only causal history; current-image support flags and
supported segment length; exact accepted raw candidate geometry; independent
shared-arclength lateral distance; heat-map entry alignment; fresh float32
response dimensions/range/hashes; reject/old-reference/expiry/reacquisition
semantics; rendering records; and complete output-video decoding.

The historical pilot002 receipt permits only the explicitly declared later
runner/renderer display/log changes. Its engine and gate hashes matched.
The full all_001 audit requires every production source hash to match.

temporal_comparison.py keeps the upstream rank-0 and image-guard baselines
separate from temporal acceptance. Current-path displacement is compared on
identical adjacent observation pairs, with rejections and new reference
epochs counted separately. It does not count a stale held path as a current
measurement or a correct result. No lateral alignment or nearest-neighbor
curve matching erases neighboring-rope jumps.

All receipts are append-only: scripts refuse to replace an existing receipt.
Partial receipts assert only the listed completed videos, never full coverage.
Reported tests establish software/output consistency, not rope identity or
path correctness. Static-scene F3 propagation is not new full SAM2/F3
measurement. The three-seconds/projected-pitch speed prior is uncalibrated.
Processing latency is separate from original-speed output playback.
