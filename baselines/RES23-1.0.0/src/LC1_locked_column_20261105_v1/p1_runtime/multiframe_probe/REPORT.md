# Real B750/H6 + S1 development-frame probe

All12 requested frames ran successfully using original unmirrored4096x3000 images.
Explicit mode:entry_band, current A7 ±8pitch plus2pitch halo; this differs from
full D4 mode. No historical target path/cache or reserved scene was read.

Actual B750 forwards:12; genuine S1 executions:12.
Whole-supported frames:0/12. Geometrically admissible locked-column frames:0/12.
All S1 outputs are PARTIAL optical straight evidence; their endpoints are never d.

| case/frame | H6 candidate | actual max gap px* | S1 free coverage | P1/S1 free union | turn fit | seconds |
|---|---|---:|---:|---:|---|---:|
| left111500/0 | curve attached | 434.9 | 58.9% | 66.8% | straight_path | 17.1 |
| left111500/15 | curve attached | 144.0 | 87.7% | 91.7% | straight_path | 17.2 |
| left111500/30 | curve attached | 225.1 | 55.7% | 82.5% | turn_localization_uncertain | 16.4 |
| left111500/45 | curve attached | 138.0 | 75.9% | 87.8% | straight_path | 16.3 |
| left111500/60 | curve attached | 178.7 | 94.4% | 94.8% | straight_path | 16.3 |
| left111500/75 | free only | 84.0 | 95.5% | 100.0% | straight_path | 15.7 |
| right113305/240 | curve attached | 474.7 | 21.5% | 58.6% | straight_path | 18.8 |
| right113305/270 | free only | 0.0 | 100.0% | 100.0% | straight_path | 18.2 |
| right113305/300 | free only | 330.0 | 42.8% | 83.4% | straight_path | 17.4 |
| right113305/330 | free only | 384.0 | 42.6% | 78.7% | straight_path | 17.3 |
| right113305/360 | free only | 78.0 | 100.0% | 100.0% | straight_path | 16.1 |
| right113305/390 | free only | 24.0 | 100.0% | 100.0% | straight_path | 16.2 |

*The input entry_condition extent is excluded from these gap/coverage statistics;
it adds no observed length and is never rendered as supported. The original
JSONL is immutable; support_scope_audit.json records this scope correction and
proves all12 complete flags remain false with unchanged evidence thresholds.

S1 coverage uses nearest actual finite supported segments within8px, without
extrapolating S1's fitted line. The union column applies only to the free stage;
a high value cannot supply missing transition or curved-path evidence.

Left f15/f60 have91.7%/94.8% union free coverage, yet their transition evidence
is insufficient and the overall path does not establish a localized bend.
Left f75 and right f270/f360/f390 can have complete observed free coverage but
H6 has no admissible curved attachment. Thus a percentage-only fusion gate
would not establish a physical or optical straight-to-curved transition.

Left f30 retains a curved candidate but has225.05px actual unsupported run,
weak transition evidence and a branch ambiguity; estimate_turn explicitly
rejects turn_localization_uncertain. The raw fitted x is diagnostic only.
Right f240 retains a curved candidate but474.68px loss of current support;
the geometry fit is straight_path. Other right samples retain free fragments.

Overlay legend:green supported free, purple supported transition/surface,
orange dashed weak candidate geometry, blue actual S1 bilateral segments.
Any fit marker is explicitly diagnostic; no fixed reference column was drawn.

Data/provenance:frames.jsonl, receipt.json, support_scope_audit.json,
compact_summary.json, and source_snapshot/ preserve coordinates, per-segment
support, source pixel hashes, decoded frame indices, source code hashes and
each output overlay SHA256. The12 overlays were rehashed successfully.
Total per-frame compute mean:16.92s;
D4 mean:11.39s.
