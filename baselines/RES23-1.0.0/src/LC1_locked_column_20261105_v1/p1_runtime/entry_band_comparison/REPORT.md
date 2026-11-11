# Explicit entry-band comparison and S1 gap audit

Date: 2026-11-05. Only frame0 of the two already authorized development videos
was decoded. Model SHA is the verified frozen B750. No reserved scene was read.

The new optional constructor arguments are:
    P1H6Adapter(mask, pitch, model_path, threads=2, camera=None,
                surface_mode="entry_band",
                entry_band_halfwidth_pitch=8., entry_band_halo_pitch=2.)

The default remains surface_mode="full". The optional D4 observation mask is
the intersection of the static SAM2 surface with the union of current automatic
A7 axis bands, half-width 8*pitch plus a retained 2*pitch halo. No image resize is
introduced. Only the current D4 detector search domain is reduced; the B750
input and H6 task/terminal domain remain full. The band is not represented as
a complete detected surface. Curved support outside it is missing.

Full-frame baseline records were produced by the earlier adapter SHA
f10b73ef48a21fd84aa72dbfe58d3eb292c3f12b4e27d7f5d834e13d026914e9.
They are retained without rewriting. comparison.json binds the new adapter
SHA, identical decoded pixel SHA, both output hashes and the comparison data.

Frame0 results:
| case | full D4 seconds | band D4 seconds | surface pixels retained | H6 tracks |
|---|---:|---:|---:|---:|
| right113305 | 25.074 | 15.895 | 41.02% | 225 to96 |
| left111500 | 16.455 | 11.646 | 46.35% | 146 to80 |

The right selected free-only path is exactly unchanged. The left free-stage
points are exactly unchanged; transition maximum y difference is 0.0254px.
The left selected surface branch on common original x2312..2472 differs by
at most0.1953px in y. Its branch score margin changes from0.07782 to0.07121.
The selected type remains a genuine legal H6 curve connection. Track identifiers
cannot be compared as historical rope identity because they derive from the
current D4 curve list. The full-path flag stays false in both modes.
This is a useful 1.58x/1.41x reduction in D4 time, not a real-time claim.

The largest left original-frame gap is in the END of the free stage:
    path indices292..363
    original xy(1851.1281,2452.1675) to(2269.2183,2533.8783)
    length426px
    point states: 48 ambiguous_gap,23 heatmap_supported,1 image_supported
It reaches the transition neighborhood; it is not merely a harmless middle
segment missing neural response. The frozen connector's subsequent transition
has4 joint-supported points. The surface branch has32 points,59.375% joint
support,117.75px joint-supported length and a62.70px maximum unsupported run.
The free-to-transition total unsupported run is434.93px.

Genuine frozen S1 was run on the same original frame with the current automatic
A7 seed, not a manual/recovered historical E1. It returns PARTIAL and299 actual
bilateral support samples; the optical endpoint is(1482.1617,2380.0022).
Every point of the426px final-free gap is beyond actual S1 supported segments:
coverage is0% within either8px or12px. Nearest finite supported segment distances
are median588.96px and maximum801.96px. Analytically extending S1's line through
the missing region would be unsupported extrapolation.

There is an earlier132px free-stage weak-raw interval at original
(1321.1547,2348.5905) to(1450.7038,2373.9093), mainly heatmap_supported; genuine
S1 covers65.2% within8px or78.3% within12px. S1 does add some current evidence
there, but this cannot validate the later426px turn-adjacent gap. Therefore no
P1+S1 fusion-complete flag is set and the P1 original flag remains false.
Lowering a global observed-percentage threshold is not justified by these data.

The overlay compares diagnostic full candidate geometry (purple dashed),
actual supported band segments (green), unsupported band segments (orange dashed)
and actual supported S1 segments (blue). Full-candidate purple remains diagnostic;
each JSON record's path_segment_support governs solid detection rendering.

