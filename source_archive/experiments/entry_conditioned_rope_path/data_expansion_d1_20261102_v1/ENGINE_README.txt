D0/D1 continuation engine

Actual run completed: independent100-step smallcheck, then fresh A750 resets
and600updates each for D0/D1. Dataset/index.json was frozen before training.
Execution receipts/logs preserve actual commands, durations and exit codes.
The original terminal audit failed only at exact OpenCV seed reconstruction;
its failure record and the separate repaired audit are preserved. See REPORT.txt.

Run all Torch commands from project root with existing vision_training Python:

vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/data_expansion_d1_20261102_v1/train_infer.py smallcheck
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/data_expansion_d1_20261102_v1/train_infer.py train --branch D0
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/data_expansion_d1_20261102_v1/train_infer.py train --branch D1
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/data_expansion_d1_20261102_v1/train_infer.py select
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/data_expansion_d1_20261102_v1/train_infer.py infer-fixed --branches D0 D1
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/data_expansion_d1_20261102_v1/train_infer.py infer-gallery --branch D1
vision_prototype/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/data_expansion_d1_20261102_v1/render_export.py --workers 4

Formal settings: all EntryUNet parameters, same A750 initial weights and fresh
AdamW optimizer, lr0.0002,600updates/max2400seconds per branch, last common
100-step checkpoint. No model choice uses evaluation targets or gallery looks.
D0 samples old10 uniformly. D1 uses same D0 old item/augmentation on even steps,
and group-uniform accepted new weak items on odd steps. Actual ids, groups,
origins, input tensor hashes, valid pixels and gradients are recorded per step.
No separate B weak-negative terms or confidence weighting are added.

Dataset index items: split=train,origin=old/new,id,source_group,image_path,
image_sha256,known_seed_xy,cache. Cache keys image,seed,target float32;
valid,roi bool;stage uint8(0..4), all975x2048. No valid stage0 is permitted.
Inference never reads target/valid labels: raw original plus frozen entry only.

Fixed outputs:
inference/fixed/{D0,D1}/{CONDITIONAL,AUTO}/{id}/response.npz
contains response float32(975,2048). Sibling response.json binds model, source,
entry, preprocessing, shape, and original crop. Missing AUTO entry yields NaN
and entry_unavailable; never interpreted as no rope. Includes old18COND+8AUTO,
plus new training images as CONDITIONAL diagnostics.

Gallery outputs:
inference/gallery/D1/{id}/entry_{nn}.npz plus sibling JSON,301 separate entries.
Original crop is[0,1050,4096,3000], input2048x975 padded to992 rows. Rendering
uses fixed0..1TURBO and original bilinear align_corners=False mapping.

final_292 has exactly292 PNG, one per unique source, frozenentry0. Two full
4096x3000 viewports plus external header/footer/colorbar. Right body has NO
path, seed, mask, line drawing or morphology; all native D1 response values
remain visible. Left pixels and full right raster are checked on export.

Do not overwrite prior run artifacts. Training/checkpoint directories and
export refuse existing destinations. On failure preserve logs and diagnostics.
