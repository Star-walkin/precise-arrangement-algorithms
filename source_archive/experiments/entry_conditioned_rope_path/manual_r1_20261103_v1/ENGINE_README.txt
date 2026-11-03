Manual U1 training/inference engine

From project root, existing vision_training environment:
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/train_manual.py smallcheck
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/train_manual.py train
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/train_manual.py select
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/train_manual.py infer-fixed
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/train_manual.py infer-gallery
vision_training/.venv/Scripts/python.exe -B experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/train_manual.py fit-stats

Do not run before protocol.json and frozen dataset/index.json are prepared.
Code authoring/numerical checks do not imply actual training. Actual command
logs, model/training_receipt.json and inference receipts record execution.

Data index: frozen=true; items split=train,origin=old(10)/manual(24),id,
source_group,image_path,image_sha256,cache,cache_sha256,known_seed_xy.
NPZ cache fields image/seed/target float32,valid/roi bool,stage uint8,
all975x2048. Optional negative bool is final valid red-line pixels at target0.
Unknown pixels stay invalid. Red lines do not become a new separate loss.

Protocol: start_model:{path,sha256},eval_groups,gallery_jobs. Gallery jobs use
id,image_path,image_sha256,source_group,seeds:[{xy:[[x,y],...]}]. Entry0 is the
pre-inference frozen display priority, not a best-looking seed selection.

Architecture and active input/augmentation settings inherit A750. Entire
135897-parameter EntryUNet trains on raw gray plus short seed. LCN remains
annotation viewing only. Original search crop[0,1050,4096,3000], half resolution,
975rows padded to992, per-search mean/std floor0.08, unchanged tensor_sample.
Single unchanged stage_balanced_mse. User visual phase numbers are retained;
the engine does not reinterpret defaultstage1 as a physical-contact truth.

Independent smallcheck100updates/max300s. Formal model resets to A750, fresh
AdamWlr0.0002/wd0.0001,800updates/max2000s,batch1,8CPUthreads. Alternating
manual/old gives50:50, old uniform shuffled cycles and manual group-balanced
cycles with shuffled image cycles. All manual images are guaranteed sampled
by the complete planned800steps. Actual counts always come from training log.
Save100step checkpoints plus final actual step if a time cap is reached.
Selection is last actual saved formal update, never evaluation-dependent.

Outputs: model/initial.pt,checkpoints/,final.pt,train_log.jsonl,fit_before.json,
fit_after.json,training_receipt.json; selected_checkpoints.json paths.U1.
Logs record sample/augmentation/input tensor SHA, gradients, red target0
response/MSE diagnostics, active stage loss, normalization and elapsed time.
Manual red zero-target diagnostic after interpolation is a logging mask only.

Inference reads raw images plus frozen entry, never target/valid caches or
evaluation references. Fixed jobs:18COND+8AUTO+manual24; missingAUTO preserved
as NaN response with explicit unavailable status. Gallery292/301conditions.
inference/fixed/U1/{CONDITIONAL,AUTO}/{id}/response.npz
inference/gallery/U1/{id}/entry_00.npz (and other frozen entry indices)
Each successful NPZ has response float32(975,2048),features float16(8,248,512).
Features are avg_pool4 of NEW trained U1 dec1, hooked in the SAME new forward
as response. They never come from old A feature cache. Pooled coordinates:
model=(original_xy-[0,1050])/2; pooled=(model+0.5)/4-0.5.
Sibling JSON binds source/model/seed/response hashes and original geometry.
Responses are raw0..1 center strengths, not calibrated identity probabilities.

Training fit and red selectivity diagnostics are supplied-label fitting, not
independent full-path accuracy. Fixed evaluation labels are read by a later
separate scorer only after predictions and paths are saved. No device,
reserved or S1 imports. Existing source/models/labels remain read-only.
