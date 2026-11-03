"""Read-only offline probe of the OLD semantic model, not an instance tracker.

Run with vision_prototype/.venv/Scripts/python.exe.  No training, hardware,
old geometric postprocessing, or application integration is performed.  The
six manual points are used only for sampling and an explicitly OOD crop;
they are NOT fed to the model and are NOT edge/centreline ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "vision_geometry_lab"
MODEL_DIR = ROOT / "vision_training/task_left_v1/models/left_camera_thick_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_image(path, image):
    ok, encoded = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(str(path))


def audit_training():
    result = []
    tasks = ["task_left_v1", "task_v1", "task_dual_midlayer_v1/train",
             "task_dual_midlayer_v1/holdout", "task_dual_thicklayer_v1/train",
             "task_dual_thicklayer_v1/holdout"]
    for task in tasks:
        folder = ROOT / "vision_training" / task
        manifest = read_json(folder / "manifest.json")
        annotations = [read_json(p) for p in (folder / "labels").glob("*/annotation.json")]
        entry = {"task": task, "manifest": str(folder / "manifest.json"),
                 "anchors": len(manifest.get("tasks", [])),
                 "annotation_files": len(annotations),
                 "annotation_fields": sorted({k for a in annotations for k in a}),
                 "classes": sorted({k for a in annotations for k in a.get("classes", {})})}
        dataset_path = folder / "distillation_dataset.json"
        if dataset_path.exists():
            dataset = read_json(dataset_path)
            entry.update({"distillation_manifest": str(dataset_path),
                          "split_frames": dict(Counter(r["split"] for r in dataset["records"])),
                          "source_windows": sorted({r["window_id"] for r in dataset["records"]}),
                          "dataset_warning": dataset.get("warning"),
                          "crop_xyxy_normalized": dataset["crop_xyxy_normalized"],
                          "input_size_wh": dataset["input_size_wh"]})
        result.append(entry)
    return result


def check_runtime_and_export(output):
    """Safe CPU checkpoint check in its existing environment; never install packages."""
    runtime = {"probe_python": sys.executable,
               "probe_torch_installed": importlib.util.find_spec("torch") is not None,
               "probe_onnxruntime_installed": importlib.util.find_spec("onnxruntime") is not None,
               "opencv_cuda_device_count": cv2.cuda.getCudaEnabledDeviceCount()}
    try:
        gpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json"],
                             capture_output=True, text=True, timeout=15)
        runtime["os_display_adapters_readonly_inventory"] = json.loads(gpu.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        runtime["os_gpu_inventory_unavailable"] = str(exc)
    training_python = ROOT / "vision_training/.venv/Scripts/python.exe"
    if not training_python.exists():
        runtime["checkpoint_crosscheck"] = "unavailable_existing_training_environment"
        return runtime
    # weights_only=True loads tensors/primitives, not arbitrary checkpoint code.
    code = """
import json, pathlib, sys, time
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / 'vision_training'))
from tiny_segmentation import TinyMultiLabelUNet
torch.set_num_threads(4)
modeldir=pathlib.Path(sys.argv[1])/'vision_training/task_left_v1/models/left_camera_thick_v1'
checkpoint=torch.load(modeldir/'best.pt',map_location='cpu',weights_only=True)
model=TinyMultiLabelUNet(base=checkpoint['base'],output_channels=len(checkpoint['classes']))
model.load_state_dict(checkpoint['model_state']);model.eval()
rows=[]
for file in sorted(pathlib.Path(sys.argv[2]).glob('*_network_input.png')):
 name=file.stem.removesuffix('_network_input')
 x=torch.from_numpy(np.array(Image.open(file).convert('RGB'),dtype=np.float32).transpose(2,0,1)/255)[None]
 start=time.perf_counter()
 with torch.no_grad():prob=torch.sigmoid(model(x))[0].numpy()[:,:,::-1]
 ms=1000*(time.perf_counter()-start)
 other=np.load(file.parent/(name+'_probabilities.npz'))['probabilities_original_orientation']
 rows.append({'name':name,'onnx_torch_max_abs_probability_error':float(np.max(np.abs(prob-other))),'torch_cpu_forward_ms':ms})
print(json.dumps({'python':sys.executable,'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,
 'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count(),
 'checkpoint_keys':sorted(checkpoint.keys()),'checkpoint_epoch':checkpoint['epoch'],
 'actual_parameter_count':sum(p.numel() for p in model.parameters()),'export_checks':rows}))
"""
    result = subprocess.run([str(training_python), "-c", code, str(ROOT), str(output)],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        runtime["checkpoint_crosscheck_error"] = result.stderr[-2000:]
    else:
        runtime["training_environment_and_checkpoint_crosscheck"] = json.loads(result.stdout)
        checks = runtime["training_environment_and_checkpoint_crosscheck"]["export_checks"]
        runtime["onnx_matches_cpu_torch_within_1e_minus_4"] = all(
            row["onnx_torch_max_abs_probability_error"] < 1e-4 for row in checks)
    return runtime


def annotate(view, points, origin=(0, 0)):
    for index, (x, y) in enumerate(points, 1):
        point = (round(x - origin[0]), round(y - origin[1]))
        cv2.circle(view, point, 9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(view, point, 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, str(index), (point[0] + 11, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
    return view


def focus_panel(image, probabilities, crop, focus, points, classes, title):
    fx1, fy1, fx2, fy2 = focus
    base = image[fy1:fy2, fx1:fx2].copy()
    panels = [annotate(base.copy(), points, (fx1, fy1))]
    captions = [title + " | clean image + query points"]
    x1, y1, x2, y2 = crop
    for index, name in enumerate(classes):
        full = cv2.resize(probabilities[index], (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        # Outside the actual inference crop is UNKNOWN, not background probability zero.
        focus_prob = np.full(base.shape[:2], np.nan, np.float32)
        ix1, iy1, ix2, iy2 = max(fx1, x1), max(fy1, y1), min(fx2, x2), min(fy2, y2)
        if ix2 > ix1 and iy2 > iy1:
            focus_prob[iy1-fy1:iy2-fy1, ix1-fx1:ix2-fx1] = full[iy1-y1:iy2-y1, ix1-x1:ix2-x1]
        heat = cv2.applyColorMap(np.uint8(np.nan_to_num(focus_prob) * 255), cv2.COLORMAP_TURBO)
        heat[~np.isfinite(focus_prob)] = 100
        overlay = cv2.addWeighted(base, .40, heat, .60, 0)
        panels.append(annotate(overlay, points, (fx1, fy1)))
        captions.append(name + " | probability: 0 blue -> 1 red (not identity)")
    rows = []
    for panel, caption in zip(panels, captions):
        if panel.shape[1] > 1100:
            panel = cv2.resize(panel, (1100, round(panel.shape[0] * 1100 / panel.shape[1])))
        row = cv2.copyMakeBorder(panel, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(row, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48,
                    (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    return np.vstack(rows)


def infer(net, image, crop, mirror, input_size):
    x1, y1, x2, y2 = crop
    local = image[y1:y2, x1:x2]
    if mirror:
        local = cv2.flip(local, 1)
    resized = cv2.resize(local, tuple(input_size), interpolation=cv2.INTER_LINEAR)
    blob = cv2.dnn.blobFromImage(resized, 1 / 255., tuple(input_size), swapRB=True, crop=False)
    net.setInput(blob)
    started = time.perf_counter()
    logits = net.forward()[0]
    elapsed_ms = (time.perf_counter() - started) * 1000
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    if probabilities.shape != (3, input_size[1], input_size[0]) or not np.isfinite(probabilities).all():
        raise ValueError("Unexpected or non-finite model output")
    if mirror:
        probabilities = probabilities[:, :, ::-1].copy()
    return probabilities, elapsed_ms, resized


def point_statistics(probabilities, crop, points, classes, radius=12):
    x1, y1, x2, y2 = crop
    full = np.array([cv2.resize(p, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                     for p in probabilities])
    rows = []
    for index, (x, y) in enumerate(points, 1):
        row = {"point_index": index, "original_xy": [float(x), float(y)], "classes": {}}
        if not (x1 <= x < x2 and y1 <= y < y2):
            row["outside_inference_crop"] = True
            rows.append(row)
            continue
        px, py = round(x - x1), round(y - y1)
        for channel, name in enumerate(classes):
            patch = full[channel, max(0, py-radius):py+radius+1, max(0, px-radius):px+radius+1]
            # Remap uses the same original-coordinate probability raster as the overlay.
            value = cv2.remap(full[channel], np.array([[x-x1]], np.float32),
                              np.array([[y-y1]], np.float32), cv2.INTER_LINEAR)[0, 0]
            row["classes"][name] = {"point_probability": float(value),
                                     "patch_radius_original_px": radius,
                                     "patch_min": float(patch.min()),
                                     "patch_median": float(np.median(patch)),
                                     "patch_max": float(patch.max())}
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=LAB / "outputs/wound_curve_v3/manual_247/reference.json")
    parser.add_argument("--output", type=Path,
                        default=LAB / "outputs/learned_curve_v4/existing_model")
    args = parser.parse_args()
    reference = read_json(args.reference)
    metadata = read_json(MODEL_DIR / "onnx_metadata.json")
    model = MODEL_DIR / "left_camera_thick_v1.onnx"
    if metadata["status"] != "shadow_mode_only_not_for_device_control":
        raise ValueError("Expected an explicitly offline/shadow model")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    if model_sha != metadata["sha256"]:
        raise ValueError("Model hash does not match its metadata")
    capture = cv2.VideoCapture(reference["clean_source"])
    capture.set(cv2.CAP_PROP_POS_FRAMES, reference["frame_index"])
    ok, image = capture.read()
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if not ok:
        raise RuntimeError("Could not read the clean source video frame")
    points = np.asarray(reference["points_original_px"], float)
    height, width = image.shape[:2]
    nx1, ny1, nx2, ny2 = metadata["crop_xyxy_normalized"]
    # Left camera was trained on horizontally mirrored canonical frames.  Crop
    # conversion below is exactly flip(full frame) then crop; outputs flip back.
    native = [width-round(nx2*width), round(ny1*height),
              width-round(nx1*width), round(ny2*height)]
    focus = [max(0, int(points[:, 0].min())-180), max(0, int(points[:, 1].min())-150),
             min(width, int(points[:, 0].max())+180), min(height, int(points[:, 1].max())+150)]
    modes = [("native_training_crop", native, False),
             ("whole_frame_OOD", [0, 0, width, height], True),
             ("manual_local_crop_OOD", focus, True)]
    net = cv2.dnn.readNetFromONNX(str(model))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output / "clean_frame.png", image)
    report = {"status": "offline_semantic_probe_only_not_instance_identity",
              "model": str(model), "model_sha256": model_sha,
              "architecture": "TinyMultiLabelUNet(base=8, output_channels=3), 122483 parameters",
              "metadata": metadata, "backend": "OpenCV DNN CPU", "opencv_version": cv2.__version__,
              "source_video": reference["clean_source"], "frame_index": reference["frame_index"],
              "time_s": reference["frame_index"] / fps, "image_size_wh": [width, height],
              "source_frame_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
              "annotation_image_fed_to_model": False, "manual_points_fed_to_model": False,
              "manual_point_semantics": reference.get("annotation_semantics"),
              "probability_semantics": "Independent uncalibrated sigmoid semantic scores; not softmax and not identity confidence",
              "training_audit": audit_training(),
              "left_training_report": read_json(MODEL_DIR / "training_report.json"),
              "limitations": ["No newly-wound same-rope instance, centreline, paired body edges, or temporal identity supervision exists in audited datasets.",
                              "Left training data are 20260922 thick-layer SAM2 propagated teacher labels; native preprocessing does not remove the 20260924 domain shift.",
                              "Some near-contact teacher masks were completed by a straight free-span line, not observed bent rope edges.",
                              "Whole-frame/local recrops are deliberately out of distribution in scale, context and aspect ratio.",
                              "Manual points are rough route hints, not verified body centre or edge ground truth. Their probabilities are diagnostic, not accuracy.",
                              "All outputs are read-only diagnostics. No physical contact, final-turn or device-command inference is made."],
              "minimum_annotation_gap": ["Define target: visible body mask and separately ordered left/right body edges after contact, not strand highlights.",
                                          "Maintain same incoming-rope identity through short frame sequences; mark occlusion, ambiguous identity and unobservable boundaries as ignore/unknown.",
                                          "Include neighbouring rope rows, flange ridges, no-rope and saturation hard negatives; separate capture sessions/cameras across train and held-out tests.",
                                          "Reserve the six-point frame as a diagnostic only; do not train on it then claim validation."],
              "runs": []}
    for name, crop, ood in modes:
        probabilities, ms, network_input = infer(net, image, crop, True, metadata["input_size_wh"])
        np.savez_compressed(args.output / (name + "_probabilities.npz"),
                            probabilities_original_orientation=probabilities,
                            original_crop_xyxy=np.array(crop), classes=np.array(metadata["classes"]))
        write_image(args.output / (name + "_network_input.png"), network_input)
        panel = focus_panel(image, probabilities, crop, focus, points, metadata["classes"], name)
        write_image(args.output / (name + "_focus.jpg"), panel)
        free_index = metadata["classes"].index("incoming_rope_free_span")
        full_view = image.copy()
        x1, y1, x2, y2 = crop
        mask = cv2.resize(probabilities[free_index], (x2-x1, y2-y1)) >= metadata["threshold"]
        local = full_view[y1:y2, x1:x2]
        local[mask] = (.45*local[mask] + .55*np.array([0, 180, 255])).astype(np.uint8)
        cv2.rectangle(full_view, (x1, y1), (x2-1, y2-1), (255, 255, 255), 4)
        annotate(full_view, points)
        write_image(args.output / (name + "_free_mask_full.jpg"), cv2.resize(full_view, (1229, 900)))
        report["runs"].append({"name": name, "out_of_distribution_recrop": ood,
                               "crop_selected_with_manual_points": name == "manual_local_crop_OOD",
                               "horizontal_mirror_input": True, "original_crop_xyxy": crop,
                               "original_pixels_per_output_pixel_xy": [(x2-x1)/512, (y2-y1)/256],
                               "forward_ms_single_cold_or_warm_run_not_benchmark": ms,
                               "output_shape_chw": list(probabilities.shape),
                               "point_statistics": point_statistics(probabilities, crop, points, metadata["classes"]),
                               "identity_confirmed": False})
    report["runtime"] = check_runtime_and_export(args.output)
    path = args.output / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(path), "runs": [
        {"name": run["name"], "ms": run["forward_ms_single_cold_or_warm_run_not_benchmark"],
         "free_span_at_points": [round(p["classes"]["incoming_rope_free_span"]["point_probability"], 4)
                                 for p in run["point_statistics"]]} for run in report["runs"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
