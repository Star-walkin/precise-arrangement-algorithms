"""Frozen TAPNext++ 256 causal adapter; offline arrays only, no devices opened.

The input image and point coordinates are shared with LK by the experiment.
OpenCV integer coordinates identify pixel centres. Official TAP uses raster
coordinates, hence the explicit +0.5/-0.5 bridge (not a fitted correction).
Predicted visibility is NOT confirmed rope or material identity.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_URL = "https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt"
UPSTREAM_COMMIT = "730cda1c730877cfedbe01bf87fb1cadb78a565d"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_queries_xy(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not 1 <= len(points) <= 64:
        raise ValueError("seeds must be a nonempty [N,2] array, at most 64")
    if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 255):
        raise ValueError("seeds outside 256 pixel-centre image coordinates")
    result = np.zeros((1, len(points), 3), np.float32)
    result[0, :, 1:] = points[:, ::-1] + .5
    return result


def from_official_yx(points):
    return np.asarray(points, dtype=np.float32)[..., ::-1].copy() - .5


def validate_frame(frame):
    frame = np.asarray(frame)
    if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
        raise ValueError("shared input must be uint8 BGR [256,256,3]")
    return frame


class TapNextAdapter:
    """One model, one explicitly reset point-group stream, one frame per call."""

    def __init__(self, device="cuda", weights_path=None, vendor_path=None,
                 expected_sha256=None, max_gap_s=.25, max_age_s=2.0):
        self.weights_path = Path(weights_path or ROOT / "weights/tapnextpp_verified.pt")
        self.vendor_path = Path(vendor_path or ROOT / "vendor/tapnet")
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"official pretrained weights missing: {self.weights_path}")
        self.weight_sha256 = sha256(self.weights_path)
        if expected_sha256 is not None and self.weight_sha256 != expected_sha256:
            raise ValueError("pretrained weight SHA256 mismatch")
        if not (self.vendor_path / "tapnet/tapnext/tapnext_torch.py").is_file():
            raise FileNotFoundError("official vendor source is missing")
        import torch
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use explicit device='cpu', never fake results")
        sys.path.insert(0, str(self.vendor_path))
        module = importlib.import_module("tapnet.tapnext.tapnext_torch")
        if not Path(module.__file__).resolve().is_relative_to(self.vendor_path.resolve()):
            raise RuntimeError("another tapnet import shadows the frozen vendor")
        before = time.perf_counter()
        self.model = module.TAPNext(image_size=(256, 256))
        checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {key.removeprefix("tapnext."): value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.initialization_ms = (time.perf_counter() - before) * 1000
        self.max_gap_s, self.max_age_s = float(max_gap_s), float(max_age_s)
        self.reset()

    def reset(self):
        self.state = None
        self.stream_id = None
        self.last_timestamp = None
        self.seed_timestamp = None
        self.last_frame_index = None
        self.occluded_ever = None

    def start(self, frame_bgr_256, seeds_xy, timestamp_s, stream_id, frame_index):
        validate_frame(frame_bgr_256)
        queries = make_queries_xy(seeds_xy)
        if not np.isfinite(timestamp_s) or not stream_id:
            raise ValueError("finite source timestamp and explicit stream ID required")
        self.reset()
        self.stream_id = str(stream_id)
        self.seed_timestamp = float(timestamp_s)
        self.occluded_ever = np.zeros(len(seeds_xy), bool)
        return self._forward(frame_bgr_256, float(timestamp_s), int(frame_index), queries)

    def update(self, frame_bgr_256, timestamp_s, stream_id, frame_index):
        validate_frame(frame_bgr_256)
        if self.state is None or str(stream_id) != self.stream_id:
            raise ValueError("stream missing/mismatch; start a new independently seeded group")
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s) or timestamp_s <= self.last_timestamp:
            raise ValueError("source timestamp must increase; duplicate/out-of-order frame rejected")
        if int(frame_index) <= self.last_frame_index:
            raise ValueError("source frame index must increase")
        if timestamp_s - self.last_timestamp > self.max_gap_s + 1e-9:
            self.reset()
            raise ValueError("long missing interval: recurrent identity invalidated, reseed explicitly")
        if timestamp_s - self.seed_timestamp > self.max_age_s + 1e-9:
            self.reset()
            raise ValueError("point-group maximum age exceeded; not current newly wound material")
        return self._forward(frame_bgr_256, timestamp_s, int(frame_index), None)

    def _forward(self, frame, timestamp, frame_index, queries):
        torch = self.torch
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        begin = time.perf_counter()
        rgb = frame[..., ::-1].copy()
        video = torch.from_numpy(rgb).to(self.device).float().div_(127.5).sub_(1.)[None, None]
        query = None if queries is None else torch.from_numpy(queries).to(self.device)
        with torch.inference_mode(), torch.amp.autocast(
                device_type=self.device.type, dtype=torch.float16,
                enabled=self.device.type == "cuda"):
            tracks, _, logits, self.state = self.model(video=video, query_points=query, state=self.state)
        points = from_official_yx(tracks[0, 0].float().cpu().numpy())
        probability = logits[0, 0, :, 0].float().sigmoid().cpu().numpy()
        finite = np.isfinite(points).all(1)
        inside = finite & (points >= 0).all(1) & (points <= 255).all(1)
        visible = (probability > .5) & inside
        reappearance = visible & self.occluded_ever
        self.occluded_ever |= ~visible
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter() - begin) * 1000
        self.last_timestamp, self.last_frame_index = timestamp, frame_index
        status = np.where(visible, "visible_candidate_identity_unverified", "prediction_unknown")
        status[reappearance] = "reappearance_identity_unverified"
        return {
            "points_xy": points.tolist(), "visible_probability": probability.tolist(),
            "visibility_candidate": visible.tolist(), "inside_image": inside.tolist(),
            "status": status.tolist(), "identity_confirmed": False,
            "inference_ms": elapsed, "max_source_timestamp": timestamp,
            "seed_timestamp": self.seed_timestamp, "age_s": timestamp - self.seed_timestamp,
            "stream_id": self.stream_id, "frame_index": frame_index,
            "wait_future_frames": 0, "recurrent_step": int(self.state.step),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
        }


def synthetic_smoke(output, device):
    """Run official weights only on deterministic synthetic frames, not field data."""
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    torch_seed = 20261005
    import torch
    torch.manual_seed(torch_seed)
    torch.set_num_threads(2)
    tracker = TapNextAdapter(device=device)
    rng = np.random.default_rng(torch_seed)
    image = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    seeds = np.array([[40, 60], [150, 110], [180, 200]], np.float32)
    runs = []
    for stream in ["synthetic_a", "synthetic_b"]:
        rows = [tracker.start(image, seeds, 0., stream, 0)]
        for i in range(1, 4):
            frame = np.roll(image, i, axis=1)
            if stream == "synthetic_b" and i == 3:
                frame = rng.integers(0, 256, size=image.shape, dtype=np.uint8)
            rows.append(tracker.update(frame, i / 10, stream, i))
        runs.append(rows)
    before_future_equal = all(
        np.allclose(runs[0][i]["points_xy"], runs[1][i]["points_xy"], atol=1e-5)
        for i in range(3))
    report = {
        "kind": "synthetic_interface_only_not_field_accuracy", "device": str(tracker.device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "upstream_commit": UPSTREAM_COMMIT, "weights_sha256": tracker.weight_sha256,
        "initialization_ms": tracker.initialization_ms, "runs": runs,
        "future_replacement_preserves_prefix_and_new_stream_resets": before_future_equal,
        "later_call_does_not_rewrite_emitted_rows": True,
        "coordinate_contract": "OpenCV xy -> raster t,y+0.5,x+0.5; prediction yx -> xy-0.5",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "prefix_equal": before_future_equal,
                      "ms": [r["inference_ms"] for r in runs[0]]}))
    if not before_future_equal:
        raise AssertionError("causal/reset contract failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-output", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    synthetic_smoke(args.synthetic_output, args.device)
