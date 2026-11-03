"""Read-only semantic ROI locator for the hybrid vision proof of concept.

The network output is never a device command and is not itself a precision
measurement.  It only constrains the original-resolution geometry search.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SemanticRegions:
    crop_xyxy: tuple[int, int, int, int]
    probabilities: np.ndarray
    masks: np.ndarray
    classes: tuple[str, ...]

    def class_index(self, name: str) -> int:
        return self.classes.index(name)

    def mask(self, name: str) -> np.ndarray:
        return self.masks[self.class_index(name)]

    def probability(self, name: str) -> np.ndarray:
        return self.probabilities[self.class_index(name)]

    def point_to_source(self, x: float, y: float) -> tuple[float, float]:
        x1, y1, x2, y2 = self.crop_xyxy
        height, width = self.masks.shape[1:]
        return x1 + x * (x2 - x1) / width, y1 + y * (y2 - y1) / height

    def point_from_source(self, x: float, y: float) -> tuple[float, float]:
        x1, y1, x2, y2 = self.crop_xyxy
        height, width = self.masks.shape[1:]
        return (x - x1) * width / (x2 - x1), (y - y1) * height / (y2 - y1)


class SemanticLocator:
    """OpenCV-DNN wrapper around the 505 KiB distilled ONNX model."""

    def __init__(self, model_path: Path | str, metadata_path: Path | str):
        self.model_path = Path(model_path).resolve()
        self.metadata_path = Path(metadata_path).resolve()
        self.metadata = json.loads(self.metadata_path.read_text(encoding='utf-8'))
        if self.metadata.get('status') != 'shadow_mode_only_not_for_device_control':
            raise ValueError('semantic model is missing its shadow-mode safety marker')
        self.classes = tuple(self.metadata['classes'])
        self.input_width, self.input_height = self.metadata['input_size_wh']
        self.crop_norm = tuple(float(v) for v in self.metadata['crop_xyxy_normalized'])
        self.threshold = float(self.metadata['threshold'])
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        output = np.empty_like(logits, dtype=np.float32)
        positive = logits >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponential = np.exp(logits[~positive])
        output[~positive] = exponential / (1.0 + exponential)
        return output

    def predict(self, frame_bgr: np.ndarray) -> SemanticRegions:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError('frame_bgr must be a nonempty BGR image')
        height, width = frame_bgr.shape[:2]
        x1 = max(0, min(width - 1, round(self.crop_norm[0] * width)))
        y1 = max(0, min(height - 1, round(self.crop_norm[1] * height)))
        x2 = max(x1 + 1, min(width, round(self.crop_norm[2] * width)))
        y2 = max(y1 + 1, min(height, round(self.crop_norm[3] * height)))
        crop = frame_bgr[y1:y2, x1:x2]
        resized = cv2.resize(crop, (self.input_width, self.input_height),
                             interpolation=cv2.INTER_LINEAR)
        # swapRB converts the application's BGR frame to the RGB order used in
        # training. No mean/std normalization was used during training.
        blob = cv2.dnn.blobFromImage(resized, scalefactor=1.0 / 255.0,
                                     size=(self.input_width, self.input_height),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        logits = self.net.forward()[0]
        probabilities = self._sigmoid(logits)
        return SemanticRegions(
            crop_xyxy=(x1, y1, x2, y2), probabilities=probabilities,
            masks=probabilities >= self.threshold, classes=self.classes,
        )


def default_semantic_locator() -> SemanticLocator:
    model_dir = (Path(__file__).resolve().parent.parent / 'vision_training' /
                 'task_v1' / 'models' / 'right_camera_poc_v1')
    return SemanticLocator(model_dir / 'right_camera_poc_v1.onnx',
                           model_dir / 'onnx_metadata.json')


def semantic_locator_for_camera(camera_identifier: str) -> SemanticLocator:
    """Return the camera-specific shadow model without guessing orientation."""
    identifier = camera_identifier.strip().lower()
    training = Path(__file__).resolve().parent.parent / 'vision_training'
    if identifier in {'left', 'left_camera', 'da9626065', 'da9627065',
                      'left_camera_observes_right_flange'}:
        model_dir = training / 'task_left_v1' / 'models' / 'left_camera_thick_v1'
        return SemanticLocator(model_dir / 'left_camera_thick_v1.onnx',
                               model_dir / 'onnx_metadata.json')
    if identifier in {'right', 'right_camera', 'da9626975',
                      'right_camera_observes_left_flange'}:
        return default_semantic_locator()
    raise ValueError(f'No semantic model configured for camera: {camera_identifier}')
