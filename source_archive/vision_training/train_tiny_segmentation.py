"""Train the first right-camera proof-of-concept segmentation model on CPU."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tiny_segmentation import TinyMultiLabelUNet


ROOT = Path(__file__).resolve().parent
BATCH_SIZE = 4
MODEL_BASE = 8
CLASSES = ['wound_rope_surface', 'flange_inner_face', 'incoming_rope_free_span']


class SegmentationDataset(Dataset):
    def __init__(self, records, manifest, augment=False):
        self.records = records
        self.augment = augment
        self.width, self.height = manifest['input_size_wh']
        self.crop_norm = manifest['crop_xyxy_normalized']

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row['image']) as source:
            image = source.convert('RGB')
        w, h = image.size
        crop = (round(self.crop_norm[0] * w), round(self.crop_norm[1] * h),
                round(self.crop_norm[2] * w), round(self.crop_norm[3] * h))
        image = image.crop(crop).resize((self.width, self.height), Image.Resampling.BILINEAR)
        masks = []
        for class_name in CLASSES:
            with Image.open(Path(row['mask_dir']) / f'{class_name}.png') as mask_source:
                mask = mask_source.convert('L').crop(crop).resize(
                    (self.width, self.height), Image.Resampling.NEAREST)
            masks.append(np.asarray(mask, dtype=np.float32) / 255.0)
        masks = np.stack(masks)
        if self.augment:
            if random.random() < .8:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(.72, 1.28))
            if random.random() < .8:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(.78, 1.22))
            if random.random() < .35:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                masks = masks[:, :, ::-1].copy()
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array), torch.from_numpy(masks.copy())


def dice_loss(logits, targets):
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def evaluate(model, loader):
    model.eval()
    intersection = torch.zeros(len(CLASSES), dtype=torch.float64)
    union = torch.zeros(len(CLASSES), dtype=torch.float64)
    dice_intersection = torch.zeros(len(CLASSES), dtype=torch.float64)
    dice_denominator = torch.zeros(len(CLASSES), dtype=torch.float64)
    with torch.inference_mode():
        for images, targets in loader:
            predictions = torch.sigmoid(model(images)) >= .5
            truth = targets >= .5
            intersection += (predictions & truth).sum(dim=(0, 2, 3)).double()
            union += (predictions | truth).sum(dim=(0, 2, 3)).double()
            dice_intersection += 2 * (predictions & truth).sum(dim=(0, 2, 3)).double()
            dice_denominator += predictions.sum(dim=(0, 2, 3)).double() + truth.sum(dim=(0, 2, 3)).double()
    iou = (intersection / union.clamp_min(1)).tolist()
    dice = (dice_intersection / dice_denominator.clamp_min(1)).tolist()
    return {'iou': dict(zip(CLASSES, iou)), 'dice': dict(zip(CLASSES, dice)),
            'macro_iou': float(sum(iou) / len(iou))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'task_v1' / 'distillation_dataset.json')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'task_v1' / 'models' / 'right_camera_poc_v1')
    parser.add_argument('--epochs', type=int, default=6)
    args = parser.parse_args()
    output = args.output.resolve()
    random.seed(20260922); np.random.seed(20260922); torch.manual_seed(20260922)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    manifest = json.loads(args.manifest.resolve().read_text(encoding='utf-8'))
    train_rows = [r for r in manifest['records'] if r['split'] == 'train']
    validation_rows = [r for r in manifest['records'] if r['split'] == 'validation']
    train_loader = DataLoader(SegmentationDataset(train_rows, manifest, True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_loader = DataLoader(SegmentationDataset(validation_rows, manifest, False),
                                   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = TinyMultiLabelUNet(base=MODEL_BASE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positive_weights = torch.tensor([2.0, 2.0, 20.0]).view(1, 3, 1, 1)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -1.0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=positive_weights)
            loss = bce + dice_loss(logits, targets)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        metrics = evaluate(model, validation_loader)
        row = {'epoch': epoch, 'train_loss': sum(losses)/len(losses), **metrics}
        history.append(row)
        print(f"epoch={epoch} loss={row['train_loss']:.4f} "
              f"macro_iou={row['macro_iou']:.4f} incoming_iou={row['iou']['incoming_rope_free_span']:.4f}",
              flush=True)
        if row['macro_iou'] > best_score:
            best_score = row['macro_iou']
            torch.save({
                'model_state': model.state_dict(), 'base': MODEL_BASE,
                'classes': CLASSES, 'input_size_wh': manifest['input_size_wh'],
                'crop_xyxy_normalized': manifest['crop_xyxy_normalized'],
                'epoch': epoch, 'validation': metrics,
            }, output / 'best.pt')
    report = {
        'schema_version': 1,
        'status': 'proof_of_concept_not_for_device_control',
        'warning': manifest['warning'],
        'train_frames': len(train_rows), 'validation_frames': len(validation_rows),
        'epochs': args.epochs, 'elapsed_s': time.perf_counter()-started,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'best_macro_iou': best_score, 'history': history,
    }
    (output / 'training_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"saved={output/'best.pt'} elapsed={report['elapsed_s']:.1f}s params={report['parameter_count']}",
          flush=True)


if __name__ == '__main__':
    main()
