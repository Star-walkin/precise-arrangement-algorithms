"""Explicit synthetic two-rope identity probes; never field ground truth.

The two queries have byte-identical RGB, coordinates and observation support.
Only their prompt channel and the selected instance target differ. Curves,
illumination and strand-like texture are analytically generated, not inferred
from unknown neighbouring ropes in the real dataset.
"""
from __future__ import annotations

import numpy as np

H, W = 256, 512


def make_pair(seed: int, *, test=False):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((H, W), dtype=np.float32)
    t = np.linspace(0, 1, W, dtype=np.float32)
    # Keep the two instances unambiguously separate in this first mechanism test.
    separation = rng.uniform(15, 34) if test else rng.uniform(20, 70)
    midpoint = rng.uniform(85, 170)
    phase = rng.uniform(-np.pi, np.pi)
    shared = rng.uniform(-25, 25) * t**2 + rng.uniform(0, 7)*np.sin(2*np.pi*t+phase)
    rows = np.stack([midpoint-separation/2+shared, midpoint+separation/2+shared])
    rows += np.stack([rng.uniform(-3, 3)*t, rng.uniform(-3, 3)*t])
    width = rng.uniform(3, 5)
    light = .6 + .35*np.cos((xx/W-.3)*np.pi) + .12*np.sin(yy/H*2*np.pi)
    gray = .10 + .018*rng.standard_normal((H, W))
    for index, row in enumerate(rows):
        normal = yy-row[None]
        body = np.exp(-.5*(normal/(width*.42))**2)
        # A bright/dark helical-like appearance is only a synthetic stressor.
        strand = .72 + .22*np.cos(xx*rng.uniform(.35, .7)+normal*.95+index)
        gray += body*strand*rng.uniform(.36, .64)*light
    gray = np.clip(gray*rng.uniform(.7, 1.15), 0, 1).astype(np.float32)
    common = np.zeros((8, H, W), np.float32)
    common[:3] = gray
    common[3] = np.clip((gray-gray.mean())/(gray.std()+.02), -3, 3)/3
    common[5] = np.broadcast_to(np.linspace(-8, 88, W, dtype=np.float32)[None]/88, (H, W))
    common[6] = np.broadcast_to(np.linspace(-24, 24, H, dtype=np.float32)[:, None]/24, (H, W))
    common[7] = 1
    inputs, targets = [], []
    for row in rows:
        value = common.copy()
        value[4] = np.exp(-.5*((yy-row[None])/1.5)**2)*(xx <= 42)
        distribution = np.exp(-.5*((yy-row[None])/1.)**2).astype(np.float32)
        distribution /= distribution.sum(axis=0, keepdims=True)
        targets.append(dict(target_distribution=distribution,
            column_valid=np.ones(W, np.float32), row_valid=np.ones((H, W), np.float32),
            slope_target=np.gradient(row).astype(np.float32), slope_valid=np.ones(W, np.float32)))
        inputs.append(value)
    return dict(inputs=np.stack(inputs), targets=targets, rows=rows.astype(np.float32),
                kind='generated_identity_probe_not_field_geometry', identity_ids=[0, 1])


def assess_pair(probabilities, pair, *, tolerance_rows=2.):
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (2, H, W) or not np.isfinite(probabilities).all():
        raise ValueError('expected finite two-query probabilities')
    predicted = probabilities.argmax(axis=1).astype(float)
    # Exclude the directly drawn prompt: score columns strictly after it.
    chosen = slice(43, W)
    error = np.abs(predicted[:, chosen]-pair['rows'][:, chosen])
    other = np.abs(predicted[:, chosen]-pair['rows'][::-1, chosen])
    fractions = np.mean(error <= tolerance_rows, axis=1)
    return dict(query_coverage_at_2_strip_rows=fractions.tolist(),
        both_queries_pass=bool(np.all(fractions >= .9)),
        median_error_strip_rows=np.median(error, axis=1).tolist(),
        wrong_instance_fraction=np.mean(other < error, axis=1).tolist(),
        median_query_separation_rows=float(np.median(np.abs(predicted[0, chosen]-predicted[1, chosen]))),
        pixels_and_crop_identical=True, only_prompt_channel_changed=True,
        physical_or_field_accuracy=False)
