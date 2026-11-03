"""Short-window, causal live-rope direction evidence for offline validation.

Input y values must be accepted *observations*, never extrapolated positions.
Positive image-y motion means leaving the visible flange in the canonical
(camera-normalized) geometry. This module neither confirms a layer transition
nor reads human-action labels or controls equipment.
"""

from __future__ import annotations

import math
from collections import deque
from statistics import median


class ShortWindowDirectionTracker:
    """Median early/late groups plus repeated support reject isolated outliers.

    Slow or ambiguous motion returns STABLE_OR_REVERSING, not a forced sign.
    The longer existing tracker may remain useful when this supplement cannot
    accumulate 0.35 local rope pitches of movement within its short window.
    """

    def __init__(
        self,
        window_s: float = 1.2,
        minimum_displacement_rope: float = .35,
        minimum_speed_rope_s: float = .12,
        required_support: int = 3,
        maximum_missing_s: float = .6,
        minimum_span_s: float = .35,
    ) -> None:
        if window_s <= 0 or maximum_missing_s < 0 or required_support < 1:
            raise ValueError("Window/support must be positive; missing age nonnegative")
        if minimum_displacement_rope <= 0 or minimum_speed_rope_s < 0:
            raise ValueError("Displacement must be positive and speed nonnegative")
        self.window_s = float(window_s)
        self.minimum_displacement_rope = float(minimum_displacement_rope)
        self.minimum_speed_rope_s = float(minimum_speed_rope_s)
        self.required_support = int(required_support)
        self.maximum_missing_s = float(maximum_missing_s)
        self.minimum_span_s = max(0., float(minimum_span_s))
        self.reset()

    def reset(self) -> None:
        self.samples: deque[tuple[float, float, float]] = deque(maxlen=512)
        self.last_update_time_s: float | None = None
        self.last_observed_time_s: float | None = None
        self.direction = "UNKNOWN"
        self.candidate = "UNKNOWN"
        self.support = 0
        self.displacement_rope = math.nan
        self.speed_rope_s = math.nan

    def update(self, time_s: float, y_or_none: float | None, pitch: float) -> str:
        """Return a direction using only samples at or before ``time_s``.

        Short missing intervals hold the last direction but never add evidence;
        after maximum_missing_s the history is discarded. Callers must not label
        this held direction as a newly measured reversal.
        """
        if not math.isfinite(time_s):
            self.reset()
            return self.direction
        if (self.last_update_time_s is not None
                and time_s <= self.last_update_time_s):
            self.reset()
        if (self.last_observed_time_s is not None
                and time_s - self.last_observed_time_s > self.maximum_missing_s + 1e-9):
            self.reset()
        self.last_update_time_s = time_s

        valid = (y_or_none is not None and math.isfinite(y_or_none)
                 and math.isfinite(pitch) and pitch > 0.)
        if not valid:
            # Missing observations break consecutive support even when the
            # already confirmed direction may be held for a short interval.
            self.candidate = "UNKNOWN"
            self.support = 0
            return self.direction

        self.last_observed_time_s = time_s
        self.samples.append((time_s, float(y_or_none), float(pitch)))
        while self.samples and time_s - self.samples[0][0] > self.window_s + 1e-9:
            self.samples.popleft()
        if len(self.samples) < 6 or time_s - self.samples[0][0] < self.minimum_span_s:
            self.direction = "UNKNOWN"
            self.candidate = "UNKNOWN"
            self.support = 0
            return self.direction

        samples = list(self.samples)
        group_size = max(3, len(samples) // 3)
        early, late = samples[:group_size], samples[-group_size:]
        elapsed = median(v[0] for v in late) - median(v[0] for v in early)
        # Divide a displacement by one local scale; do not divide absolute y by
        # per-frame pitch, which could invent motion when pitch estimates vary.
        scale = median(v[2] for v in samples)
        self.displacement_rope = (median(v[1] for v in late)
                                  - median(v[1] for v in early)) / scale
        self.speed_rope_s = self.displacement_rope / elapsed if elapsed > 0 else 0.
        if (abs(self.displacement_rope) < self.minimum_displacement_rope
                or abs(self.speed_rope_s) < self.minimum_speed_rope_s):
            self.direction = "STABLE_OR_REVERSING"
            self.candidate = "UNKNOWN"
            self.support = 0
            return self.direction

        candidate = "LEAVING" if self.displacement_rope > 0 else "APPROACHING"
        self.support = self.support + 1 if candidate == self.candidate else 1
        self.candidate = candidate
        self.direction = (candidate if self.support >= self.required_support
                          else "STABLE_OR_REVERSING")
        return self.direction
