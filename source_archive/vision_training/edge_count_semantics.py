"""Direction-aware labels for axial distance from the camera's observed edge.

An edge distance in rope pitches is not, by itself, a measured spool revolution.
"""
from __future__ import annotations

import math


OBSERVED_EDGE = {'right': '左', 'left': '右'}
TOWARD_DIRECTION = {'right': '向左', 'left': '向右'}
AWAY_DIRECTION = {'right': '向右', 'left': '向左'}


def count_kind(camera: str, physical_direction: str) -> str:
    if camera not in OBSERVED_EDGE:
        raise ValueError(f'unknown camera role: {camera}')
    if physical_direction == TOWARD_DIRECTION[camera]:
        return 'remaining'
    if physical_direction == AWAY_DIRECTION[camera]:
        return 'departed'
    return 'undetermined'


def parse_count(value) -> float | None:
    try:
        count = float(value)
    except (TypeError, ValueError):
        return None
    return count if math.isfinite(count) and count >= 0 else None


def gap_from_human_count(kind: str, count: float | None) -> float | None:
    """Map the user's convention to an axial gap measured in rope pitches."""
    if count is None:
        return None
    if kind == 'remaining':
        return count
    if kind == 'departed' and count >= 1:
        return count - 1
    return None


def human_count_from_gap(kind: str, gap: float | None) -> float | None:
    if gap is None or not math.isfinite(gap) or gap < 0:
        return None
    if kind == 'remaining':
        return gap
    if kind == 'departed':
        return gap + 1
    return None


def display_count(annotation: dict) -> str:
    for key in ('edge_distance_rope_count', 'departed_rope_count',
                'remaining_rope_count'):
        count = parse_count(annotation.get(key))
        if count is not None:
            return f'{count:g}'
    return 'unknown'


def update_directional_count(annotation: dict, camera: str,
                             physical_direction: str, count_text: str) -> dict:
    """Return a copy; preserve old unknown-direction labels until reviewed."""
    revised = dict(annotation)
    kind = count_kind(camera, physical_direction)
    count = parse_count(count_text)
    value = f'{count:g}' if count is not None else 'unknown'
    revised['physical_direction'] = physical_direction
    revised['count_reference_edge'] = OBSERVED_EDGE[camera]
    revised['edge_distance_rope_count'] = value
    revised['edge_distance_count_kind'] = kind
    gap = gap_from_human_count(kind, count)
    revised['edge_gap_rope_count'] = f'{gap:g}' if gap is not None else 'unknown'
    revised['counting_convention'] = {
        'remaining': 'remaining_excludes_active_rope',
        'departed': 'departed_includes_active_rope',
    }.get(kind, 'direction_undetermined')
    if kind == 'remaining':
        revised['remaining_rope_count'] = value
        revised['departed_rope_count'] = 'unknown'
    elif kind == 'departed':
        revised['remaining_rope_count'] = 'unknown'
        revised['departed_rope_count'] = value
    # When direction is unknown, do not destroy a legacy remaining count.
    return revised
