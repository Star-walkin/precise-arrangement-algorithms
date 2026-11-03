"""Hash-bound annotation uncertainty; never edits the human label or supplies input features.

Quality regions override precision supervision/scoring only. An inferred or
coarse-direction annotation is neither exact geometry nor background evidence.
"""
from pathlib import Path
import hashlib
import json
import math


QUALITY_KINDS = {'context_inferred', 'coarse_direction_only'}


def quality_path(sample):
    return Path(sample['annotation_path']).with_suffix('.quality.json')


def load_label_quality(sample):
    """Return validated exclusions; an absent sidecar has an explicit null hash."""
    path = quality_path(sample)
    empty = dict(quality_sha256=None, precision_exclude_polygons=[], regions=[])
    if not path.exists():
        return empty
    raw = path.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('Unsupported annotation quality schema')
    if data.get('sample_id') != sample['sample_id']:
        raise ValueError('Quality sample_id mismatch')
    if data.get('image_sha256') != sample['image_sha256']:
        raise ValueError('Quality image_sha256 mismatch')
    annotation_hash = hashlib.sha256(Path(sample['annotation_path']).read_bytes()).hexdigest()
    if data.get('annotation_sha256') != annotation_hash:
        raise ValueError('Quality annotation_sha256 mismatch; review changed label again')
    if data.get('coordinate_system') != 'full_image_xy':
        raise ValueError('Quality coordinates must be full_image_xy')
    regions = data.get('regions')
    if not isinstance(regions, list):
        raise ValueError('Quality regions must be a list')
    shape = sample.get('image_shape_hw')
    if not (isinstance(shape, list) and len(shape) == 2 and
            all(isinstance(v, int) and v > 0 for v in shape)):
        raise ValueError('Quality requires image_shape_hw')
    height, width = shape
    seen = set()
    polygons = []
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError('Quality region must be an object')
        region_id = region.get('region_id')
        if not isinstance(region_id, str) or not region_id or region_id in seen:
            raise ValueError('Quality region_id must be unique and nonempty')
        seen.add(region_id)
        if region.get('kind') not in QUALITY_KINDS:
            raise ValueError('Unsupported quality kind')
        if region.get('exclude_from_precision') is not True:
            raise ValueError('Uncertain regions must be excluded from precision')
        polygon = region.get('polygon_xy')
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError('Quality polygon requires at least three vertices')
        points = []
        for point in polygon:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError('Quality polygon vertices must be [x,y]')
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in point):
                raise ValueError('Quality polygon must be finite')
            x, y = map(float, point)
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError('Quality polygon outside source image')
            points.append([x, y])
        area2 = sum(a[0]*b[1]-b[0]*a[1]
                    for a, b in zip(points, points[1:]+points[:1]))
        if abs(area2) < 1e-6:
            raise ValueError('Quality polygon has zero area')
        polygons.append(points)
    result = dict(quality_sha256=hashlib.sha256(raw).hexdigest(),
                  precision_exclude_polygons=polygons, regions=regions)
    if 'trusted_supervision_polylines' in data or 'trusted_supervision_rule' in data:
        # Only the explicitly confirmed V8 cross-section format has this field.
        # Validate against the hash-bound original; never change detector input.
        from .centerline_quality_cutoff_v8 import validate_trusted_supervision
        annotation = json.loads(Path(sample['annotation_path']).read_text('utf-8-sig'))
        result['supervision_polylines'] = validate_trusted_supervision(sample, annotation, data)
    return result
