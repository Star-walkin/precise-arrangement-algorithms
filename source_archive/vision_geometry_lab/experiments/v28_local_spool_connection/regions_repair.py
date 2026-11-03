"""Image-only V28 repair: coarse spool search and distinct crossing corridors.

The existing camera ONNX is reused offline.  A second full-width prediction
removes its metadata crop's artificial side truncation; neither prediction is
rope identity evidence.  The predicted free class is deliberately not a search
permit: on the open left debug frames it includes a button-panel edge.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PROTOTYPE = ROOT / "vision_prototype"
if str(PROTOTYPE) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE))
from semantic_locator import semantic_locator_for_camera
from surface_centerlines import source_wound_mask

_LOCATORS = {}


def _locator(camera):
    if camera not in _LOCATORS:
        _LOCATORS[camera] = semantic_locator_for_camera(camera)
    return _LOCATORS[camera]


def _source_class(regions, name, shape):
    full = np.zeros(shape[:2], np.uint8)
    x0, y0, x1, y1 = regions.crop_xyxy
    full[y0:y1, x0:x1] = cv2.resize(regions.mask(name).astype(np.uint8),
                                  (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST)
    return full.astype(bool)


def _largest_near(mask, reference, radius):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count < 2:
        return np.zeros_like(mask)
    near = cv2.distanceTransform((~reference).astype(np.uint8), cv2.DIST_L2, 5) <= radius
    choices = []
    for index in range(1, count):
        component = labels == index
        if stats[index, cv2.CC_STAT_AREA] >= 1000 and np.any(component & near):
            choices.append((int(stats[index, cv2.CC_STAT_AREA]), index))
    return labels == max(choices)[1] if choices else np.zeros_like(mask)


def _line_mask(shape, points, radius):
    mask = np.zeros(shape[:2], np.uint8)
    if points is None:
        return mask.astype(bool)
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        return mask.astype(bool)
    cv2.polylines(mask, [np.round(points).astype(np.int32)], False, 1,
                  max(1, int(round(2*radius))), cv2.LINE_8)
    return mask.astype(bool)


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)] if len(xs) else None


def detect_regions(image_bgr, free_seed, width_px, camera="left", arc=None):
    """Return full-original bool masks and JSON-safe summary, with no label input.

    ``spool_mask`` alone permits old-layer rope structures. ``search_mask`` also
    permits hypotheses across unknown transitions and the measured free corridor.
    ``background_mask`` is a compatibility alias for prohibited search pixels;
    only ``explicit_background`` is reported as image-based background exclusion.
    """
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("A full original BGR frame is required")
    shape = image_bgr.shape
    width = float(width_px) if width_px is not None and np.isfinite(width_px) else 24.
    width = max(5., width)
    locator = _locator(camera)
    original_crop = tuple(locator.crop_norm)
    primary = locator.predict(image_bgr)
    # Reuse exactly the local model. This adds coverage, not a learned component.
    supplemental_crop = (0., original_crop[1], 1., original_crop[3])
    try:
        locator.crop_norm = supplemental_crop
        supplemental = locator.predict(image_bgr)
    finally:
        locator.crop_norm = original_crop
    primary_wound = source_wound_mask(primary, shape)
    supplementary_wound = source_wound_mask(supplemental, shape)
    wound = primary_wound | supplementary_wound
    wound = cv2.morphologyEx(wound.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((9, 9), np.uint8)).astype(bool)
    raw_flange = (_source_class(primary, "flange_inner_face", shape) |
                  _source_class(supplemental, "flange_inner_face", shape))
    flange = _largest_near(raw_flange, wound, 3*width)
    # The seed is original-pixel observed free-span geometry. The network's free
    # class can be inspected, but it cannot admit machine-panel line segments.
    raw_semantic_free = (_source_class(primary, "incoming_rope_free_span", shape) |
                         _source_class(supplemental, "incoming_rope_free_span", shape))
    free = _line_mask(shape, free_seed, .85*width)
    distance = cv2.distanceTransform((~wound).astype(np.uint8), cv2.DIST_L2, 5)
    transition = (distance <= 4*width) & ~wound
    forward_unknown = np.zeros(shape[:2], bool)
    seed = np.asarray(free_seed, float) if free_seed is not None else np.empty((0, 2))
    forward_end = None
    if seed.ndim == 2 and seed.shape[1] == 2 and len(seed) >= 2 and np.isfinite(seed).all():
        direction = seed[-1]-seed[0]
        norm = np.linalg.norm(direction)
        if norm > 1:
            direction /= norm
            lengths = np.arange(0., min(100*width, np.hypot(*shape[:2])), max(2., width/4))
            xy = seed[-1] + lengths[:, None]*direction
            valid = ((xy[:,0] >= 0) & (xy[:,0] < shape[1]) &
                     (xy[:,1] >= 0) & (xy[:,1] < shape[0]))
            xy = xy[valid]
            if len(xy):
                ix = np.clip(np.round(xy[:,0]).astype(int),0,shape[1]-1)
                iy = np.clip(np.round(xy[:,1]).astype(int),0,shape[0]-1)
                arrival = np.flatnonzero(wound[iy,ix])
                # Ray extension reaches only the first observed coarse body.
                # If the ray never reaches it, no long arbitrary corridor opens.
                if len(arrival):
                    forward_end = xy[arrival[0]].tolist()
                    forward_unknown = _line_mask(shape,[seed[-1],xy[arrival[0]]],1.75*width) & ~wound & ~free
                    transition |= forward_unknown
    transition &= ~free
    search = wound | transition | free
    covered = np.zeros(shape[:2], bool)
    sx0,sy0,sx1,sy1 = supplemental.crop_xyxy
    covered[sy0:sy1,sx0:sx1] = True
    # Coarse location is an exclusion aid only. Near painted flange remains
    # explicitly unknown for a possible overlying rope, never an old-layer band.
    near_flange = cv2.distanceTransform((~flange).astype(np.uint8),cv2.DIST_L2,5) <= 2*width if flange.any() else flange
    explicit_background = covered & ~search & ~near_flange
    outside = ~search
    unknown = transition | (outside & ~explicit_background)
    masks = dict(spool_mask=wound, wound_mask=wound, flange_mask=flange,
                 free_corridor=free, transition_mask=transition, search_mask=search,
                 background_mask=outside, outside_search_mask=outside,
                 explicit_background=explicit_background, unknown_mask=unknown,
                 semantic_primary_wound=primary_wound,
                 semantic_supplementary_wound=supplementary_wound,
                 semantic_raw_free=raw_semantic_free, semantic_raw_flange=raw_flange,
                 forward_unknown_corridor=forward_unknown, semantic_covered_mask=covered)
    summary = dict(schema="v28_regions_repair_1", coordinate_system="full_original_image_xy",
                   image_shape_hw=list(shape[:2]), input_uses_labels=False,
                   geometry_completion_is_evidence=False, camera=camera,
                   model_path=str(locator.model_path), metadata_crop_norm=list(original_crop),
                   supplemental_crop_norm=list(supplemental_crop),
                   original_crop_xyxy=list(primary.crop_xyxy), supplemental_crop_xyxy=list(supplemental.crop_xyxy),
                   width_reference_px=width, transition_radius_px=4*width,
                   forward_corridor_end_xy=forward_end,
                   semantic_free_is_search_permission=False,
                   background_mask_note="prohibited_search_compatibility_alias; explicit_background_is_separate",
                   unknown_note="near-spool crossing hypotheses and unsearched pixels are not measured rope or proved background",
                   masks={name: dict(area_px=int(mask.sum()),bbox_xyxy=_bbox(mask)) for name,mask in masks.items()},
                   removed_flange_island_area_px=int((raw_flange & ~flange).sum()),
                   removed_semantic_free_area_px=int((raw_semantic_free & ~free & ~transition & ~wound).sum()),
                   status="AUTO_COARSE_REGIONS" if wound.any() else "AUTO_REGION_UNAVAILABLE")
    return dict(**masks, summary=summary)


def render_regions(image_bgr, regions):
    """Region verification on original pixels, independent of path rendering."""
    canvas = image_bgr.copy()
    colors = {"explicit_background":(80,70,180),"flange_mask":(10,180,245),
              "transition_mask":(220,80,220),"spool_mask":(50,210,50),"free_corridor":(255,170,30)}
    for name,color in colors.items():
        mask=regions[name]
        canvas[mask]=(.75*canvas[mask]+.25*np.asarray(color)).astype(np.uint8)
        if name != "explicit_background":
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas,contours,-1,color,4)
    canvas=cv2.resize(canvas,(2048,1500))
    for row,(name,color) in enumerate(colors.items()):
        cv2.putText(canvas,name,(20,45+38*row),cv2.FONT_HERSHEY_SIMPLEX,1.,color,2,cv2.LINE_AA)
    return canvas


def save_regions(output_dir, image_bgr, regions):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output_dir/"regions.npz",**{key:value for key,value in regions.items() if isinstance(value,np.ndarray)})
    (output_dir/"regions.json").write_text(json.dumps(regions["summary"],ensure_ascii=False,indent=2),encoding="utf-8")
    for name,image in (("regions_overlay.png",render_regions(image_bgr,regions)),
                       ("original.png",cv2.resize(image_bgr,(2048,1500)))):
        ok,encoded=cv2.imencode(".png",image)
        if not ok:
            raise ValueError("PNG encoding failed")
        encoded.tofile(output_dir/name)
