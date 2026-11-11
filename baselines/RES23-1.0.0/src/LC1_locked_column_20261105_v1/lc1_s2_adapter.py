"""Faithful S2 evidence adapter; explicit P1 seed is an input condition.

Executes the supplied straight_core.py unchanged. It never fills UNKNOWN
intervals, declares a curved path, or turns optical support into rope identity.
S2 keeps the supplied straight axis; it does not independently re-estimate its
slope. The optional guidance.py auto proposer is intentionally not executed in
this conditioned two-frame smoke test (its A7 location is a separate dependency).
"""
from pathlib import Path
import hashlib
import json
import types
import numpy as np

DEFAULT_S2_DIR = Path(__file__).resolve().parent/'supplied_assets/straight_gradient_20261105_v2'
SOURCE = 'IR-GRAD-S2.0.0 original core, explicit current-P1-seed-conditioned evidence'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class S2Adapter:
    def __init__(self, source_dir=DEFAULT_S2_DIR, *, config_overrides=None,
                 retain_step_records=False):
        self.source_dir = Path(source_dir).resolve()
        core_path = self.source_dir/'straight_core.py'
        guidance_path = self.source_dir/'guidance.py'
        self.hashes = dict(core_sha256=_sha(core_path), guidance_sha256=_sha(guidance_path))
        self.overrides = dict(config_overrides or {})
        self.retain_step_records = bool(retain_step_records)
        # Exact supplied bytes, with no writes/__pycache__ in the source assets.
        module = types.ModuleType('_lc1_supplied_original_s2')
        module.__file__ = str(core_path)
        exec(compile(core_path.read_bytes(), str(core_path), 'exec'), module.__dict__)
        self.core = module
        self.config = dict(module.DEFAULTS) | self.overrides
        self._guidance = None
        self._guidance_relocation = None

    def _load_guidance(self):
        if self._guidance is None:
            source = self.source_dir/'guidance.py'
            module = types.ModuleType('_lc1_supplied_original_s2_guidance')
            module.__file__ = str(source)
            exec(compile(source.read_bytes(), str(source), 'exec'), module.__dict__)
            recovery = (Path(__file__).resolve().parent/'p1_runtime/source_archive/数据资产/05_派生与实验结果'
                        /'camera_condition_free_straight_20261030_v7/free_seed_recovery.py')
            if not recovery.is_file():
                raise FileNotFoundError('Original guidance A7 dependency is unavailable: '+str(recovery))
            self._guidance_relocation = dict(original_module_A7_RECOVERY=str(module.A7_RECOVERY),
                relocated_A7_RECOVERY=str(recovery), relocated_file_sha256=_sha(recovery),
                source_file_modified=False, change_scope='loaded_module_attribute_only')
            module.A7_RECOVERY = recovery
            self._guidance = module
        return self._guidance

    def step_auto(self, original_bgr, *, e1_xy=None, extra_seeds=None,
                  surface_mask=None, pitch_px=None, guidance_config=None):
        """Optional original guidance + original S2; ambiguity stays explicit.

        All proposed axes are evaluated by the same original S2 core. Only the
        axis selected by the unchanged guidance decision can become the output;
        a longer S2 result does not override AMBIGUOUS/UNKNOWN or supply identity.
        e1_xy=None allows image-only A7 proposals; a supplied E1 remains a
        separately labelled competing condition, not ground truth.
        """
        guidance = self._load_guidance()
        proposals = guidance.propose_guidance(original_bgr, e1_xy,
            extra_seeds=extra_seeds, surface_mask=surface_mask, pitch_px=pitch_px,
            config=guidance_config)
        results = []
        selected = None
        for candidate in proposals['candidates']:
            width = candidate.get('width_px')
            width = 22. if width is None else float(width)
            result = self.step(original_bgr, candidate['seed_xy'],
                seed_source='original_current_guidance:'+candidate['source'],
                width_px=width, candidate_id=candidate['candidate_id'])
            results.append(dict(candidate_id=candidate['candidate_id'], result=result))
            if (proposals['status'] == 'SELECTED' and
                    candidate['candidate_id'] == proposals['selected_candidate_id']):
                selected = result
        if selected is None:
            selected = self.step(original_bgr, None,
                seed_source='original_guidance_has_no_unambiguous_selected_axis', width_px=22.)
        selected['diagnostics'].update(guidance_executed=True,
            guidance_result=proposals, guidance_A7_relocation=self._guidance_relocation,
            guidance_source_sha256=self.hashes['guidance_sha256'],
            s2_auto_candidate_results=results,
            candidate_selection_uses_unchanged_guidance_not_S2_extent=True)
        # Do not put selected itself in its own nested candidates list.
        # A shallow copy removes the recursive list from per-candidate records.
        for row in results:
            if row['result'] is selected:
                copy_result = dict(selected)
                copy_result['diagnostics'] = {k:v for k,v in selected['diagnostics'].items()
                                              if k!='s2_auto_candidate_results'}
                row['result'] = copy_result
        selected['whole_path_candidate_supported'] = False
        selected['same_rope_identity_confirmed'] = False
        return selected

    def step(self, original_bgr, seed_xy, *, seed_source, width_px,
             candidate_id='explicit_current_P1_A7'):
        if not isinstance(seed_source, str) or not seed_source.strip():
            raise ValueError('Explicit seed_source is required')
        provenance = dict(source=SOURCE, source_dir=str(self.source_dir), **self.hashes,
            configuration=dict(self.config), config_overrides=self.overrides,
            coordinate_system='original_unmirrored_image_xy', seed_source=seed_source,
            seed_is_input_not_new_detection=True, guidance_executed=False,
            line_abc_is_conditioned_axis_not_independently_refitted=True,
            current_bilateral_profiles_do_not_establish_material_identity=True,
            optical_endpoint_is_not_physical_turn=True)
        base = dict(valid=False, source=SOURCE, s2_status='UNKNOWN', line_abc=None,
            line_points_xy=[], measured_center_points_xy=[], supported_segments_xy=[],
            unknown_intervals_px=[], unknown_segments_xy=[], path_xy=[],
            whole_path_candidate_supported=False, same_rope_identity_confirmed=False,
            optical_endpoint_xy=None, diagnostics=provenance)
        if seed_xy is None:
            provenance['reason'] = 'missing_explicit_seed'
            return base
        if (not np.isfinite(width_px) or not self.config['width_min_px'] <= float(width_px) <= self.config['width_max_px']):
            raise ValueError('Explicit optical width must be finite and within original S2 limits')
        candidate = dict(candidate_id=candidate_id, seed_xy=seed_xy, width_px=float(width_px))
        prediction = self.core.analyze_straight(original_bgr, candidate, self.overrides)
        origin = np.asarray(prediction['line_origin_xy'], float)
        direction = np.asarray(prediction['line_direction_xy'], float)
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        intervals = prediction['supported_intervals_px']
        selected_records = [r for r in prediction['step_records'] if r['supported'] and
                            any(a-1e-5 <= r['distance_after_seed_px'] <= b+1e-5 for a, b in intervals)]
        supported = []
        shifted = []
        segments = []
        step = float(self.config['step_px'])
        for a, b in intervals:
            records = [r for r in selected_records if a-1e-5 <= r['distance_after_seed_px'] <= b+1e-5]
            # Preserve every selected supported run separately; do not connect
            # either rejected candidates or gaps on the conditioned axis.
            segment = []
            previous_t = None
            for record in records:
                t = float(record['distance_after_seed_px'])
                if previous_t is not None and t-previous_t > step*1.1:
                    if segment:
                        segments.append(segment)
                    segment = []
                center = origin+t*direction
                segment.append(center.tolist())
                supported.append(center.tolist())
                shifted.append((center+float(record['center_shift_px'])*normal).tolist())
                previous_t = t
            if segment:
                segments.append(segment)
        unknown = prediction['unknown_intervals_px']
        unknown_segments = [[(origin+float(t)*direction).tolist() for t in interval] for interval in unknown]
        raw = dict(prediction)
        if not self.retain_step_records:
            raw.pop('step_records', None)
        provenance.update(original_prediction=raw, selected_supported_sample_count=len(supported),
            supported_axis_points_are_cross_sections_with_current_image_evidence=True,
            measured_center_points_include_original_profile_center_shift=True,
            gaps_remain_unknown=True, optical_axis_validated_not_independently_estimated=True,
            source_core_sha256_after_execution=_sha(self.source_dir/'straight_core.py'))
        provenance['path_segment_support'] = [bool(np.linalg.norm(np.asarray(b)-a) <= step*1.1)
                                              for a, b in zip(supported[:-1], supported[1:])]
        base.update(s2_status=prediction['status'], supported_segments_xy=segments,
            unknown_intervals_px=unknown, unknown_segments_xy=unknown_segments,
            optical_endpoint_xy=prediction['endpoint_xy'])
        if prediction['status'] != 'PARTIAL' or len(supported) < 8:
            provenance['reason'] = 'no_sufficient_selected_current_profile_support'
            return base
        abc = np.array([normal[0], normal[1], -normal@origin])
        if abc[1] < 0:
            abc = -abc
        base.update(valid=True, line_abc=abc.tolist(), line_points_xy=supported,
                    measured_center_points_xy=shifted, path_xy=supported)
        provenance['reason'] = 'conditioned_axis_with_current_bilateral_evidence'
        return base


def compare_p1_unknown_gaps(p1, s2, width_px, minimum_gap_px=40.):
    """Describe original P1 unknown gaps using ONLY measured S2 centers.

    Coverage means nearby local bilateral profile evidence, not same-material
    proof, not a new whole-path authorization. Sparse collinear extrapolation
    cannot fill a sample: nearest measured support must lie within one step.
    """
    path = np.asarray(p1['path_xy'], float)
    flags = np.asarray(p1['diagnostics']['path_segment_support'], bool)
    measured = np.asarray(s2.get('measured_center_points_xy', []), float).reshape(-1, 2)
    if len(flags) != len(path)-1:
        raise ValueError('P1 segment support does not align with its path')
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    changes = np.diff(np.r_[False, ~flags, False].astype(int))
    results = []
    step = float(s2['diagnostics']['configuration']['step_px'])
    for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        length = float(lengths[start:stop].sum())
        if length < minimum_gap_px:
            continue
        a, b = path[start], path[stop]
        delta = b-a
        chord = float(np.linalg.norm(delta))
        tangent = delta/max(chord, 1e-12)
        normal = np.array([-tangent[1], tangent[0]])
        positions = np.linspace(0., chord, max(2, int(np.ceil(chord/step))+1))
        projections = (measured-a)@tangent if len(measured) else np.array([])
        cross_dist = np.abs((measured-a)@normal) if len(measured) else np.array([])
        nearby = projections[cross_dist <= .5*width_px]
        covered = np.array([bool(len(nearby) and np.min(np.abs(nearby-t)) <= .8*step) for t in positions])
        edges = np.diff(np.r_[False, ~covered, False].astype(int))
        longest = max((positions[min(j, len(positions)-1)]-positions[i]
                       for i, j in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))), default=0.)
        results.append(dict(p1_start_index=int(start), p1_stop_index=int(stop),
            start_xy=a.tolist(), end_xy=b.tolist(), p1_unknown_path_length_px=length,
            p1_unknown_chord_x_span_px=abs(float(delta[0])),
            p1_unknown_chord_length_px=chord, evaluated_samples=len(positions),
            s2_current_profile_covered_samples=int(covered.sum()),
            s2_current_profile_covered_fraction=float(covered.mean()),
            maximum_remaining_unknown_chord_px=float(longest),
            coverage_requires_lateral_distance_px=.5*width_px,
            same_rope_identity_confirmed=False, whole_path_authorized=False))
    return results


def smoke_two_current_frames():
    """Only the two explicitly authorized source frames; no other decoding."""
    import cv2
    from lc1_s1_adapter import S1Adapter
    root = Path(__file__).resolve().parent
    inputs = root/'p1_runtime/smoke_current_frame'
    out = root/'p1_runtime/s2_current_seed_smoke'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite an S2 smoke run')
    out.mkdir(parents=True, exist_ok=True)
    adapter = S2Adapter(retain_step_records=True)
    s1_adapter = S1Adapter(source_dir=root/'supplied_assets/straight_gradient_20261104_v1',
                           retain_step_records=True)
    cases = [('left111500__f0', 'dual_20260924_111500', 'left', 'left_DA9627065.avi'),
             ('right113305__f0', 'dual_20260923_113305', 'right', 'right_DA9626975.avi')]
    summary = []
    for name, session, camera, filename in cases:
        p1 = json.loads((inputs/(name+'.json')).read_text(encoding='utf-8-sig'))
        video = Path(r'E:\study\program\济宁\双相机同步')/session/filename
        cap = cv2.VideoCapture(str(video))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError('Cannot decode authorized frame zero')
        pixel_hash = hashlib.sha256(frame.tobytes()).hexdigest()
        if pixel_hash != p1['smoke_original_pixel_sha256']:
            raise ValueError('Source pixels do not match the current P1 smoke frame')
        seed = p1['seed_xy']
        width = float(p1['diagnostics']['current_A7_seed']['width_px'])
        s2 = adapter.step(frame, seed, seed_source=p1['seed_source'], width_px=width)
        s1 = s1_adapter.step(frame, seed, seed_source=p1['seed_source'])
        gaps = compare_p1_unknown_gaps(p1, s2, width)
        data = dict(source_video=str(video), camera=camera, frame_index=0,
                    original_pixel_sha256=pixel_hash, source_P1_record=str(inputs/(name+'.json')),
                    seed_xy=seed, candidate_width_px=width, s2=s2, s1=s1,
                    P1_unknown_gap_comparison=gaps, decoding_scope='this_frame_zero_only')
        (out/(name+'.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        rendered = frame.copy()
        for segment in s2['supported_segments_xy']:
            if len(segment)>1:
                cv2.polylines(rendered, [np.rint(segment).astype(np.int32)], False, (0, 255, 0), 3, cv2.LINE_AA)
        for start, end in s2['unknown_segments_xy']:
            endpoints = np.linspace(start, end, 40)
            for k in range(0, len(endpoints)-1, 2):
                cv2.line(rendered, tuple(np.rint(endpoints[k]).astype(int)),
                         tuple(np.rint(endpoints[k+1]).astype(int)), (0, 80, 255), 2, cv2.LINE_AA)
        cv2.polylines(rendered, [np.rint(seed).astype(np.int32)], False, (255,255,0), 4, cv2.LINE_AA)
        cv2.putText(rendered, 'S2 current profile evidence: green; UNKNOWN: red dashed; E1 condition: cyan',
                    (30,60), cv2.FONT_HERSHEY_SIMPLEX, .9, (255,255,255), 2, cv2.LINE_AA)
        cv2.imencode('.jpg', rendered, [cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(str(out/(name+'_S2_overlay.jpg')))
        summary.append(dict(case=name, s2_valid=s2['valid'], s2_status=s2['s2_status'],
            s2_supported_samples=len(s2['line_points_xy']),
            s2_axis_extent_px=s2['diagnostics']['original_prediction'].get('axis_extent_px'),
            s2_supported_length_px=s2['diagnostics']['original_prediction'].get('supported_length_px'),
            s2_unknown_intervals_px=s2['unknown_intervals_px'],
            s1_valid=s1['valid'], s1_supported_samples=len(s1['line_points_xy']),
            P1_unknown_gap_comparison=gaps))
    receipt = dict(adapter_sha256=_sha(__file__), source_hashes=adapter.hashes,
        original_S2_sources_modified=False, original_S1_sources_modified=False,
        model_or_thresholds_changed=False, guidance_auto_proposals_executed=False,
        decoded_frames=2, frame_indices=[0,0], results=summary,
        whole_path_authorized=False, scope='conditional current-image comparison, not accuracy')
    (out/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return receipt


if __name__ == '__main__':
    print(json.dumps(smoke_two_current_frames(), ensure_ascii=False, indent=2))
