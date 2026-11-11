"""Faithful adapter for the supplied frozen IR-GRAD-S1.0.0 core.

S1 is conditioned on an explicit far-to-spool E1. This adapter generates no
seed, loads no alternative algorithm, and never interprets an optical endpoint
as physical contact. Input and returned coordinates are original image pixels.
"""
from __future__ import annotations
import hashlib
import json
import types
from pathlib import Path
import cv2
import numpy as np

DEFAULT_S1_DIR=Path(r'E:\study\program\济宁\临时文件\straight_gradient_20261104_v1')
SOURCE='IR-GRAD-S1.0.0 frozen S1, explicit E1-conditioned optical straight evidence'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class S1Adapter:
    """S1Adapter().step(original_bgr, seed_xy, seed_source='current_frame_P1_A7').

    The input seed's first point must be farther from the spool and its second
    point toward the spool. The adapter neither reverses nor invents that order.
    `valid` only licenses the measured optical straight-line fit, not semantic
    rope identity or a complete free-plus-bent rope path.
    """
    def __init__(self,source_dir=DEFAULT_S1_DIR,*,config_overrides=None,
                 verify_frozen_hashes=True,retain_step_records=False):
        self.source_dir=Path(source_dir).resolve()
        core_path=self.source_dir/'gradient_core.py'
        config_path=self.source_dir/'config.json'
        receipt=json.loads((self.source_dir/'freeze_receipt.json').read_text(encoding='utf-8-sig'))
        self.hashes={'core_sha256':_sha(core_path),'configuration_sha256':_sha(config_path)}
        self.hashes['core_matches_freeze_receipt']=self.hashes['core_sha256']==receipt['core_sha256']
        self.hashes['configuration_matches_freeze_receipt']=self.hashes['configuration_sha256']==receipt['configuration_sha256']
        if verify_frozen_hashes and not (self.hashes['core_matches_freeze_receipt'] and self.hashes['configuration_matches_freeze_receipt']):
            raise ValueError('Supplied S1 code/config do not match their freeze receipt; no substitution made')
        self.config=json.loads(config_path.read_text(encoding='utf-8-sig'))
        self.overrides=dict(config_overrides or {})
        self.config.update(self.overrides)
        self.retain_step_records=bool(retain_step_records)
        # Compile the exact original bytes without writing __pycache__ to the
        # read-only supplied asset directory or importing its batch runner.
        module=types.ModuleType('_lc1_supplied_frozen_s1')
        module.__file__=str(core_path)
        exec(compile(core_path.read_bytes(),str(core_path),'exec'),module.__dict__)
        self.core=module

    def step(self,bgr_original,seed_xy,*,seed_source):
        if not isinstance(seed_source,str) or not seed_source.strip():
            raise ValueError('seed_source is required; S1 is not an autonomous E1 detector')
        provenance={'source':SOURCE,'source_dir':str(self.source_dir),**self.hashes,
                    'config_overrides':self.overrides,'seed_source':seed_source,
                    'coordinate_system':'original_unmirrored_image_xy',
                    'seed_is_input_not_detection':True,
                    'seed_order':'first far/free, second toward spool',
                    'physical_contact':None,'same_material_identity':None,
                    'method_scope':'single optical straight axis only, no curved path',
                    'ambiguity':'S1 has no semantic/candidate-identity disambiguator; PARTIAL is not same-rope confirmation.'}
        base={'valid':False,'detected_partial':False,'s1_status':'UNKNOWN','source':SOURCE,
              'line_abc':None,'line_points_xy':[],'supported_segments_xy':[],
              'path_xy':[],'whole_path_candidate_supported':False,
              'optical_endpoint_xy':None,'diagnostics':provenance}
        if seed_xy is None:
            provenance.update(stop_reason='missing_explicit_E1',s1_core_executed=False)
            return base
        # This exact supplied function performs all bilateral evidence decisions.
        fields,prediction=self.core.analyze(bgr_original,seed_xy,self.config)
        del fields
        status=prediction['status']
        segments=prediction.get('supported_segments_xy',[])
        measured=np.asarray([p for segment in segments for p in segment],np.float64).reshape(-1,2)
        # prediction.points_xy[0] is input E1[1], so deliberately never use that
        # list to fit a line or count measured points.
        provenance.update(s1_core_executed=True,stop_reason=prediction.get('stop_reason'),
                          supported_extension_px=prediction.get('supported_extension_px',0.),
                          supported_sample_count=int(len(measured)),
                          optical_endpoint_is_not_physical_contact=True)
        raw=dict(prediction)
        records=raw.pop('step_records',[]) if not self.retain_step_records else raw.get('step_records',[])
        provenance['s1_step_record_count']=len(records)
        provenance['s1_supported_step_count']=sum(bool(r.get('supported')) for r in records)
        provenance['s1_prediction']=raw
        base.update(s1_status=status,detected_partial=status=='PARTIAL')
        if status!='PARTIAL' or len(measured)<8 or not np.isfinite(measured).all():
            provenance['adapter_gate']='UNKNOWN_or_insufficient_current_bilateral_samples'
            return base
        vx,vy,x0,y0=map(float,cv2.fitLine(measured.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).reshape(-1))
        abc=np.array([-vy,vx,vy*x0-vx*y0],np.float64)
        norm=np.linalg.norm(abc[:2])
        if not np.isfinite(abc).all() or norm<1e-9:
            provenance['adapter_gate']='degenerate_supported_point_fit'
            return base
        abc/=norm
        if abc[1]<0:abc=-abc
        residual=np.abs(measured@abc[:2]+abc[2])
        provenance.update(adapter_gate='current_S1_bilateral_supported_straight_fit',
                          line_fit_uses_E1_points=False,
                          line_fit_residual_median_px=float(np.median(residual)),
                          line_fit_residual_p95_px=float(np.percentile(residual,95)),
                          line_fit_warning='A fit to optical support can follow equipment or neighbouring winding if supplied E1 identifies that structure.')
        base.update(valid=True,line_abc=abc.tolist(),line_points_xy=measured.tolist(),
                    supported_segments_xy=segments,path_xy=measured.tolist(),
                    optical_endpoint_xy=prediction.get('endpoint_xy'))
        return base


# Explicit alias for callers that prefer a descriptive name; same original core.
S1Frontend=S1Adapter
