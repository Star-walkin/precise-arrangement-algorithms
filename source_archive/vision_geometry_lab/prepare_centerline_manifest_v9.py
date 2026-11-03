"""Prepare a reference-only, whole-session-separated V9 geometry inventory.

No label/image/quality file is copied or changed. C03 and C06 remain inventory
diagnostics only. Full-image ROIs are a fixed global input scope, not a crop
selected from target lines. Re-running validates source hashes and requires
byte-identical derived outputs; it never overwrites existing output files.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json

from .centerline_review_v5 import load_manifest, validate_annotation, read_clean_image
from .centerline_quality_v5 import load_label_quality, quality_path


ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OLD=ROOT/'vision_geometry_lab/outputs/centerline_v5/pilot/manifest.json'
DEFAULT_SUPPLEMENT=ROOT/'vision_geometry_lab/outputs/centerline_v7/review_batch01/manifest.json'
DEFAULT_OUTPUT=ROOT/'vision_geometry_lab/outputs/centerline_v9/dataset'
RESERVED_SESSION='dual_20260923_113029'
SUPPLEMENT_POLICY={
    'C01':('dual_20260923_101517','train','geometry_train'),
    'C02':('dual_20260923_101517','train','geometry_train'),
    'C03':('dual_20260923_101517','train','diagnostic_only'),
    'C04':(RESERVED_SESSION,'holdout','geometry_holdout'),
    'C05':(RESERVED_SESSION,'holdout','geometry_holdout'),
    'C06':(RESERVED_SESSION,'holdout','diagnostic_only'),
    'C07':('R_run_stop_run_20260922_01','train','geometry_train'),
    'C08':('R_run_stop_run_20260922_01','train','geometry_train'),
}
TARGET_NAMES={'centerline','edge_flange','edge_surface'}


def file_hash(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): result.update(block)
    return result.hexdigest()


def _json_bytes(value):
    return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')


def _remember(snapshot,path,expected=None,optional=False):
    path=Path(path).resolve()
    actual=file_hash(path) if path.is_file() else None
    if actual is None and not optional: raise ValueError(f'Missing source file: {path}')
    if expected is not None and actual!=expected: raise ValueError(f'Source SHA256 mismatch: {path}')
    if str(path) in snapshot and snapshot[str(path)]!=actual: raise ValueError('Source changed during preparation')
    snapshot[str(path)]=actual
    return actual


def _check_snapshot(snapshot):
    for name,expected in snapshot.items():
        path=Path(name); current=file_hash(path) if path.is_file() else None
        if current!=expected: raise ValueError(f'Source changed during preparation: {name}')


def _validate_source_manifests(old,supplement):
    if len(old['samples'])!=12 or len(supplement['samples'])!=8:
        raise ValueError('Expected exactly old 12 plus supplemental 8 samples')
    if [sum(s['split']==split for s in old['samples']) for split in ('train','holdout')]!=[6,6]:
        raise ValueError('Old manifest must preserve six train and six holdout samples')
    all_samples=old['samples']+supplement['samples']
    ids=set(); groups={}; image_paths=set(); annotation_paths=set()
    for sample in all_samples:
        # Check metadata BEFORE opening any referenced image or video. No video
        # is opened anywhere, and all 9/25 path/session references are rejected.
        if '20260925' in json.dumps(sample,ensure_ascii=False):
            raise ValueError('Future 9/25 blind-test sources are forbidden')
        if sample['sample_id'] in ids: raise ValueError('Duplicate source sample_id')
        ids.add(sample['sample_id'])
        for key,seen in (('image_path',image_paths),('annotation_path',annotation_paths)):
            value=str(Path(sample[key]).resolve())
            if value in seen: raise ValueError(f'Duplicate source {key}')
            seen.add(value)
        groups.setdefault(sample['group_id'],set()).add(sample['split'])
        if sample['group_id']==RESERVED_SESSION and sample['split']!='holdout':
            raise ValueError('Reserved 113029 session must remain holdout')
    if any(len(splits)!=1 for splits in groups.values()):
        raise ValueError('Same-session train/holdout leakage')
    old_groups={s['group_id'] for s in old['samples']}
    if old_groups & {s['group_id'] for s in supplement['samples']}:
        raise ValueError('Supplement source session overlaps the old pilot')
    found=set()
    for sample in supplement['samples']:
        cid=sample['sample_id'].split('__',1)[0]
        if cid not in SUPPLEMENT_POLICY or cid in found: raise ValueError('Supplement must contain unique C01 through C08')
        found.add(cid); group,split,_=SUPPLEMENT_POLICY[cid]
        if (sample['group_id'],sample['split'])!=(group,split):
            raise ValueError(f'{cid} source group/split differs from the fixed policy')
    if found!=set(SUPPLEMENT_POLICY): raise ValueError('Missing supplemental candidate')
    return {group:next(iter(splits)) for group,splits in sorted(groups.items())}


def build_dataset(old_manifest=DEFAULT_OLD,supplement_manifest=DEFAULT_SUPPLEMENT,output=DEFAULT_OUTPUT):
    """Validate sources and construct JSON values without writing anything."""
    old_manifest=Path(old_manifest).resolve(); supplement_manifest=Path(supplement_manifest).resolve(); output=Path(output).resolve()
    sources={}; _remember(sources,old_manifest); _remember(sources,supplement_manifest)
    old=load_manifest(old_manifest); supplement=load_manifest(supplement_manifest)
    groups=_validate_source_manifests(old,supplement)
    inventory_rows=[]; geometry=[]
    for source_name,path,manifest in (('old_pilot_12',old_manifest,old),('supplement_8',supplement_manifest,supplement)):
        for sample in manifest['samples']:
            cid=sample['sample_id'].split('__',1)[0] if source_name=='supplement_8' else None
            image_sha=_remember(sources,sample['image_path'],sample['image_sha256'])
            annotation_sha=_remember(sources,sample['annotation_path'])
            annotation=validate_annotation(json.loads(Path(sample['annotation_path']).read_text('utf-8-sig')),sample)
            read_clean_image(sample)  # Exact PNG bytes, expected shape, unchanged pixels.
            qpath=quality_path(sample); quality_sha=_remember(sources,qpath,optional=True)
            quality=load_label_quality(sample)
            if quality['quality_sha256']!=quality_sha: raise ValueError('Quality changed during validation')
            use=SUPPLEMENT_POLICY[cid][2] if cid else 'geometry_'+sample['split']
            if cid=='C03':
                if annotation['status']!='absent' or any(annotation['polylines'].values()):
                    raise ValueError('C03 must remain confirmed absent without a fabricated seed')
                reason='confirmed_absent_no_seed; diagnostic only, no seed-path training or score'
            elif cid=='C06':
                if annotation['status']!='partial' or not annotation['notes'].strip():
                    raise ValueError('C06 must remain documented partial/free-span-only')
                reason='free_span_only_no_wound_path_gt; diagnostic only, not wound training or score'
            else:
                if annotation['status'] not in ('clear','partial'):
                    raise ValueError('Geometry samples require reviewed clear/partial targets and real free_seed')
                reason='reviewed partial target geometry, with all current quality exclusions'
            trusted=quality.get('supervision_polylines')
            if cid in ('C07','C08'):
                if (not isinstance(trusted,dict) or set(trusted)!=TARGET_NAMES or len(trusted['centerline'])!=7
                        or trusted['centerline']!=annotation['polylines']['centerline'][:7]):
                    raise ValueError(f'{cid} requires validated C7 cutoff and exactly three trusted target lines')
                raw_quality=json.loads(qpath.read_text('utf-8-sig'))
                if raw_quality.get('trusted_supervision_rule',{}).get('last_trusted_center_index_1based')!=7:
                    raise ValueError(f'{cid} requires the confirmed C7 rule')
            h,w=sample['image_shape_hw']; full_roi=[0,0,w,h]
            derived=copy.deepcopy(sample)
            derived.update(roi_xyxy=full_roi,roi_source='fixed full-image global input scope; independent of GT geometry',
                           source_roi_xyxy=sample['roi_xyxy'][:],geometry_role=use,
                           source_manifest_path=str(path),source_manifest_sha256=sources[str(path)],
                           annotation_sha256=annotation_sha,quality_sha256=quality_sha,
                           quality_path=str(qpath.resolve()))
            validate_annotation(annotation,derived)  # Original full-image coordinates stay valid.
            if use!='diagnostic_only': geometry.append(derived)
            inventory_rows.append(dict(sample_id=sample['sample_id'],candidate_id=cid,source_set=source_name,
                source_manifest_path=str(path),source_manifest_sha256=sources[str(path)],
                group_id=sample['group_id'],split=sample['split'],intended_use=use,use_reason=reason,
                included_in_geometry_manifest=use!='diagnostic_only',
                seed_path_training=use=='geometry_train',wound_geometry_scoring=use!='diagnostic_only',
                image_path=sample['image_path'],image_sha256=image_sha,image_shape_hw=sample['image_shape_hw'],
                source_roi_xyxy=sample['roi_xyxy'],global_input_roi_xyxy=full_roi,
                annotation_path=sample['annotation_path'],annotation_sha256=annotation_sha,
                annotation_status=annotation['status'],annotation_reviewed=annotation['reviewed'],annotation_notes=annotation['notes'],
                original_line_point_counts={key:len(value) for key,value in annotation['polylines'].items()},
                quality_snapshot=dict(path=str(qpath.resolve()),sha256=quality_sha,
                    region_count=len(quality['regions']),region_kinds=[r['kind'] for r in quality['regions']],
                    trusted_target_point_counts={key:len(value) for key,value in trusted.items()} if trusted else None),
                original_source_metadata=copy.deepcopy(sample['source']),
                image_orientation=sample.get('provenance',{}).get('image_orientation')))
    counts={split:sum(s['split']==split for s in geometry) for split in ('train','holdout')}
    if counts!={'train':10,'holdout':8} or len(geometry)!=18:
        raise ValueError('Fixed geometry composition must be ten train and eight holdout')
    if any(s['sample_id'].startswith(('C03__','C06__')) for s in geometry):
        raise ValueError('C03/C06 cannot enter the geometry manifest')
    _check_snapshot(sources)
    inventory=dict(schema_version=1,purpose='V9 reference inventory; not a training execution',samples=inventory_rows,
        counts=dict(total=20,geometry_train=10,geometry_holdout=8,diagnostic_only=2),
        source_file_sha256=sources,source_manifest_paths=[str(old_manifest),str(supplement_manifest)],
        source_files_modified=False,images_labels_quality_copied=False,group_split=groups,
        reserved_session=RESERVED_SESSION,reserved_scope='entire session and both cameras remain holdout; development validation, not blind test',
        geometry_manifest_path=str(output/'manifest.json'),source_notes_preserved=True,
        blind_20260925_read=False,video_decoding=False,hardware_access=False,training_started=False)
    inventory_bytes=_json_bytes(inventory)
    result=dict(schema_version=1,task='seed_conditioned_new_winding_centerline',samples=geometry,
        purpose='V9 eighteen-sample geometry-only reference manifest; fixed full-image global scope',
        inventory_path=str(output/'inventory.json'),inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
        source_manifest_sha256={str(p):sources[str(p)] for p in (old_manifest,supplement_manifest)},
        geometry_train_count=10,geometry_holdout_count=8,diagnostic_only_sample_ids=[r['sample_id'] for r in inventory_rows if r['intended_use']=='diagnostic_only'],
        roi_policy='all source images use [0,0,W,H]; no target-selected ROI or target-defined seed',
        seed_policy='existing human free_seed unchanged; no fake or synthesized seed',
        group_split=groups,reserved_holdout_session=RESERVED_SESSION,
        holdout_semantics='development validation by whole session; not pristine blind test',
        training_authorized=False,training_started=False,physical_control=False,
        preparation_source_sha256=file_hash(__file__))
    return result,inventory


def prepare(old_manifest=DEFAULT_OLD,supplement_manifest=DEFAULT_SUPPLEMENT,output=DEFAULT_OUTPUT):
    output=Path(output).resolve()
    manifest,inventory=build_dataset(old_manifest,supplement_manifest,output)
    expected={output/'manifest.json':_json_bytes(manifest),output/'inventory.json':_json_bytes(inventory)}
    if output.exists():
        for path,data in expected.items():
            if not path.is_file() or path.read_bytes()!=data:
                raise FileExistsError('Existing derived output or source snapshot differs; preserving every file, no overwrite')
        return dict(status='already_prepared_verified_no_writes',geometry_samples=18,inventory_samples=20,
                    manifest=str(output/'manifest.json'),source_files_modified=False)
    output.mkdir(parents=True,exist_ok=False)
    for path,data in expected.items():
        with path.open('xb') as stream: stream.write(data)
    load_manifest(output/'manifest.json')
    _check_snapshot(inventory['source_file_sha256'])
    return dict(status='prepared_reference_only',geometry_samples=18,geometry_train=10,geometry_holdout=8,
                diagnostic_only=2,inventory_samples=20,manifest=str(output/'manifest.json'),
                inventory=str(output/'inventory.json'),source_files_modified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-manifest',type=Path,default=DEFAULT_OLD)
    parser.add_argument('--supplement-manifest',type=Path,default=DEFAULT_SUPPLEMENT)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    print(json.dumps(prepare(args.old_manifest,args.supplement_manifest,args.output),ensure_ascii=False))
