import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from vision_geometry_lab.centerline_quality_cutoff_v8 import build_quality_cutoff_payload
from vision_geometry_lab.prepare_centerline_manifest_v9 import prepare, build_dataset, SUPPLEMENT_POLICY


def _write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def _fixture(root):
    root=Path(root); (root/'sources').mkdir(); old=[]; supplement=[]
    for index in range(20):
        cid=f'C{index-11:02d}' if index>=12 else None
        group,split,_=SUPPLEMENT_POLICY[cid] if cid else (f'old_{"train" if index<6 else "holdout"}_{index//3}', 'train' if index<6 else 'holdout','geometry')
        sid=cid+'__fixture' if cid else f'old_{index:02d}'
        image=root/'sources'/f'{sid}.png'; Image.new('RGB',(101,101),(index,20,30)).save(image)
        annotation_path=root/'sources'/f'{sid}.json'
        sample=dict(sample_id=sid,group_id=group,split=split,image_path=str(image),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            image_shape_hw=[101,101],roi_xyxy=[0,0,95,90],annotation_path=str(annotation_path),
            source=dict(video_path='metadata_only_not_opened.avi',frame_index=index,time_s=float(index),camera='fixture'))
        lines=dict(centerline=[[10+10*i,50] for i in range(8)],edge_flange=[[5,45],[30,45],[60,45],[90,45]],
                   edge_surface=[[5,55],[40,55],[70,55],[90,55]],free_seed=[[10,50],[20,50]])
        label=dict(schema_version=1,sample_id=sid,image_sha256=sample['image_sha256'],coordinate_system='full_image_xy',reviewed=True,
            status='absent' if cid=='C03' else 'partial',polylines={key:[] for key in lines} if cid=='C03' else lines,
            ignore_polygons=[],notes='absent' if cid=='C03' else 'free_span_only, no wound GT' if cid=='C06' else f'original note {index}')
        _write(annotation_path,label)
        if cid in ('C07','C08'):
            payload=build_quality_cutoff_payload(sample,annotation_path.read_bytes(),7)['quality_payload']
            _write(annotation_path.with_suffix('.quality.json'),payload)
        (old if index<12 else supplement).append(sample)
    old_path=root/'old.json'; supplement_path=root/'supplement.json'
    for path,samples in ((old_path,old),(supplement_path,supplement)):
        _write(path,dict(schema_version=1,task='seed_conditioned_new_winding_centerline',samples=samples))
    return old_path,supplement_path


class PrepareManifestV9Tests(unittest.TestCase):
    def test_reference_only_counts_full_roi_notes_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); old,supplement=_fixture(root)
            before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result=prepare(old,supplement,root/'dataset')
            self.assertEqual(result['geometry_train'],10); self.assertEqual(result['geometry_holdout'],8)
            m=json.loads((root/'dataset/manifest.json').read_text('utf-8'))
            inv=json.loads((root/'dataset/inventory.json').read_text('utf-8'))
            self.assertEqual(len(m['samples']),18); self.assertEqual(len(inv['samples']),20)
            self.assertTrue(all(s['roi_xyxy']==[0,0,101,101] for s in m['samples']))
            self.assertFalse(any(s['sample_id'].startswith(('C03__','C06__')) for s in m['samples']))
            for row in inv['samples']:
                original=json.loads(Path(row['annotation_path']).read_text('utf-8'))
                self.assertEqual(row['annotation_notes'],original['notes'])
                self.assertEqual(row['annotation_sha256'],hashlib.sha256(Path(row['annotation_path']).read_bytes()).hexdigest())
                if row['candidate_id'] in ('C03','C06'):
                    self.assertEqual(row['intended_use'],'diagnostic_only')
                    self.assertFalse(row['seed_path_training']); self.assertFalse(row['wound_geometry_scoring'])
                if row['candidate_id'] in ('C07','C08'):
                    self.assertEqual(row['quality_snapshot']['trusted_target_point_counts']['centerline'],7)
            self.assertTrue(all(p.read_bytes()==data for p,data in before.items()))
            self.assertEqual(set(p.name for p in (root/'dataset').iterdir()),{'manifest.json','inventory.json'})

    def test_idempotence_does_not_overwrite_or_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); old,supplement=_fixture(root); output=root/'dataset'
            prepare(old,supplement,output); before={p:p.read_bytes() for p in output.iterdir()}
            self.assertEqual(prepare(old,supplement,output)['status'],'already_prepared_verified_no_writes')
            self.assertTrue(all(p.read_bytes()==data for p,data in before.items()))
            sentinel=output/'manifest.json'; sentinel.write_bytes(b'user-owned changed output')
            with self.assertRaisesRegex(FileExistsError,'no overwrite'): prepare(old,supplement,output)
            self.assertEqual(sentinel.read_bytes(),b'user-owned changed output')

    def test_same_session_leakage_is_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); old,supplement=_fixture(root); data=json.loads(old.read_text('utf-8'))
            data['samples'][6]['group_id']=data['samples'][0]['group_id']; _write(old,data)
            with self.assertRaisesRegex(ValueError,'Same-session'): prepare(old,supplement,root/'dataset')
            self.assertFalse((root/'dataset').exists())

    def test_wrong_label_image_binding_and_quality_hash_are_rejected(self):
        for corruption in ('label_image_hash','quality_annotation_hash','png_bytes'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); old,supplement=_fixture(root); data=json.loads(supplement.read_text('utf-8'))
                row=data['samples'][6]; label=Path(row['annotation_path'])
                if corruption=='label_image_hash':
                    value=json.loads(label.read_text('utf-8')); value['image_sha256']='b'*64; _write(label,value)
                elif corruption=='quality_annotation_hash': label.write_bytes(label.read_bytes()+b'\n')
                else: Path(row['image_path']).write_bytes(b'changed image')
                with self.assertRaises(ValueError): prepare(old,supplement,root/'dataset')
                self.assertFalse((root/'dataset').exists())

    def test_c03_c06_cannot_be_promoted_by_geometry_like_polylines(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); old,supplement=_fixture(root)
            manifest,inventory=build_dataset(old,supplement,root/'dataset')
            self.assertFalse(any(s['sample_id'].startswith('C06__') for s in manifest['samples']))
            c06=next(s for s in inventory['samples'] if s['candidate_id']=='C06')
            self.assertGreater(c06['original_line_point_counts']['centerline'],2)
            c03=json.loads(supplement.read_text('utf-8'))['samples'][2]; path=Path(c03['annotation_path'])
            value=json.loads(path.read_text('utf-8')); value['status']='partial'
            value['polylines']=json.loads(Path(json.loads(supplement.read_text('utf-8'))['samples'][0]['annotation_path']).read_text('utf-8'))['polylines']; _write(path,value)
            with self.assertRaisesRegex(ValueError,'C03'): prepare(old,supplement,root/'dataset')

    def test_c07_c08_require_confirmed_c7_quality(self):
        for cid in ('C07','C08'):
            with self.subTest(cid=cid), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); old,supplement=_fixture(root)
                row=next(s for s in json.loads(supplement.read_text('utf-8'))['samples'] if s['sample_id'].startswith(cid+'__'))
                path=Path(row['annotation_path']).with_suffix('.quality.json')
                value=json.loads(path.read_text('utf-8')); del value['trusted_supervision_polylines']; del value['trusted_supervision_rule']; _write(path,value)
                with self.assertRaisesRegex(ValueError,'requires validated C7'): prepare(old,supplement,root/'dataset')

    def test_reserved_holdout_and_blind_source_rejected(self):
        for corruption in ('reserved_train','blind'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); old,supplement=_fixture(root); data=json.loads(supplement.read_text('utf-8'))
                if corruption=='reserved_train': data['samples'][3]['split']='train'
                else: data['samples'][0]['source']['video_path']='forbidden_20260925.avi'
                _write(supplement,data)
                with self.assertRaises(ValueError): prepare(old,supplement,root/'dataset')
                self.assertFalse((root/'dataset').exists())
