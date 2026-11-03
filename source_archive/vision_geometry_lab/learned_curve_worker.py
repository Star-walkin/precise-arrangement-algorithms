"""Isolated SAM2 baseline worker; immutable existing weights, no hardware access.

All multimask proposals are preserved. Selection is fixed to model score before
any reference-point diagnostics; reference points are not sent to this worker
except when explicitly present in a case's input prompts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha256(value):
    payload=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('protocol')
    args=parser.parse_args()
    protocol_path=Path(args.protocol)
    protocol=json.loads(protocol_path.read_text('utf-8'))
    if not isinstance(protocol,dict) or not isinstance(protocol.get('cases'),list):
        raise ValueError('Protocol must be an object with a cases list')
    names=[case.get('name') for case in protocol['cases']]
    if len(names)!=len(set(names)):
        raise ValueError('Protocol contains duplicate case names')
    protocol_sha256=_file_sha256(protocol_path)
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'vision_training/vendor/sam2'))
    import numpy as np
    from PIL import Image
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(2)
    checkpoint=root/'vision_training/checkpoints/sam2.1_hiera_tiny.pt'
    checkpoint_sha256=_file_sha256(checkpoint)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    config='configs/sam2.1/sam2.1_hiera_t.yaml'
    started=time.perf_counter()
    model=build_sam2(config,str(checkpoint),device=device,apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model)
    load_ms=(time.perf_counter()-started)*1000
    records=[]
    cache_key=None
    image_hashes={}
    with torch.inference_mode():
        for case in protocol['cases']:
            folder=Path(protocol['output'])/case['name']
            folder.mkdir(parents=True,exist_ok=True)
            key=(case['image'],tuple(case['roi']))
            x0,y0,x1,y1=case['roi']
            if not (0<=x0<x1 and 0<=y0<y1):
                raise ValueError('Invalid ROI: '+case['name'])
            encoder_ms=0.
            cached=key==cache_key
            if not cached:
                with Image.open(case['image']) as source:
                    rgb=np.asarray(source.convert('RGB'))
                if x1>rgb.shape[1] or y1>rgb.shape[0]:
                    raise ValueError('ROI outside image: '+case['name'])
                start=time.perf_counter()
                predictor.set_image(rgb[y0:y1,x0:x1].copy())
                if device=='cuda': torch.cuda.synchronize()
                encoder_ms=(time.perf_counter()-start)*1000
                cache_key=key
            points=np.asarray(case['points'],np.float32)
            labels=np.asarray(case['labels'],np.int32)
            if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
                raise ValueError('Invalid prompt coordinates')
            if labels.shape!=(len(points),) or not np.any(labels==1) or not np.isin(labels,[0,1]).all():
                raise ValueError('Prompt labels must match points and include a positive')
            if np.any((points[:,0]<x0)|(points[:,0]>=x1)|(points[:,1]<y0)|(points[:,1]>=y1)):
                raise ValueError('Prompt outside pre-registered ROI: '+case['name'])
            start=time.perf_counter()
            masks,scores,_=predictor.predict(point_coords=points-[x0,y0],
                point_labels=labels,multimask_output=True)
            if device=='cuda': torch.cuda.synchronize()
            decoder_ms=(time.perf_counter()-start)*1000
            selected=int(np.argmax(scores))
            mask_hashes=[]
            for index,mask in enumerate(masks):
                path=folder/f'mask_{index}.png'
                Image.fromarray(mask.astype(np.uint8)*255).save(path)
                mask_hashes.append(_file_sha256(path))
            image_hashes.setdefault(case['image'],_file_sha256(case['image']))
            record=dict(case,selected_index=selected,scores=[float(s) for s in scores],
                         selection='argmax_model_score; no reference metric used',
                        protocol_sha256=protocol_sha256,case_sha256=_json_sha256(case),
                        image_sha256=image_hashes[case['image']],mask_sha256=mask_hashes,
                        checkpoint_sha256=checkpoint_sha256,
                        encoder_ms=encoder_ms,decoder_ms=decoder_ms,encoder_reused=cached,
                        device=device,checkpoint=str(checkpoint),config=config,
                        identity_confirmed=False,physical_contact_confirmed=False,
                        last_turn_confirmed=False,new_layer_confirmed=False,force_command=None)
            (folder/'inference.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),'utf-8')
            records.append(record)
            print(case['name'],f'encoder={encoder_ms:.0f}ms decoder={decoder_ms:.0f}ms',flush=True)
    metadata=dict(model='sam2.1_hiera_tiny',torch_version=torch.__version__,device=device,
                  load_ms=load_ms,protocol_sha256=protocol_sha256,
                  checkpoint_sha256=checkpoint_sha256,
                  weights_modified=False,training_performed=False,hardware_access=False,cases=records)
    metadata_keys=('branch_type','protocol_status','primary_pre_registered',
                   'exploratory_after_primary','post_hoc','blind','parent_protocol_sha256',
                   'negative_semantics','training','hardware_access')
    metadata.update({key:protocol[key] for key in metadata_keys if key in protocol})
    (Path(protocol['output'])/'worker_results.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),'utf-8')


if __name__=='__main__': main()
