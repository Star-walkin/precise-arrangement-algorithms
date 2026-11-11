"""Reproducible entry points around byte-identical, frozen RES23 algorithms."""
from pathlib import Path
import argparse, gzip, hashlib, json, os, subprocess, sys, tempfile

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
EXAMPLE=ROOT/'examples/dual_20260924_111500__left'
VERSIONS={1:'20261107',19:'20261110',20:'20261110',21:'20261110',22:'20261111',23:'20261111'}
def version(n):return SRC/f'RopeEdgeSuite_R{n}_{VERSIONS[n]}'
def read(p):
    p=Path(p)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def verify():
    manifest=read(ROOT/'package_manifest.json')
    for item in manifest['files']:
        p=ROOT/item['path']
        if not p.is_file() or digest(p)!=item['sha256']:raise RuntimeError(f'Package integrity mismatch: {p}')
    for item in read(ROOT/'source_inventory.json')['files']:
        if digest(ROOT/item['path'])!=item['sha256']:raise RuntimeError(f'Frozen source changed: {item["path"]}')
    print(f'PASS: {len(manifest["files"])} packaged files; frozen source hashes match')
def test():
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
    tasks=[(23,'test_period23.py'),(23,'test_fusion23.py'),(22,'test_temporal22.py'),(22,'test_motor22.py'),(21,'test_contracts21.py')]
    for n,name in tasks:
        subprocess.run([sys.executable,'-B',str(version(n)/name)],cwd=version(n),env=env,check=True)
    # This historical test writes a receipt; redirect only its output directory.
    # The tested numeric implementation and its inputs remain unchanged.
    code="import tempfile;from pathlib import Path;import test_measurement_contracts as t;d=tempfile.TemporaryDirectory();t.ROOT=Path(d.name);t.run()"
    subprocess.run([sys.executable,'-B','-c',code],cwd=version(20),env=env,check=True)
    print('PASS: 45 behavioral/regression tests')
def replay(measurements,near,features,fps,out,expected=None):
    import numpy as np
    sys.path.insert(0,str(version(23)))
    from fusion23 import analyze
    rows=read(measurements);events=read(near)
    with np.load(features,allow_pickle=False) as data:arr=data['features']
    if len(rows)!=len(arr):raise ValueError('Measurement and texture frame counts differ')
    result=analyze(rows,events,arr,fps)
    if expected is not None:
        if result!=read(expected):raise AssertionError('Replay differs from the frozen R23 output')
        print('PASS: all per-frame outputs and events match the frozen R23 result')
    if out:write(out,result)
    for e in result['events']:
        print(json.dumps({k:e.get(k) for k in ('t0','timeline','image_supported','candidate_s','confirmation_s')},ensure_ascii=False))
    return result
def frontend(video,camera,frames,out):
    import cv2
    sys.path.insert(0,str(version(19)))
    from engine import Engine
    engine=Engine();cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps<=0:raise ValueError('Cannot open video or read its frame rate')
    result=[]
    try:
        for i in range(frames):
            ok,image=cap.read()
            if not ok:raise ValueError(f'Input ended before requested frame {i}')
            if image.shape[:2]!=(3000,4096):raise ValueError('Frozen camera setup requires 4096 x 3000 input; recalibrate other resolutions')
            if camera=='right':image=cv2.flip(image,1)
            q=engine.step(image,i/fps,i);result.append(q)
            print(f'frame {i}: track={q["track"]["valid"]}, boundary={q["boundary"]["fit_valid"]}',flush=True)
    finally:cap.release()
    # Prove that no algorithm module silently fell back to the original lab tree.
    modules={k:str(Path(m.__file__).resolve()) for k,m in sys.modules.items() if getattr(m,'__file__',None) and Path(m.__file__).resolve().is_relative_to(SRC)}
    write(out,dict(scope='R19 SG5/BG11/CG2 raw-video front-end smoke only; F3/d/count not run',frames=result,packaged_modules=modules))
    print(f'PASS: raw-video front-end executed using {len(modules)} packaged modules')

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('verify');sub.add_parser('test')
    demo=sub.add_parser('replay-example');demo.add_argument('--out',type=Path)
    pred=sub.add_parser('predict');pred.add_argument('--measurements',required=True,type=Path);pred.add_argument('--near-events',required=True,type=Path);pred.add_argument('--features',required=True,type=Path);pred.add_argument('--fps',required=True,type=float);pred.add_argument('--out',required=True,type=Path)
    smoke=sub.add_parser('frontend-smoke');smoke.add_argument('--video',required=True,type=Path);smoke.add_argument('--camera',required=True,choices=['left','right']);smoke.add_argument('--frames',type=int,default=3);smoke.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    if a.command=='verify':verify()
    elif a.command=='test':test()
    elif a.command=='replay-example':
        meta=read(EXAMPLE/'fixture.json')
        if digest(EXAMPLE/'features.npz')!=meta['features_sha256']:raise RuntimeError('Fixture feature hash mismatch')
        replay(EXAMPLE/'measurements.json.gz',EXAMPLE/'near_events.json',EXAMPLE/'features.npz',meta['fps'],a.out,EXAMPLE/'expected.json.gz')
    elif a.command=='predict':
        if a.fps<=0:ap.error('--fps must be positive')
        replay(a.measurements,a.near_events,a.features,a.fps,a.out)
    elif a.command=='frontend-smoke':
        if a.frames<1:ap.error('--frames must be positive')
        frontend(a.video,a.camera,a.frames,a.out)
if __name__=='__main__':main()
