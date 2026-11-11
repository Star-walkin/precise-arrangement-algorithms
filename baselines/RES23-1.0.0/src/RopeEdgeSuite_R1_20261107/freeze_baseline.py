"""Create a new immutable-by-hash baseline; never alter original experiments."""
from pathlib import Path
import hashlib,json,shutil,sys,platform,importlib.metadata
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
COMPONENTS=['LC1_locked_column_20261105_v1','LC1_fixed_d2238p7_full_20261106_v1',
 'LC1_boundary_BG1_20261106_v1','LC1_boundary_BG3_point_global_20261107_v2',
 'LC1_boundary_BG4_local_temporal_20261107_v1','LC1_ECW_20261107_v1']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
    out=ROOT/'baseline';out.mkdir(exist_ok=True);src=out/'source';src.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Baseline already sealed; create a new version to change it')
    files=[]
    for name in COMPONENTS:files+=list((PARENT/name).glob('*.py'))
    lc=PARENT/COMPONENTS[0];bg1=PARENT/COMPONENTS[2]
    for folder in [lc/'supplied_assets/straight_gradient_20261104_v1',lc/'supplied_assets/straight_gradient_20261105_v2',
                   lc/'p1_runtime/source_archive',lc/'vendor_archive',bg1/'vendor']:
        files += [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ['.py','.json','.md','.txt'] and '__pycache__' not in p.parts]
    files += [lc/'supplied_assets/step_0750.pt']
    rows=[]
    for p in sorted(set(files)):
        dest=src/p.relative_to(PARENT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        assert sha(dest)==sha(p);rows.append(dict(path=str(dest.relative_to(out)),source=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    cal=out/'calibration';cal.mkdir(exist_ok=True)
    stem='dual_20260924_111500__pair0015__left'
    assets=[PARENT/'camera_condition_sam2_20261009_v1/all_001/body_branches_002/masks/left_camera'/f'{stem}.png',
            PARENT/'camera_condition_bright_rope_boundaries_20261023_v2/all_002/boundaries/records/left_camera'/f'{stem}.json']
    for p,n in zip(assets,['left_search_mask.png','left_pitch_source.json']):
        dest=cal/n;shutil.copy2(p,dest);rows.append(dict(path=str(dest.relative_to(out)),source=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    definition=dict(name_cn='绳缘识别系统',name_en='Rope Edge Suite',short_name='RES-1.0',version='1.0.0',date='2026-11-07',
        stages=['P1/B750+A7/H6 current entry/reacquisition and diagnostic curved path',
                'S1 original core + fixed-spatial-anchor causal tracker; S2 available diagnostic branch',
                'BG1 current yellow flange/visible coil gate',
                'supplied 01/F3 raw observations -> BG3 point classes/continuous curves -> BG4 local recovery/count',
                'BG4 robust causal count/short prediction + fixed-background camera continuity',
                'ECW-3 current local free rope/visible interface + causal proximity windows'],
        defaults=dict(camera='left',width=4096,height=3000,mirror=False,d_original_px=2238.7,pitch_prior_px=23.29412648284868,
                      F3_processing_scale=.5,display_count_within=5,ECW_enter_width=.60,ECW_enter_s=.30,ECW_hold_s=.40,ECW_exit_s=.20),
        known_limits=['d is a provisional user-supplied calibration, not a universally transferred bend estimate',
                      'visible flange/coil interface does not establish physical metal contact',
                      'some S1/local axes require long extrapolation',
                      'P1/H6 full curved path often fails its evidence gate',
                      'neural P1 remains in the existing full system; ECW additions are traditional algorithms',
                      'absolute rope count and contact accuracy have no ground truth'],
        policy='Frozen baseline is never tuned on transfer videos; runner plumbing is separately versioned and parity-tested.',
        reference_results=dict(BG4=str(PARENT/COMPONENTS[4]/'full_003'),ECW=str(PARENT/COMPONENTS[5]/'delivery')))
    dump(out/'definition.json',definition)
    rows.append(dict(path='definition.json',source='new_baseline_definition',bytes=(out/'definition.json').stat().st_size,sha256=sha(out/'definition.json')))
    import cv2
    packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','torch','Pillow','matplotlib','imageio-ffmpeg']}
    packages['cv2_runtime']=cv2.__version__
    env=dict(python=sys.version,platform=platform.platform(),packages=packages)
    dump(out/'environment.json',env);rows.append(dict(path='environment.json',source='runtime_metadata',bytes=(out/'environment.json').stat().st_size,sha256=sha(out/'environment.json')))
    references=[]
    for p in [PARENT/COMPONENTS[4]/'full_003/compact_validation.json',PARENT/COMPONENTS[5]/'delivery/manifest.json',PARENT/COMPONENTS[5]/'delivery/experiment_results.json']:
        references.append(dict(path=str(p),sha256=sha(p)))
    dump(out/'manifest.json',dict(version='RES-1.0',files=rows,reference_outputs=references,source_count=len(rows)))
    print('Sealed',len(rows),'files',sum(r['bytes'] for r in rows),'bytes')
if __name__=='__main__':main()
