"""Archive the existing B750/H6 pipeline without changing historical algorithms."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,datetime

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/entry_conditioned_rope_path'
H6=BASE/'postprocess_b_20261031_v6'
OUT=ROOT/'工程导航/05_实验版本档案/活动绳整链基准_P1_20261101'
ASSETS=ROOT/'数据资产/05_派生与实验结果'
F3ARCH=ROOT/'工程导航/05_实验版本档案/卷绳边界_全卷面基准_F3_20261028'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def freeze():
    OUT.mkdir(parents=True,exist_ok=False)
    files={};origin_sources={}
    def add(path,kind,expected=None):
        path=Path(path).resolve();digest=sha(path)
        if expected:assert digest==expected,(path,digest,expected)
        relative=path.relative_to(ROOT);dest=OUT/'snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        assert sha(dest)==digest
        files[str(relative)]=dict(origin_path=str(path),snapshot=str(dest.relative_to(OUT)),sha256=digest,bytes=path.stat().st_size,kind=kind)
        return digest
    # Existing model and the exact H1-H6 source chain, not H7C/D.
    h6=load(H6/'freeze_final.json');model=load(H6/'inputs.json')['model_B']
    add(model['path'],'model_B750',model['sha256'])
    for dirname in ['postprocess_b_20261030_v1','postprocess_b_20261030_v2','postprocess_b_20261031_v3','postprocess_b_20261031_v4','postprocess_b_20261031_v5','postprocess_b_20261031_v6']:
        folder=BASE/dirname
        for p in folder.glob('*.py'):add(p,'historical_pipeline_source')
        for name in ['inputs.json','freeze_final.json','experiment_report.json','environment.json','README.txt','review_receipt.json','all_001/summary.json','all_001/independent_geometry_audit.json']:
            if (folder/name).exists():add(folder/name,'historical_protocol_or_evidence')
    for p,h in (h6['sources']|h6['reused_sources']).items():assert sha(p)==h;origin_sources[p]=h
    old=BASE/'runs/run_20261030_002'
    for p,h in load(old/'frozen_upstream.json')['sources'].items():add(p,'network_pipeline_source',h);origin_sources[p]=h
    for name in ['frozen_upstream.json','PROTOCOL.md','REPORT.md','branches/B/model/config.json','data/index.json']:
        if (old/name).exists():add(old/name,'B_training_provenance')
    for p in [BASE/'paired_train.py',BASE/'prepare_auto_seeds.py',BASE/'postprocess_missing_entries.py',BASE/'batch_ab_preview.py']:
        if p.exists():add(p,'B_inference_or_entry_source')
    a7=ASSETS/'camera_condition_free_straight_20261030_v7'
    for p,h in load(a7/'freeze_all.json')['sources_sha256'].items():add(p,'A7_source',h);origin_sources[p]=h
    for name in ['freeze_all.json','experiment_report.json','protocol.json']:
        add(a7/name,'A7_protocol')
    # Store F3's exact existing source/evidence archive and locked external manifest.
    f3=load(F3ARCH/'baseline_manifest.json')
    for item in f3['snapshot_files']:
        p=F3ARCH/item['snapshot_relpath'];add(p,'F3_frozen_archive',item['sha256'])
        if item.get('original_source_path'):
            op=Path(item['original_source_path'])
            if op.exists() and sha(op)==item['sha256']:add(op,'F3_original_layout_source',item['sha256']);origin_sources[str(op)]=item['sha256']
    for n in ['baseline_manifest.json','dependency_manifest.json','verification_receipt.json','README.md','verify_baseline.py']:
        add(F3ARCH/n,'F3_archive_index')
    a6=ASSETS/'camera_condition_incoming_template_20261029_v6'
    for p in a6.glob('*.py'):add(p,'automatic_entry_source')
    for p in a6.glob('*.json'):
        if p.stat().st_size<25_000_000:add(p,'automatic_entry_protocol')
    add(ASSETS/'camera_condition_images_20261009_v2/manifest.csv','open400_source_manifest')
    sources=load(H6/'inputs.json')
    responses=[]
    for job in sources['jobs']:
        for k,p in enumerate(job['response_files']):responses.append(dict(id=job['id'],entry_index=k,path=p,sha256=sha(p)))
    add(H6/'all_001/summary.json','H6_gallery_manifest')
    dump(OUT/'external_response_manifest.json',dict(responses=responses,not_copied=True,source_images=400,eligible_images=292,unresolved_entries=108))
    manifest=dict(name='活动绳整链基准 P1',version='IR-P1.0.0',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        requested_baseline='existing originalB750 + H6, before H7 heatmap trials',
        model_B=model,components=['SAM2/C4卷面范围','C3/D4/F1/F2/F3卷面边界','A6自动入口与A7自由直轴','冻结B750入口条件U-Net','H6单轨/不跨轨接续'],
        input_output='原图与其已生成上游观测 -> 自动短入口/方案B -> 原图热图自由段 -> F3中心轨迹候选 -> H6活动绳候选路径',
        source_images=400,eligible_images=292,unresolved_entries=108,source_snapshot_count=len(files),files=list(files.values()),
        exact_core_source_hashes=origin_sources,external_response_manifest_sha256=sha(OUT/'external_response_manifest.json'),
        standalone_arbitrary_new_frame_verified=False,saved_upstream_scene_predictions_required=True,
        evidence_scope='Historic open400 replay baseline; no new accuracy or physical identity validation from naming/freezing.',
        video_adaptation_separate='IR-V1.0.0 uses current video images and causal history; it is not silently part of P1.',
        labels_read=False,reserved_access=False,devices=False,old_sources_or_models_changed=False)
    dump(OUT/'baseline_manifest.json',manifest)
    text=('活动绳整链基准 P1 / IR-P1.0.0\n\n'
        '固定用户暂停热图优化时的原方案B第750步模型与H6接续。H7的C/D不纳入。\n'
        '处理链：原图及SAM2/C4卷面范围 → C3/D4/F3边界转中心轨迹；A6/A7自动入口 → B条件热图 → 原图/热图直段 → H6不跨轨接续。\n'
        '历史400开放图：292张具备入口、108张入口未解决。命名与冻结不等于识别准确率或完整同根验收。\n\n'
        '重要边界：历史运行使用预先生成的上游SAM2/F3等观测。档案锁定该完整依赖链，但并不声称任意新原图已经实现无先验冷启动。\n'
        '源码、B权重和协议按原项目层级保存到snapshot；大型原始数据/响应/卷面逐图结果保持原址并锁哈希。旧运行与失败不改。\n'
        '若后续源码改变，先用snapshot回溯差异；不要直接覆盖原文件。verify_baseline.py默认仅验证档案副本和B响应字节，不导入算法、不读真值、不连接设备。\n'
        '视频开发独立为IR-V1.0.0，新增因果历史、异常跳动拒绝与过期/重捕获。它的传播线不能称为当帧完整F3重跑。\n')
    (OUT/'README.txt').write_text(text,encoding='utf8')
    print('Archived',manifest['version'],len(files),'files',str(OUT))

if __name__=='__main__':freeze()
