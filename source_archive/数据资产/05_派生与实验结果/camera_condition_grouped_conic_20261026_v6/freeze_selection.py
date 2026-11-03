"""以已保存证据冻结C3第二版，随后才允许全量输出；不改历史文件。"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    output=HERE/'freeze_all_001.json'
    if output.exists():raise FileExistsError(output)
    pilot=HERE/'pilot_002';old=HERE/'pilot_001';ev=HERE/'evaluation_002'
    audit=read(pilot/'integrity_audit.json');score=read(ev/'summary.json')
    assert audit['passed']==38 and audit['failed']==0 and not audit['global_errors']
    assert score['predeclared_continue_criterion_met'] and score['source_unchanged']
    context=read(pilot/'run_context.json');hashes=context['source_sha256']
    for path,value in hashes.items():assert sha(Path(path))==value
    assert read(ev/'run_context.json')['conic_module_sha256']==hashes[str(HERE/'conic_family.py')]
    # 未遮点的38张可视结果与首轮逐字节相同，所以首轮的视觉优缺点继续成立。
    comparisons=[]
    for directory in ('overlays','smooth_overlays','group_overlays','qa_compare'):
        for path in sorted((old/directory).glob('*/*.jpg')):
            current=pilot/path.relative_to(old)
            comparisons.append({'relative':path.relative_to(old).as_posix(),
                                'sha256':sha(current),'identical':sha(path)==sha(current)})
    assert len(comparisons)==152 and all(row['identical'] for row in comparisons)
    evidence=[pilot/'integrity_audit.json',pilot/'summary.json',ev/'summary.json',ev/'run_context.json',
              ev/'comparison_to_c2_c3_001_and_linear.json',HERE/'independent_qa_pilot_001.json',
              HERE/'evaluation_protocol.json',HERE/'revision_note.json']
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'name':'整组圆锥 C3',
            'version':'RB-C3.0.0','selected_pilot':'pilot_002','selected_evaluation':'evaluation_002',
            'full_run':'all_001','source_sha256':hashes,
            'evidence_sha256':{str(p):sha(p) for p in evidence},
            'pilot_visuals_identical':comparisons,
            'decision':'User-requested grouped all-point single-conic implementation; fixed linear gate passes; not superior to C2.',
            'limits':'C3 proxy hit95.10% vs C2 97.29%; capped loss0.10068 vs0.04413. Known visible rejected groups retained. Same38 open development images, not overexposure truth. No devices or reserved data.'}
    output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'version':record['version'],'visual_files_identical':len(comparisons),'frozen':True}))

if __name__=='__main__':main()
