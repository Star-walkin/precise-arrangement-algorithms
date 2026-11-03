"""Register completed evidence and hash changed/navigation files; no data moves."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    audit = json.loads((HERE/'audit_repair_receipt_002.json').read_text(encoding='utf-8'))
    assert audit['exit_code'] == 0 and not audit['errors']
    nav = ROOT/'数据资产/README.md'
    prior_manifest = HERE/'delivery_hash_manifest.json'
    before = (json.loads(prior_manifest.read_text(encoding='utf-8'))['data_navigation_before_sha256']
              if prior_manifest.exists() else sha(nav))
    marker = 'data_expansion_d1_20261102_v1'
    text = nav.read_text(encoding='utf-8-sig')
    if marker not in text:
        note = '\n\n## 2026-11-03 自动弱监督数据扩充 D1\n\n[D0/D1实际报告](../experiments/entry_conditioned_rope_path/data_expansion_d1_20261102_v1/REPORT.txt) · [292张原图与D1原始响应](../experiments/entry_conditioned_rope_path/data_expansion_d1_20261102_v1/final_292/) · [数据与展示用途索引](../experiments/entry_conditioned_rope_path/data_expansion_d1_20261102_v1/final_292_index.json)。24新增真实图/3来源组，原图检查后接受156段自动弱中心共30229px、198段未知；旧10图保留，7训练组与历史评价3组隔离。D0/D1从A750全参数各600步，AUTO部分后段5px参考覆盖25.88%→35.99%，邻圈选择性和旧图拟合退化；无完整弯后或边缘身份真值。292PNG/301新前向浮点已实际导出，全量像素与数据审计通过；原失败审计及线程复现修复另存，不覆盖历史。自动弱标注不等于独立人工真值，292图中24新训练图、208同训练来源展示帧、20历史评价来源展示帧、40其他开放来源帧；不是292张独立测试。原图与旧基线不改，无设备、保留集或S1访问。\n'
        with nav.open('ab') as f:
            f.write(note.encode('utf-8'))
    files = list(HERE.glob('*.py')) + list(HERE.glob('*.ps1')) + list(HERE.glob('*.txt'))
    files += [ROOT/'AGENTS.md', nav, HERE.parent/'README.md',
              HERE/'protocol.json', HERE/'dataset/index.json', HERE/'dataset/statistics.json',
              HERE/'selected_checkpoints.json', HERE/'final_292_index.json',
              HERE/'experiment_report.json', HERE/'audit_repair_receipt_002.json',
              HERE/'final_visual_review.json']
    record = dict(date_local='2026-11-03',scope='completed offline experiment/navigation only',
                  data_navigation_before_sha256=before,data_navigation_after_sha256=sha(nav),
                  no_git_initialized=True,no_original_or_frozen_prior_artifact_changed=True,
                  files=[dict(path=str(x),sha256=sha(x)) for x in sorted(set(files))])
    (HERE/'delivery_hash_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Registered and hashed',len(record['files']),'files.')

if __name__ == '__main__':
    main()
