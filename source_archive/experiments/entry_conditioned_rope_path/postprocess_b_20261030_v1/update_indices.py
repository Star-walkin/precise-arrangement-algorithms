"""Register only actual H1 delivery evidence, preserving existing documents."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
report=load(HERE/'experiment_report.json')
summary=load(HERE/'all_001/summary.json')
assert summary['completed']==summary['planned']==292 and not summary['failures']
audits=sorted((HERE/'all_001').glob('independent_audit_*.json'),key=lambda p:p.stat().st_mtime_ns)
assert audits and load(audits[-1])['exit_code']==0
before=load(HERE/'documentation_before.json')
for row in before:assert sha(row['path'])==row['sha256'], 'Document changed during work; preserve it: '+row['path']
relative='experiments/entry_conditioned_rope_path/postprocess_b_20261030_v1'
absolute='D:/precise_arrangement/'+relative
paragraph=(
    '按用户指定方案B图库的292图、299个自动入口复推原B第750步模型；原400图中108个旧入口失败不改称无绳。'
    '固定12图开发及明确留档的可见范围/记录隔离修复后冻结，再跑预定8图和全量。实际292/292、退出0，'
    '原图与热图亮暗剖面、同轴重现、方向连续候选及F3弱提示；六步长条按多入口分行，另存分步图和估计绳带。'
    '独立结构审计通过，299份数值响应范围与旧B记录一致；结构计数不是身份准确率。'
    '导杆和早转旧圈部分改善，但暗尾多解、局部偏位、短尾和工具错入口仍保留；绿仅局部匹配，橙待确认，F3橙虚线不入主掩膜。'
    '未证明整根同根或相对基线普遍净增益，不替换F3/A6/A7；未训练、未读标签/保留集、未接设备，无HTML。')
entries={
 '数据资产/README.md':'# 数据资产入口',
 '数据资产/05_派生与实验结果/README.md':'# 派生与实验结果',
 'experiments/entry_conditioned_rope_path/README.md':'# 入口条件化目标绳段离线实验',
}
title='## 2026-10-30：热图接续 H1 / IR-H1.0.0（方案B后处理候选）'
for relative_doc,heading in entries.items():
    p=ROOT/relative_doc;text=p.read_text(encoding='utf-8')
    assert title not in text and heading in text
    block=(heading+'\n\n'+title+'\n\n'
      f'[292张六步长条图]({absolute}/all_001/长条对照图) · [分步图]({absolute}/all_001/分步图像) · '
      f'[算法说明]({absolute}/H1算法与结果说明.txt) · [实际报告]({absolute}/experiment_report.json)。'+paragraph+'\n')
    p.write_text(text.replace(heading,block,1),encoding='utf-8')
p=ROOT/'AGENTS.md';text=p.read_text(encoding='utf-8');anchor='- V14训练拟合接近100%'
assert anchor in text and '热图接续H1 / IR-H1.0.0' not in text
bullet=(f'- 2026-10-30新增[热图接续H1 / IR-H1.0.0]({relative}/experiment_report.json)：'
  '复用用户方案B冻结U-Net及原自动入口，对同开放400图中的292图/299入口输出六步长条；108个旧入口失败另记。'
  '数值推理和结构审计通过，导杆后直段接续有质性改善；暗尾多解、局部偏位及工具错入口仍在，橙虚线F3尾段仅供复核、不并主掩膜。'
  '并未证明整根身份或普遍净增益，不替换F3/A6/A7，无新训练、标签读取或设备操作。\n')
p.write_text(text.replace(anchor,bullet+anchor,1),encoding='utf-8')
after=[dict(path=row['path'],before_sha256=row['sha256'],after_sha256=sha(row['path'])) for row in before]
dest=HERE/'documentation_change_receipt.json';assert not dest.exists()
dest.write_text(json.dumps(dict(files=after,report_sha256=sha(HERE/'experiment_report.json')),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'updated_documents':len(after),'report_exists':True,'actual_all_count':summary['completed']}))
