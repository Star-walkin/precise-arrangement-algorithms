"""Register completed M1 evidence without moving/replacing historic assets."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    report=load(HERE/'experiment_report.json');audit=load(HERE/'audit_delivery_receipt.json')
    assert audit['exit_code']==0 and sha(HERE/'REPORT.txt')==report['report_sha256']
    marker='manual_r1_20261103_v1'
    note=('2026-11-03 [人工监督M1＋匹配R1](experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/REPORT.txt)：'
          '24人工目标/51邻绳负线与旧10共同训练，U-Net从A750全参数800步；新空间特征上369关系对/600步。'
          '292原像素三栏/301条件实际前向与路径审计通过。两处目标邻绳代理响应差D1→U1 AUTO .1617→.5662，'
          '但新R1相对旧R1固定8图AUTO后段37.86%→33.37%、CONDITIONAL31.43%→32.41%，非一致净增益；'
          'pair0075 E1入口/跨圈错误保留，未验证完整身份或真实末端，不替换旧基线。大量新产物在C:/precise_arrangement_results/manual_r1_20261103_v1；无设备/保留集/S1访问。')
    edits=[]
    agents=ROOT/'AGENTS.md';text=agents.read_text(encoding='utf-8-sig');before=sha(agents)
    if marker not in text:
        anchor='- V14训练拟合接近100%'
        assert text.count(anchor)==1
        text=text.replace(anchor,'- '+note+'\n'+anchor)
        agents.write_text(text,encoding='utf-8')
    edits.append(dict(path=str(agents),before_sha256=before,after_sha256=sha(agents)))
    notes={BASE/'README.md': '\n\n## 2026-11-03 人工监督M1＋匹配R1\n\n[实际报告](manual_r1_20261103_v1/REPORT.txt) · [292三栏原像素PNG](C:/precise_arrangement_results/manual_r1_20261103_v1/final_292/) · [来源/模型/入口索引](manual_r1_20261103_v1/final_292_index.json)。24人工目标中心/51红邻绳，旧10保留；U-Net800步、新R1 369对/600步，301同次新响应/特征与路径。有限邻圈可分性改善，但新R1未全面优于旧R1，错误入口、跨圈和拒识仍保留；不报告完整身份准确率，不替换旧基线。292/301独立像素/来源审计通过，无设备或保留集访问。\n',
           ROOT/'数据资产/README.md': '\n\n## 2026-11-03 人工中心与邻绳监督M1\n\n[实际报告](../experiments/entry_conditioned_rope_path/manual_r1_20261103_v1/REPORT.txt) · [292图](C:/precise_arrangement_results/manual_r1_20261103_v1/final_292/)。用户确认24图、24目标中心45039.916px及51红邻绳17484.611px已原样冻结；未知保持ignore，5图监督管外缘冲突仅派生像素排除，不改人工线。与旧10共7训练来源组、旧8评价3组隔离。U-Net800步/R1 600步，原始热图与路径分开展示；邻圈代理区分改善不等于完整身份，明确pair0075入口/跨圈失败保留。D盘空间不足，仅本轮新dataset/inference/paths经junction存C，新PNG直接存C，原始数据及历史目录不移动。审计292/301通过，无保留集或设备访问。\n'}
    for path,value in notes.items():
        before=sha(path)
        if marker not in path.read_text(encoding='utf-8-sig'):
            with path.open('ab') as f:f.write(value.encode('utf-8'))
        edits.append(dict(path=str(path),before_sha256=before,after_sha256=sha(path)))
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+list(HERE.glob('*.txt'))
    files += [HERE/'protocol.json',HERE/'experiment_report.json',HERE/'final_292_index.json',
              HERE/'selected_checkpoints.json',HERE/'r1_actual_audit.json',HERE/'audit_delivery_receipt.json',
              HERE/'dataset/index.json',HERE/'model/training_receipt.json',HERE/'relation/model/training_receipt.json',
              HERE/'evaluation/summary.json',HERE/'visual_review/review_receipt.json']
    receipt=dict(date_local='2026-11-03',no_git_initialized=True,original_assets_or_user_annotations_modified=False,
                 navigation_changes=edits,files=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(files))],
                 bulk_directory='C:/precise_arrangement_results/manual_r1_20261103_v1',
                 new_junctions={str(HERE/n):str(Path('C:/precise_arrangement_results/manual_r1_20261103_v1')/n) for n in ['dataset','inference','paths']})
    (HERE/'delivery_hash_manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Registered M1 evidence and',len(receipt['files']),'file hashes.')
if __name__=='__main__':main()
