"""Append actual completed evidence to project navigation; preserve old entries."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    r=json.loads((HERE/'experiment_report.json').read_text(encoding='utf8'))
    if r['videos']!=16 or r['status']!='completed_offline':raise RuntimeError('Incomplete output')
    tag='活动绳整链P1与视频时序V1'
    detail=(f'2026-11-01新增[活动绳整链基准P1 / IR-P1.0.0](D:/precise_arrangement/工程导航/05_实验版本档案/活动绳整链基准_P1_20261101/README.txt)与'
        f'[视频时序V1 / IR-V1.0.0](D:/precise_arrangement/experiments/active_rope_video_20261101_v1/experiment_report.json)：'
        f'原B750＋H6整链338份快照封存，H7不纳入；400图溯源为16段视频390帧及10张独立照片。'
        f'16/16完整来源视频离线运行，{r["analysis_frames"]}次分析、{r["new_B_forwards"]}次原B新前向，'
        f'最近5次/2秒已接受历史约束当前最多3候选，异常拒绝不回填；1秒旧参考过期后UNKNOWN，连续新候选另立重捕获状态。'
        f'输出2Hz分析/5fps原速四子图。场景沿用最早开放锚点SAM2/F3与因果配准，并非逐帧完整F3重测或任意视频冷启动。'
        f'时序接受与平稳不等于同根身份，未证明在线实时、完整贴面接续或真实准确率；无训练、保留集或设备访问。')
    paths=[ROOT/'工程导航/05_实验版本档案/README.md',ROOT/'数据资产/README.md',
        ROOT/'experiments/entry_conditioned_rope_path/README.md']
    changes=[]
    for p in paths:
        text=p.read_text(encoding='utf8');before=sha(p)
        if tag not in text:
            text+='\n\n## 2026-11-01：'+tag+'\n\n'+detail+'\n\n[16段四子图视频](D:/precise_arrangement/experiments/active_rope_video_20261101_v1/四子图视频) · [来源映射](D:/precise_arrangement/experiments/active_rope_video_20261101_v1/source_trace/source_trace.json)\n'
            p.write_text(text,encoding='utf8')
        changes.append(dict(path=str(p),before=before,after=sha(p)))
    p=ROOT/'AGENTS.md';text=p.read_text(encoding='utf8');before=sha(p)
    if 'IR-V1.0.0' not in text:
        marker='- V14训练拟合'
        pos=text.index(marker)
        text=text[:pos]+'- '+detail+'\n'+text[pos:];p.write_text(text,encoding='utf8')
    changes.append(dict(path=str(p),before=before,after=sha(p)))
    (HERE/'registration_receipt.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(changes,ensure_ascii=True))
if __name__=='__main__':main()
