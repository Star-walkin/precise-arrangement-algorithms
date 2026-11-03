"""Attach actual final audit and fixed-sample review to the delivered record."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    audit=read(HERE/'all_001/independent_audit.json');visual=read(HERE/'audit/visual_review_agent.json')
    assert audit['status']=='passed' and audit['checked_videos']==16
    assert visual['status']=='complete' and visual['reviewed_fixed_positions']==48
    p=HERE/'experiment_report.json';r=read(p)
    r['independent_audit']=dict(path='all_001/independent_audit.json',sha256=sha(HERE/'all_001/independent_audit.json'),
        checks=audit['checks'],exit_code=audit['exit_code'],complete_videos_decoded=16,float_arrays_checked=1528)
    r['visual_review']=dict(path='audit/visual_review_agent.json',sha256=sha(HERE/'audit/visual_review_agent.json'),
        videos=16,fixed_positions=48,not_identity_accuracy=True)
    r['temporal_comparison']=dict(path='all_001/temporal_comparison_common_domain.json',
        sha256=sha(HERE/'all_001/temporal_comparison_common_domain.json'),shared_domain_pairs=712,
        unchanged=682,lower_transverse_change=24,higher_transverse_change=6,
        baseline_p95_px_per_s=23.096,temporal_p95_px_per_s=22.156,
        interpretation='limited temporal candidate selection effect; rejections and stable errors are not accuracy gains')
    p.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
    report=HERE/'REPORT.txt';text=report.read_text(encoding='utf8')
    marker='最终验收回执（实际运行）'
    if marker not in text:
        text+='\n'+marker+'\n'
        text+='全量16/16退出0。all_001/independent_audit.json：109,920项结构检查通过，1,528浮点数组核查，16段完整输出视频解码核查，错误/警告0。\n'
        text+='audit/visual_review_agent.json：16段×入口后/中点/末尾=48固定位置已实际查看。截图漏末尾时仅从既有输出视频补解码，没有新增前向或替换失败帧。\n'
        text+='仍见101643右30秒及左右启停末尾接受前缀过短；L_run三处固定位置均UNKNOWN且卷面响应弱混杂；左启停76秒原图可见明绳但UNKNOWN。\n'
        text+='这些失败保留在原速视频，未以结构审计或低抖动替代身份准确率。\n'
        report.write_text(text,encoding='utf8')
    if '共同弧长时序对照' not in text:
        text+='\n共同弧长时序对照\n'
        text+='采用all_001/temporal_comparison_common_domain.json：两帧×基线/时序四条路径共用同一弧长和法向。712个共同域配对中682不变、24横向变化降低、6增大；p95从23.096到22.156 px/s。\n'
        text+='原始候选几何大多未改变，仅显示有限候选选择作用；495次有当前图像候选的观测仍被拒，不能由抖动变小推断身份准确。单次处理耗时中位2.209秒、p95 5.108秒，未达到2Hz计算实时。\n'
        report.write_text(text,encoding='utf8')
    files=list((HERE/'四子图视频').glob('*.mp4'));assert len(files)==16
    receipt=dict(status='complete',videos=len(files),bytes=sum(p.stat().st_size for p in files),
        outputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)],
        report_sha256=sha(HERE/'experiment_report.json'),independent_audit_sha256=sha(HERE/'all_001/independent_audit.json'),
        source_protocol_sha256=sha(HERE/'all_001/protocol.json'),input_data_changed=False,devices=False)
    (HERE/'delivery_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='outputs'}))
if __name__=='__main__':main()
