"""Attach the actual fixed48-position human/model visual review receipt."""
import json,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
review=read(H/'visual_review/review.json')
assert review['status']=='complete' and len(review['reviewed_videos'])==16 and review['positions_reviewed']==48
report=read(H/'experiment_report.json')
report['actual_fixed_position_visual_review']=dict(videos=16,positions=48,path='visual_review/review.json',
    sha256=sha(H/'visual_review/review.json'),scope='fixed3s/midpoint/end, corresponding four-panel rope-and-coil regions',
    identity_accuracy_established=False)
(H/'experiment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
p=H/'REPORT.txt';text=p.read_text(encoding='utf8')
if '固定48位置实际目视复核' not in text:
    text+='\n固定48位置实际目视复核\n'
    text+='16段每段约3秒/中点/末尾，共48处对应原图、B热图、候选和最终位置的卷面/活动绳区域，已实际用view_image查看，记录见visual_review/review.json。完整原图预览仍保留。\n'
    text+='观察到灰线随当前卷面、亮暗分布和可见工具遮挡形成不同的行与缺口；未把几何摘要变化直接当作真实材料运动或准确率。\n'
    text+='仍有暗侧/边缘断线；L_run若干位置未知或仅短前缀；R_static中点/末尾虽然原图和热图有明显斜绳，最终仍是STALE；左右启停片段也未证明完整贴面接续。\n'
    text+='因此本轮确认了逐帧新观测及同帧灰线接续的数据流程，未确认完整目标身份、普遍接续改善或在线实时能力。\n'
    p.write_text(text,encoding='utf8')
receipt=read(H/'delivery_receipt.json');receipt.update(report_sha256=sha(H/'experiment_report.json'),visual_review_sha256=sha(H/'visual_review/review.json'))
(H/'delivery_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(status='complete',videos=16,positions=48,report_sha256=receipt['report_sha256'])))
