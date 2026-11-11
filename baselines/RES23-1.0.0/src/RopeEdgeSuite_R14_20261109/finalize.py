import subprocess
import cv2,numpy as np
import imageio_ffmpeg
from common import *
def main():
    video=read(ROOT/'delivery/video_receipt.json');audit=read(ROOT/'audit_receipt.json');test=read(ROOT/'test_receipt.json');f3=read(ROOT/'F3_integration_receipt.json')
    assert video['decoded_frames']==600 and audit['passed'] and test['passed'] and f3['passed']
    assert sha(video['path'])==video['sha256']
    prior_checks=[]
    for folder in [R11,R12,R13]:
        manifest=read(folder/'delivery_manifest.json')
        for q in manifest['files']:
            p=Path(q['path'])
            if not p.is_absolute():p=folder/p
            assert sha(p)==q['sha256'],str(p)
        prior_checks.append(dict(version=folder.name,files_unchanged=len(manifest['files'])))
    clip=ROOT/'delivery/RES14_SG2_BG7_113745_boundary_30s-42s.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-ss','30','-i',video['path'],'-t','12',
        '-an','-c:v','libx264','-crf','21','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(clip)],check=True)
    cap=cv2.VideoCapture(str(clip));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(1080,1920,3);n+=1
    fps=cap.get(5);cap.release();assert n==120 and fps==10
    dump(ROOT/'delivery/clip_receipt.json',dict(path=str(clip),decoded_frames=n,fps=fps,source_seconds=[30,42],sha256=sha(clip)))
    s=audit['statistics'];diag=read(ROOT/'R11_diagnosis.json')['stats'];hidden=read(ROOT/'hidden_strip_validation.json')
    eligible=[r for r in hidden['rows'] if r['eligible_reference']]
    accepted=[r for r in eligible if r['prediction_valid']]
    report=f'''RES-1.4 / SG2 + BG7 可见锚点约束的遮挡边界拟合

针对视频：RopeEdgeSuite_R11_20261108/runs/dual_20260923_113745_refined/delivery/RES11_BG5_dual_20260923_113745_refined_full.mp4
源视频：113745 左相机，600帧，4096×3000，10fps；本轮只读取该源视频。

结论
完成了边界算法修改和完整视频验证。图中37.0秒的错误局部凸起不再作为接受边界；排除活动绳/宽亮带后，使用其他可见边缘拟合连续轮廓。遮挡段明确显示为橙色模型估计，不冒充当前像素观测。当前跟踪使用已经验证的SG2同源同帧结果。
这一轮是边界与跟踪验证版，未重新运行全片F3与根数计数器，也没有证实实体贴边窗口。原R11、R12和R13交付均保持不变。

一、31秒后“跟踪不足/ECW证据不足”的原因
R11仍使用旧跟踪器。31.0～59.9秒共290帧：活动绳跟踪检查仅{diag['R11']['track_pass']}帧通过，ECW局部测量仅{diag['R11']['ecw_pass']}帧通过。
第316帧记录为current_S1_support_span_insufficient；第350、370、390等帧记录为no_current_verified_E1，跟踪诊断为no_previous_valid_current_S1。也就是连续支持段被打断后失锁，随后重捕获又没有可靠入口，形成长时间中断。
第370帧ECW直接返回no_S1_current_axis：它缺少当前活动绳轴线，是上游失锁引起的连锁失败。旧ECW还要求自由绳两侧都有黄色背景（two_yellow_flanks_insufficient），并在第一处大缺口截断点集；近卷面时灰色背景、反光和遮挡会使该条件失效。
SG2已经实现有限缺口的分段实测拟合、0.5秒搜索记忆、连续两帧身份确认和当前卷面搜索范围；本轮接入其来源与像素绑定的同帧测量。31秒后跟踪检查通过{diag['SG2']['track_pass']}/290帧。41.2秒仍没有当前测量，没有沿用历史解析式冒充结果。
SG2解决跟踪可用性，不等于已看到入卷接触点。视频分别显示绳轴状态、计数列遮挡、边界推断和局部贴边证据，避免把所有问题都显示成“活动绳跟踪不足”。

二、为什么旧蓝线会跟着活动绳/导杆走
旧BG5.1并非没有拟合：它先做四次多项式鲁棒拟合，再用候选点的局部残差PCHIP插值修正。错误点成片出现时，与整体模型的残差可能不大；局部修正还会把最终曲线拉回错误点。
“黄色在上、灰色/银色在下”不足以区分卷面、活动绳和银色导杆。截图中的细活动绳和较粗银色导杆都是干扰源。曲线光滑不能替代点的身份判断。
原蓝线的定义是黄色法兰与卷面之间的可见交界，尚不是通过实体标定确认的法兰内侧接触线。

三、BG7修改
1. 使用同帧BG5候选点作为输入，不直接接受其最终局部曲线。按当前亮带宽度、连续跨度及SG2活动绳附近排除带，先去除干扰点。高反光区域也可被保守排除，并非所有红色区域都被语义识别为导杆。
2. 用覆盖足够范围、曲率方向合理且d附近有可见支持的可靠帧建立形状。形状更新还需与已有形状相容，避免遮挡时把错误轮廓写入模型。只使用当前及过去帧。
3. 全局曲线使用y=c0+c1*u+c2*u²+c3*u³+c4*u⁴，u=(x−0.42W)/(0.30W)。当前可见点鲁棒拟合整体位置和倾斜；遮挡时固定高阶形状，不让局部错误点决定整条边缘。不再叠加PCHIP局部残差修正。
4. 所有可见段和推断段使用同一解析式，d处数值也来自该式。青色表示可见锚点支持，橙色表示同一曲线的推断段。改变历史形状或去掉空间分块后，检查d处预测敏感度；缺少当前锚点、形状过时或拟合不一致时拒绝输出。
5. 区分轮廓拟合资格与计数边界资格。能画出轮廓，不能证明遮挡后存在可计数的像素。计数列受遮挡或敏感度超过0.5个先验绳距时，不提供计数资格；显示轮廓的宽松门槛不用于认证根数。
6. ROI适配器只保留当前卷面掩膜与拟合边界下方、遮挡之外的交集，不凭模型补出纹理像素。F3可以在剩余可用卷面上运行；若d邻域不完整，计数仍不可用。

四、验证结果
600帧来源、同帧绑定、因果性、排除点不参与拟合、解析式/曲线/d交点一致性检查全部通过；6项功能回归测试通过。
全片轮廓拟合检查通过{s['outline_fit_pass']}/600；其中d附近有当前支持{s['locally_supported_d']}帧、d为模型补全{s['model_only_d']}帧。31秒后轮廓通过{s['outline_after31_pass']}/290。
单独的计数边界资格通过{s['count_boundary_eligible']}/600，尚未与全片F3/活动绳交点联合认证，不能视作这么多帧根数正确。检测到d附近宽亮带干扰{s['foreground_at_d']}帧。
37.0秒：旧边界y(d)=2172.55px，新拟合y(d)=2184.93px，开发敏感度指标约±3.79px。该处是遮挡推断，实际误差未知，未宣称新值是真值。
留出重建测试：预选8帧，隐藏d左右各180px的当前边缘点；7帧有足够可比较的留出点，6帧拟合通过，1帧因敏感度过大被拒绝。通过样本的中位绝对偏差范围{min(r['median_abs_error_px'] for r in accepted):.2f}～{max(r['median_abs_error_px'] for r in accepted):.2f}px；各样本90%分位偏差最大{max(r['p90_abs_error_px'] for r in accepted):.2f}px。参考是留出的检测点，不是人工边界真值，也不是遮挡后的真实轮廓。
抽取141/261/336/370/380/388/461/581共8帧测试新ROI与原01/F3接口，7帧实际重新检测出绳边，380帧因拟合失败未运行F3。370/388帧虽然其他卷面有F3点，d处被遮挡，计数资格仍为否。
这些数值是检查通过率和留出一致性，不能当作实体边界或剩余根数准确率。

五、尚未解决
当前方法减少了活动绳与导杆对边界拟合的拉偏；长时间只有单侧可见、形状变化、相机移动、暗色遮挡仍可能使模型不确定。
本轮没有重新标定d，2238.7px仍是参考值；没有恢复已验证的实体贴边窗口和全片剩余根数。下一步需要把入卷点附近的当前绳路证据与该模型边界联合核验。完整F3计数也应在该边界模型验证后重跑，不能把此前ROI上的结果直接贴上新边界。

交付
完整边界/跟踪对照视频：{video['path']}
30～42秒片段：{clip}
37秒对照画面：{ROOT/'delivery/preview_370.jpg'}
核心实现：boundary_bg7.py；ROI适配：roi_adapter.py；完整逐帧结果：boundary_final。
诊断：R11_diagnosis.json；验证：audit_receipt.json、hidden_strip_validation.json、F3_integration_receipt.json。
'''
    (ROOT/'修改与验证报告.txt').write_text(report,encoding='utf8')
    dump(ROOT/'summary.json',dict(boundary=s,tracking=diag,previous_delivery_checks=prior_checks,
        full_video_F3_rerun=False,full_video_counter_rerun=False,physical_contact_accuracy_validated=False,
        comparison_frame=370,source_video_sha256=read(ROOT/'boundary_final/binding.json')['source_sha256']))
    files=list(ROOT.glob('*.py'))+list(ROOT.glob('*.json'))+[ROOT/'修改与验证报告.txt']+list((ROOT/'delivery').glob('*'))
    files+=list((ROOT/'boundary_final').rglob('*.json'))+list((ROOT/'boundary_final').rglob('*.json.gz'))
    files+=list((ROOT/'F3_integration').glob('*'))
    files=[p for p in files if p.name!='delivery_manifest.json']
    dump(ROOT/'delivery_manifest.json',dict(status='complete',scope='boundary and tracking experiment; sampled F3 integration',
        files=[dict(path=str(p.resolve()),size=p.stat().st_size,sha256=sha(p)) for p in files]))
    print(json.dumps(dict(complete=True,files=len(files),boundary=s,prior_preserved=prior_checks),ensure_ascii=False))
if __name__=='__main__':main()
