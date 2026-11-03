"""仅在全量输出和审计完成后，登记C3用户确认基准与C4开发结果。"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    target=HERE/'index_change_log.json'
    if target.exists():raise FileExistsError(target)
    s=read(HERE/'all_001/summary.json');a=read(HERE/'all_001/independent_audit.json')
    e=read(HERE/'evaluation_002/summary.json')
    assert s['planned']==s['completed']==400 and not s['failed']
    assert a['planned']==a['passed']==400 and not a['failed'] and not a['global_errors']
    known={
        ROOT/'数据资产/README.md':'a44d69821a618a92c9e97931d813facb3f4b408f3a049f8b5fafbaf1bb80509a',
        ROOT/'数据资产/05_派生与实验结果/README.md':'c33a1359219488dfb6997c19d8a83e7f035e284743e1620f4315361337cdff25',
        ROOT/'工程导航/05_实验版本档案/README.md':'9f6b6e62b20f7a3550bc409508f6384147f5dac2781107eb992c6192f042aa51'}
    for p,h in known.items():
        if sha(p)!=h:raise ValueError('Concurrent index change: '+str(p))
    baseline=ROOT/'工程导航/05_实验版本档案/卷绳边界_整链基准_C3_20261026/README.md'
    review=HERE/'all_001/review.html';report=HERE/'结果说明.html'
    new=e['bright']['overall']['v7']['all_point_hit_rate']
    old=e['bright']['overall']['c3']['all_point_hit_rate']
    entry=f"""## 2026-10-26：C3整链基准确认与C4全卷面试验

用户查看400图后认可C3展示效果尚可，已要求将完整链路作为后续钢丝绳边界开发基准：[整链基准 C3 / RB-C3.0.0]({baseline.as_posix()})。已封存104份源码、模型及证据快照，核验400原图和9,276逐图依赖，0失败；这是用户质性确认，不把此前C3/C2代理比较改写为真实准确率。

新增 [全卷面 C4 / RB-C4.0.0 的400图审阅页]({review.as_posix()}) 与 [结果及限制]({report.as_posix()})。工作ROI统一为原SAM2 v2卷面与B1亮带ROI的并集，原SAM另存参考；400张原色ROI（外部0）和400张全分辨率边界PNG分开保存，左右各200。重新运行完整工作区的活动绳屏蔽，筛选暗区黄点、整组单圆锥拟合并有限外推。全量400/400完成，运行异常0；独立文件/像素/结构审计400/400通过。{s['nonempty_curve_images']}图有曲线，{s['images_with_dark_observations']}图有新暗区点，空输出、拒识及首轮失败均保留。

固定38开发图旧亮区观测一致性C4为{new:.2%}、C3为{old:.2%}；边侧隐藏点代理与同输入简单C3外推完全相同，未证明暗区净增益。ruler_left_05的原SAM语义漏分仍在：并集只恢复已接受亮区，未恢复其余暗面；不将矩形补回区称为SAM自然轮廓。全量模型数量、像素数和结构通过不代表暗区真实逐根准确率或切点覆盖。C4等待人工复核，不替换本次用户选定的C3基准；未访问保留集、未连接设备或改变上线权限。

"""
    log=[]
    for p,before in known.items():
        content=p.read_text(encoding='utf-8')
        if 'C3整链基准确认与C4全卷面试验' in content:raise ValueError('Duplicate entry')
        # 标题后插入新记录，历史段落及其当时结论原样保留。
        first,rest=content.split('\n',1)
        p.write_text(first+'\n\n'+entry+rest.lstrip('\n'),encoding='utf-8')
        log.append({'path':str(p),'before_sha256':before,'after_sha256':sha(p),
                    'updated_utc':datetime.now(timezone.utc).isoformat(),
                    'change':'Added user-accepted C3 full-chain baseline and completed C4 open-development evidence; historical content retained'})
    target.write_text(json.dumps(log,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Updated three asset/version indexes')
if __name__=='__main__':main()
