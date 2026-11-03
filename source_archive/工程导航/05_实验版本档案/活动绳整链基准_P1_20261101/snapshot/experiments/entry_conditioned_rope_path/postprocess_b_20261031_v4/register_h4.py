"""Register actual offline evidence in existing navigation, without rewriting history."""
from pathlib import Path
from common import HERE,ROOT,load,dump,sha
report=load(HERE/'experiment_report.json')
assert report['exit_code']==0 and report['processed_images']==292
assert report['independent_audit']['summary']['status']=='passed'
counts=report['path_output_counts']
connected=counts.get('straight_free_to_F3_center_track',0)
free=counts.get('free_fragment_no_resolved_connection',0)
none=counts.get('no_candidate',0)
rel='experiments/entry_conditioned_rope_path/postprocess_b_20261031_v4'
line=(f'2026-10-31新增[热图接续H4 / IR-H4.0.0]({rel}/experiment_report.json)：同292图/301条件复用B浮点响应，有限角度/偏移重估直段，以中心高、双侧低热形态确定范围，A7仅初值；保护持续直脊，弱热图接轨不能自动胜过暂不接轨。实际292/292退出0，{connected}图接轨、{free}图仅自由段、{none}图无候选；108旧入口未解决仍留400分母。用户R_static095反例长斜段恢复，首版短亮段评分偏置及统一视野积分修订分别保留；热图形态与覆盖改善不等于完整同根身份验证。无训练/新前向/标签/保留集/设备访问，不替换既有基线。')
files=[ROOT/'AGENTS.md',ROOT/'数据资产/README.md',ROOT/'experiments/entry_conditioned_rope_path/README.md']
before={str(p):sha(p) for p in files}
for p in files:
    text=p.read_text(encoding='utf-8-sig')
    if 'postprocess_b_20261031_v4/experiment_report.json' in text:continue
    if p.name=='AGENTS.md':
        target='- V14训练拟合接近100%'
        assert target in text
        text=text.replace(target,'- '+line+'\n'+target,1)
    else:
        base='../' if p.parent.name=='数据资产' else ''
        if p.parent.name=='entry_conditioned_rope_path':base='../../'
        paragraph=line.replace(']('+rel+']',']('+base+rel+']')
        # Absolute paths avoid ambiguity in both navigation locations.
        paragraph=paragraph.replace(']('+rel+'/','](D:/precise_arrangement/'+rel+'/')
        text+='\n\n## 2026-10-31：热图接续 H4 / IR-H4.0.0\n\n'+paragraph+'\n\n'+f'[292张四栏图](D:/precise_arrangement/{rel}/all_001/四栏检测结果) · [固定13图原图/H3/H4对照](D:/precise_arrangement/{rel}/comparison_all_001/固定13图_原图_H3_H4)\n'
    p.write_text(text,encoding='utf-8')
dump(HERE/'modification_receipt.json',dict(git_repository=False,new_directory=str(HERE),
    modified_indexes={str(p):dict(before_sha256=before[str(p)],after_sha256=sha(p)) for p in files},
    new_code_sha256={str(p):sha(p) for p in HERE.glob('*.py')},
    historical_sources_changed=False,raw_data_changed=False,model_changed=False,
    labels_read=False,reserved_data_accessed=False,devices_connected=False))
print('registered H4 actual evidence')
