"""Append actual offline evidence to project navigation; preserve prior records."""
from common import HERE,ROOT,load,dump,sha
r=load(HERE/'experiment_report.json')
assert r['exit_code']==0 and r['processed_images']==292 and r['audit']['result']['status']=='passed'
counts=r['path_output_counts'];connected=counts.get('straight_free_to_F3_center_track',0);free=counts.get('free_fragment_no_resolved_connection',0)
rel='experiments/entry_conditioned_rope_path/postprocess_b_20261031_v5'
line=(f'2026-10-31新增[热图接续H5 / IR-H5.0.0]({rel}/experiment_report.json)：同292图/301条件复用B响应；14张固定失败/对照开发、8张冻结后开放素材核查，既有保留集不读。按扣除入口/未知的实测直段长度分48px档，再比较转向、受支持F3中心轨迹与图像；取消2000px评分截止，限制穿过卷面后的远侧域外延长。热图漏段仅在A7有限观测和原图双边同时通过时单独补回；同A7相反条件按已实测外→内支持判任务方向。实际292/292退出0，{connected}图接轨、{free}图仅自由段；108旧入口未解决仍留400分母。0195反例提前转向有局部改善，残余漏段、方向/末端歧义和弱尾保留，未建立“几乎全部正确”或完整同根准确率。无训练/新前向/标签/保留集/设备访问，不替换既有基线。')
files=[ROOT/'AGENTS.md',ROOT/'数据资产/README.md',ROOT/'experiments/entry_conditioned_rope_path/README.md']
before={str(p):sha(p) for p in files}
for p in files:
    content=p.read_text(encoding='utf-8-sig')
    if 'postprocess_b_20261031_v5/experiment_report.json' in content:continue
    if p.name=='AGENTS.md':
        marker='- V14训练拟合接近100%';assert marker in content
        content=content.replace(marker,'- '+line+'\n'+marker,1)
    else:
        text=line.replace(']('+rel+'/','](D:/precise_arrangement/'+rel+'/')
        content+='\n\n## 2026-10-31：热图接续 H5 / IR-H5.0.0\n\n'+text+'\n\n'+f'[292张四栏图](D:/precise_arrangement/{rel}/all_001/四栏检测结果) · [固定案例对照](D:/precise_arrangement/{rel}/comparison_all_001)\n'
    p.write_text(content,encoding='utf-8')
dump(HERE/'modification_receipt.json',dict(git_repository=False,new_directory=str(HERE),
    index_changes={str(p):dict(before_sha256=before[str(p)],after_sha256=sha(p)) for p in files},
    code_sha256={str(p):sha(p) for p in HERE.glob('*.py')},
    frozen_prediction_sources_changed=False,original_data_changed=False,model_changed=False,
    labels_read=False,reserved_data_accessed=False,devices_connected=False))
print('H5 actual evidence registered')
