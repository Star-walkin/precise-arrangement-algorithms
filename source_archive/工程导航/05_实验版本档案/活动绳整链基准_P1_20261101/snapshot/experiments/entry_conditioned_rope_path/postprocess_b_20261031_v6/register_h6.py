from common import HERE,ROOT,load,dump,sha
r=load(HERE/'experiment_report.json');assert r['exit_code']==0 and r['audit']['result']['status']=='passed'
c=r['output_counts'];rel='experiments/entry_conditioned_rope_path/postprocess_b_20261031_v6'
line=(f'2026-10-31新增[热图接续H6 / IR-H6.0.0]({rel}/experiment_report.json)：同292图/301条件复用B；16固定图开发、旧8开放回归后全量。直接复用F3接受曲线及来源谱系、完整C4范围，以固定相邻边界对构建中心，源内有界推断另标。桥与贴面硬检非目标中心穿越/切触/重叠，贴面锁单轨；完整H5直轴保留，弱证据跨轨尾从任务路径中分离，真实自由投影交点须连续原图证据覆盖。实际292/292退出0，{c.get("straight_free_to_F3_center_track",0)}图接轨候选、{c.get("free_fragment_no_resolved_connection",0)}图仅前缀；108旧入口未解决仍留400分母。0345有单轨接续候选，0255非法桥被拒但完整接续未解决。无交叉/精确来源只证明几何约束，不能证明同根身份或几乎全对；漏段和源缺口保留。无训练/新前向/标签/保留集/设备访问，不替换F3/A7等既有基线。')
files=[ROOT/'AGENTS.md',ROOT/'数据资产/README.md',ROOT/'experiments/entry_conditioned_rope_path/README.md'];before={str(p):sha(p) for p in files}
for p in files:
    s=p.read_text(encoding='utf-8-sig')
    if rel+'/experiment_report.json' in s:continue
    if p.name=='AGENTS.md':
        marker='- V14训练拟合接近100%';assert marker in s;s=s.replace(marker,'- '+line+'\n'+marker,1)
    else:s+='\n\n## 2026-10-31：热图接续H6 / IR-H6.0.0\n\n'+line.replace(']('+rel+'/','](D:/precise_arrangement/'+rel+'/')+'\n'
    p.write_text(s,encoding='utf-8')
dump(HERE/'modification_receipt.json',dict(git_repository=False,new_directory=str(HERE),
    index_changes={str(p):dict(before_sha256=before[str(p)],after_sha256=sha(p)) for p in files},
    code_sha256={str(p):sha(p) for p in HERE.glob('*.py')},original_data_changed=False,model_changed=False,
    labels_read=False,reserved_data_accessed=False,devices_connected=False))
print('H6 actual results registered')
