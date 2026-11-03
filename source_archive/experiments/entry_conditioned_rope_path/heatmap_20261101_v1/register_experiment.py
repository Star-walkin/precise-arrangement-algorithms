"""Register completed experimental evidence without replacing frozen baselines."""
from common import HERE,BASE,ROOT,load,dump,sha
def main():
    r=load(HERE/'experiment_report.json');a=load(HERE/'postprocess/all_001/independent_geometry_audit.json')
    assert r['run_images']==292 and a['status']=='passed'
    rel='experiments/entry_conditioned_rope_path/heatmap_20261101_v1'
    trained='、'.join(f'{k}{v["steps"]}步' for k,v in r['training_actual'].items())
    line=(f'2026-11-01新增[热力图输入与监督权重对照H7 / IR-H7.0.0]({rel}/experiment_report.json)：'
        f'从同B750续训{trained}，共同{r["main_common_additional_step"]}步比较；10训练图/4组、8历史开放评价图/3组隔离，未新增目标正标注。'
        'C降低工程弱负权重并温和续训，D同条件增加保留原灰的LCN辅助通道；训练侧选择'
        f'{r["selected_variant"]}后冻结，完成48主评价记录（含9条缺失自动入口）及入口消融、同292图/301条件新前向和冻结H6后处理。'
        '左L_run反例来源未参加旧训练，旧监督97.03%仍ignore；400张图库不等于400张完整监督。'
        '新旧固定色标/浮点结果、改善与退化均保留，未建立几乎所有图清晰识别或完整同根身份准确率。'
        '原B/H6及其他基线保持不变；无设备/保留集或S1新标签访问。')
    files=[ROOT/'AGENTS.md',ROOT/'数据资产/README.md',BASE/'README.md'];before={str(p):sha(p) for p in files}
    for p in files:
        s=p.read_text(encoding='utf-8-sig')
        if rel+'/experiment_report.json' in s:continue
        if p.name=='AGENTS.md':
            marker='- V14训练拟合接近100%';assert marker in s;s=s.replace(marker,'- '+line+'\n'+marker,1)
            s=s.replace('维护日期：2026-10-31','维护日期：2026-11-01',1)
        else:
            s+='\n\n## 2026-11-01：热力图输入与监督权重对照H7\n\n'+line.replace(']('+rel+'/','](D:/precise_arrangement/'+rel+'/')+'\n'
            s+=f'\n[292张四栏图](D:/precise_arrangement/{rel}/postprocess/all_001/四栏检测结果) · [原图/旧B/新热图](D:/precise_arrangement/{rel}/postprocess/all_001/原图_旧B_新热力图) · [说明](D:/precise_arrangement/{rel}/REPORT.txt)\n'
        p.write_text(s,encoding='utf8')
    dump(HERE/'modification_receipt.json',dict(git_repository=False,new_directory=str(HERE),
        files={str(p):dict(before=before[str(p)],after=sha(p)) for p in files},code_sha256={str(p):sha(p) for p in HERE.glob('*.py')},
        historical_inputs_changed=False,existing_models_changed=False,new_experimental_models=True,production_changed=False,
        labels='Only existing authorized train/evaluation references; no S1 labels or protected data',devices=False))
    print('H7 actual results registered')
if __name__=='__main__':main()
