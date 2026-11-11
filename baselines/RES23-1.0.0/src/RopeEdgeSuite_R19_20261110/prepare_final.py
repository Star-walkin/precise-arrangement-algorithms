from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'RopeEdgeSuite_R18_20261110'
for name in ['camera_cg2.py','engine.py','run_fresh.py','control_audit.json','sg5_regression_test.json']:
    if (ROOT/name).exists():continue
    text=(OLD/name).read_text(encoding='utf8')
    if name.endswith('.py'):
        text=text.replace('RES-1.8','RES-1.8.1').replace('RES18','RES181').replace('BG10','BG11')
        text=text.replace('from boundary_bg10 import Boundary','from boundary_bg11 import Boundary')
        text=text.replace("ROOT/'boundary_bg10.py'","ROOT/'boundary_bg11.py',ROOT/'boundary_bg10.py'")
    (ROOT/name).write_text(text,encoding='utf8')
text=(OLD/'build_results.py').read_text(encoding='utf8')
text=text.replace('import html,shutil','import html,shutil,os').replace('from compare_results import main as compare','from mixed_comparison import main as compare,choose_run')
text=text.replace("id=item['id'];run=ROOT/runs_name/id;receipt=read(run/'receipt.json');stats=comparisons[id]","id=item['id'];run=choose_run(id);folder=os.path.relpath(run,ROOT).replace(chr(92),'/');receipt=read(run/'receipt.json');stats=comparisons[id]")
text=text.replace("poster=f'{runs_name}/{id}/preview_", "poster=f'{folder}/preview_")
text=text.replace("video=f'{runs_name}/{id}/{v.name}'", "video=f'{folder}/{v.name}'")
text=text.replace('{runs_name}/{id}/receipt.json','{folder}/receipt.json')
text=text.replace('RES-1.8','RES-1.8.1')
text=text.replace('BG10 边界分组拟合','BG10 边界分组拟合及 BG11 启动补充')
text=text.replace('<h2>确认的失效原因及修改</h2>', '<p>最后的启动补充只用于有先前可信形状、当前可见锚点充分且空间留出误差较小的情况。对未触发补充规则的视频，经完整记录的分支一致性核查复用 RES-1.8 视频，画面版本标识保持 RES-1.8；这部分不是另一次全帧计算。详见 <a href="trace_invariance.json">完整轨迹分支审计</a>。触发修正的来源按新版本重新计算、导出和逐帧解码校验。</p><h2>确认的失效原因及修改</h2>')
text=text.replace('</li></ol>','</li><li>232941 截断片段：活动绳近边后，剔除候选点比例约 40%，刚超过旧 35% 门槛。BG11 仅在已有可信形状、至少 120 个当前锚点、足够空间分布、凸性、低残差、低空间留出敏感性和形状一致性同时成立时，补充初始化。该片段边界从 0/9 提高至 6/9；被遮挡部分仍为推断，不据此确认根数或实体贴边。</li></ol>')
text=text.replace('fresh_replay_parity.json','../RopeEdgeSuite_R18_20261110/fresh_replay_parity.json')
text=text.replace('24 帧核验</a>', '24 帧核验</a>及 <a href="revision_fresh_parity.json">启动修正直接运行核验</a>')
(ROOT/'build_results.py').write_text(text,encoding='utf8')
