import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'RopeEdgeSuite_R15_20261109'
shutil.copy2(OLD/'compare_results.py',ROOT/'compare_results.py')
text=(OLD/'build_results.py').read_text(encoding='utf8')
text=text.replace("runs_name='runs_bg9'","runs_name='runs'").replace('comparison_bg9.json','comparison.json')
text=text.replace('RES-1.5','RES-1.6').replace('SG3 活动绳分段跟踪','SG4 双分支活动绳跟踪').replace('SG3-BG9-CG2','SG4-BG9-CG2')
text=text.replace('SG3 / CG2 输出','SG2 / SG3 / CG2 输出').replace('重新按时间顺序拟合边界','重新按时间顺序进行 SG4 当前进绳核验、分支选择和边界拟合')
text=text.replace('frozen_final_BG9/hashes.json','frozen/hashes.json').replace('experiments/camera_test.json','../RopeEdgeSuite_R15_20261109/experiments/camera_test.json')
text=text.replace('人工查看范围另列于','助手画面复核范围另列于')
text=text.replace('</li></ol>','</li><li>113745 反光漂移：中间版 SG3 在 39.7–40.0 秒把黄色挡板反光当成了活动绳，此版已拒绝交付。SG4 优先保留严格 S1 当前检测；严格分支失败时，分段候选须通过固定进绳位置的当前双边缘、方向、宽度和近期严格身份核验。两分支均无当前证据时保持未知，不使用旧直线冒充当前检测。</li></ol>')
text=text.replace('<p>源文件限制：','<p>仍需区分的情况：活动绳尚未入画时，未知是正确拒绝；工具长时间遮挡时，只能输出有标记的模型推断。部分视频引导器前后的真实绳段不完全共线，识别出远端绳段并不证明可将其直接外推用于卷面计数。本版不承诺每帧都正确，不把内部通过率当作准确率。</p><p>源文件限制：')
text=text.replace('数字对照见 <a','直接运行与缓存复算一致性见 <a href="fresh_replay_parity.json">24 帧核验</a>；反光反例测试见 <a href="sg4_regression_test.json">13 项定向回归</a>；数字对照见 <a')
(ROOT/'build_results.py').write_text(text,encoding='utf8')
