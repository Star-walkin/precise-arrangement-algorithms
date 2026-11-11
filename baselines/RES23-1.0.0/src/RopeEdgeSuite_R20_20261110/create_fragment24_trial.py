from common20 import *
text=(ROOT/'probe_fragments23.py').read_text(encoding='utf8')
text=text.replace('fragments23','fragments24').replace('region_fragments23','region_fragments24')
text=text.replace("text=text.replace(\"version='BG5_current_pixel_region_gate_v2'\",\"version='BG5_current_fragments_R20_23'\")", "text=text.replace(\"version='BG5_current_pixel_region_gate_v2'\",\"version='BG5_current_fragments_R20_24'\")\n    text=text.replace('ytol=max(.75*stats[main,cv2.CC_STAT_HEIGHT]', 'ytol=max(1.5*stats[main,cv2.CC_STAT_HEIGHT]')")
text=text.replace("306))","41))")
(ROOT/'probe_fragments24.py').write_text(text,encoding='utf8')
batch=(ROOT/'correct_fragments23.py').read_text(encoding='utf8').replace('fragments23','fragments24').replace('current_boundary24.py','current_boundary23.py')
(ROOT/'correct_fragments24.py').write_text(batch,encoding='utf8')
