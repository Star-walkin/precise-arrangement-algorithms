"""Compare cached BG1 raw and fitted evidence; do not change detections/counts."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from inspect_regions import SOURCE

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'full_002'
INDEX = 240
record = json.loads((RUN / 'frames' / f'{INDEX:06d}.json').read_text(encoding='utf8'))
cap = cv2.VideoCapture(str(SOURCE))
cap.set(cv2.CAP_PROP_POS_FRAMES, INDEX)
ok, original = cap.read()
cap.release()
assert ok
assert hashlib.sha256(original.tobytes()).hexdigest() == record['source_pixel_sha256']
mask = cv2.imdecode(np.fromfile(RUN / record['masks']['surface']['path'], np.uint8), 0) > 0
raw = original.copy()
fitted = original.copy()
raw_raster = np.zeros(original.shape[:2], np.uint8)
for line in record['F3']['raw_observed_polylines_xy']:
    cv2.polylines(raw_raster, [np.rint(line).astype(np.int32)], False, 255, 2)
raw[(raw_raster > 0) & mask] = (40, 245, 40)
for inferred in (False, True):
    raster = np.zeros(original.shape[:2], np.uint8)
    for curve in record['F3']['fitted_curves']:
        xy = np.rint(curve['xy']).astype(np.int32)
        states = curve['point_states']
        for j in range(len(xy) - 1):
            is_inferred = states[j] == 'inferred' or states[j + 1] == 'inferred'
            if is_inferred == inferred:
                cv2.line(raster, tuple(xy[j]), tuple(xy[j + 1]), 255, 2)
    fitted[(raster > 0) & mask] = (20, 160, 255) if inferred else (40, 245, 40)

x0, y0, x1, y1 = 1750, 2260, 2470, 2780
width, height = x1 - x0, y1 - y0
canvas = Image.new('RGB', (width * 3 + 40, height + 200), '#f1f4f8')
draw = ImageDraw.Draw(canvas)
font_path = 'C:/Windows/Fonts/msyh.ttc'
font = ImageFont.truetype(font_path, 24)
small = ImageFont.truetype(font_path, 21)
draw.text((20, 12), '同一帧核查：现有视频画的是原始短线段，已保存的局部拟合曲线尚未接入显示和计数', font=font, fill='#172435')
titles = ['原图局部｜第 240 帧（24.0 秒）', '当前视频来源｜原始观测短线段', '已缓存但未使用｜通过验证的局部拟合']
for i, (a, title) in enumerate(zip((original, raw, fitted), titles)):
    b = a.copy()
    cv2.line(b, (2239, y0), (2239, y1 - 1), (0, 255, 255), 2)
    panel = Image.fromarray(cv2.cvtColor(b[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    left = 10 + i * (width + 10)
    canvas.paste(panel, (left, 92))
    draw.text((left, 55), title, font=small, fill='#172435')
draw.text((20, height + 110), '绿：当前图像支持的线段/拟合路径；橙：短缺口推断；黄：固定计数列 x = 2238.7。三图为相同原图坐标裁切。', font=small, fill='#172435')
draw.text((20, height + 144), '全帧统计：307 条原始短线段、133 条局部拟合片段。片段数不是钢丝绳根数；此图未进行跨轨迹重关联或重新计数。', font=small, fill='#172435')
out = ROOT / 'delivery' / 'BG1_raw_vs_cached_fit_frame240.png'
canvas.save(out)
print(out)
