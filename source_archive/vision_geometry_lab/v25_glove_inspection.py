"""Posthoc same-pixel viewport for the completed, immutable glove challenge.

This crop is chosen from the saved seed origin and BOTH output endpoints solely
for human inspection. It never affects inference, scores, or a visibility mask.
"""
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .v25_glove_challenge import BASE, ARMS, read, sha, write, verify


def run(base=BASE):
    p = verify(base); folder = base/'qualitative_challenge'
    out = folder/'endpoint_inspection.png'
    if out.exists() or (folder/'endpoint_inspection.json').exists():
        raise FileExistsError('Do not overwrite posthoc inspection')
    r = read(folder/'results.json')
    if r['protocol_sha256'] != sha(folder/'protocol.json') or r['source_png_sha256'] != sha(folder/'source.png'):
        raise ValueError('Prediction/source binding differs')
    points = [p['source']['past_seed_xy'][-1]]
    points.extend(v['summary']['final_xy'] for v in r['results'].values() if v['summary']['final_xy'] is not None)
    xy = np.asarray(points)
    with Image.open(folder/'source.png') as image:
        source = image.convert('RGB')
    box = [max(0, int(np.floor(xy[:, 0].min()))-200), max(0, int(np.floor(xy[:, 1].min()))-200),
           min(source.width, int(np.ceil(xy[:, 0].max()))+200), min(source.height, int(np.ceil(xy[:, 1].max()))+200)]
    width, height = box[2]-box[0], box[3]-box[1]
    canvas = Image.new('RGB', (3*(width+20), height+130), '#14202d')
    draw = ImageDraw.Draw(canvas); font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 24)
    draw.text((16, 10), '事后终点复核：三栏为完全相同的原图像素窗口（1:1），不是模型输入ROI。', font=font, fill='white')
    for column, arm in enumerate(('source', *ARMS)):
        patch = source.crop(box); paint = ImageDraw.Draw(patch)
        if arm != 'source':
            segments = r['results'][arm]['rollout']['segments']
            for segment in segments:
                if len(segment) >= 2:
                    paint.line([(float(x)-box[0], float(y)-box[1]) for x, y in segment], fill='#ff36de', width=3)
            end = r['results'][arm]['summary']['final_xy']
            if end:
                x, y = end[0]-box[0], end[1]-box[1]
                paint.ellipse((x-7, y-7, x+7, y+7), outline='#00ffff', width=2)
        x = column*(width+20)
        draw.text((x+8, 51), '原始像素（无标记）' if arm == 'source' else arm+'：品红候选 / 青圈终点', font=font, fill='white')
        canvas.paste(patch, (x, 88))
    canvas.save(out)
    write(folder/'endpoint_inspection.json', dict(schema='v25_glove_posthoc_view_1',
        script_sha256=sha(__file__), protocol_sha256=sha(folder/'protocol.json'),
        results_sha256=sha(folder/'results.json'), source_sha256=sha(folder/'source.png'),
        bbox_xyxy=box, magnification=1, same_source_pixel_crop_for_all=True,
        selection='posthoc bounding box of seed origin and both predicted final points, expanded 200 original pixels',
        no_quantitative_occlusion_gt=True, used_for_inference=False, image_sha256=sha(out)))
    print(json.dumps(dict(path=str(out), bbox=box)))


if __name__ == '__main__':
    run()
