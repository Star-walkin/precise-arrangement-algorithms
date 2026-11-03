"""Fixed predeclared gallery QA views and minimum training-side verification pack."""
from common import *
import numpy as np,gzip,shutil
from PIL import Image,ImageDraw,ImageFont
Image.MAX_IMAGE_PIXELS=150_000_000
def font(n):return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
def readgz(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def main():
    protocol=load(HERE/'protocol.json');dest=HERE/'visual_review';dest.mkdir(exist_ok=True);rows=[]
    ids=protocol['controlled_h6_ids'];lookup={j['id']:j for j in protocol['jobs']}
    for ident in ids:
        p=HERE/'交付_292张原图与热力图'/(ident+'.png');im=Image.open(p).convert('RGB');im.thumbnail((1800,1000))
        q=dest/(ident+'.png');im.save(q);rows.append(dict(id=ident,source=str(p),preview=str(q),qa_preview_only=True))
    # Native pixel detail pairs for two user failures; no resizing.
    crops={'L_run_20260923_01__f000585__left':[900,2150,2100,2800],
        'dual_20260923_101643__pair0255__right':[1100,2050,2300,2700],
        'dual_20260923_101517__pair0345__right':[1200,1900,2400,2550]}
    for ident,box in crops.items():
        with Image.open(HERE/'交付_292张原图与热力图'/(ident+'.png')) as full:
            for name,xoffset in [('原图',0),('热图与R1',4128)]:
                b=[box[0]+xoffset,box[1]+160,box[2]+xoffset,box[3]+160]
                full.crop(b).save(dest/(ident+'__native_'+name+'.png'))
    pack=HERE/'最小接续核验包';pack.mkdir(exist_ok=True);items=[];index=load(PAIR/'data/index.json')
    for ident in ['dual_20260924_111500__left__f00240','dual_20260924_111143__right__f00130']:
        item=next(i for i in index['items'] if i['id']==ident);s=load(PAIR.parent/'run_20261029_001/data/label_sources'/(ident+'.json'));weak=load(item['supervision_annotation'])
        raw=Image.open(item['image_path']).convert('RGB');overlay=raw.copy();draw=ImageDraw.Draw(overlay)
        for line in s['trusted_center_segments_xy']:draw.line([tuple(p) for p in line],fill=(40,220,50),width=3)
        neg=weak.get('neighbor');
        if neg:draw.line([tuple(p) for p in neg['points_xy']],fill=(255,60,60),width=4)
        r=readgz(HERE/'fixed_001/R1/fixed/CONDITIONAL'/(ident+'.json.gz'))['result']
        for c in r['candidates'][:3]:draw.line([tuple(p) for p in c['points_xy']],fill=(30,200,255),width=2)
        for line in s['trusted_center_segments_xy']:
            x,y=line[-1];draw.ellipse((x-10,y-10,x+10,y+10),outline='yellow',width=4)
        board=Image.new('RGB',(4096,3180),'white');board.paste(overlay,(0,180));d=ImageDraw.Draw(board)
        d.text((25,12),ident+'  训练侧最小接续核验包',font=font(40),fill='black')
        d.text((25,68),'绿：历史reviewed部分中心；红：Codex历史弱邻绳；蓝：R1预测，均不补为人工真值。',font=font(33),fill='black')
        d.text((25,118),'黄圈：既有可信范围终点，非真实绳端；请只核实可见的弯后正目标短段和同处竞争邻绳。',font=font(31),fill='black')
        q=pack/(ident+'__核验上下文.png');board.save(q);shutil.copyfile(item['image_path'],pack/(ident+'__原图.png'))
        items.append(dict(id=ident,context=str(q),source=item['image_path'],source_sha256=sha(item['image_path']),
            reviewed_segments=s['trusted_center_segments_xy'],engineering_weak_neighbor=neg,
            requested='Confirm a short target fragment after the known bend, the competing incorrect neighbor, and a far anchor only if directly visible.',
            new_independent_human_confirmation=False,not_used_as_new_training_truth=True))
    dump(pack/'核验说明.json',dict(items=items,new_label_count=0,originals_unchanged=True))
    dump(dest/'selection.json',dict(gallery_fixed_ids=ids,rows=rows,native_crops=crops,selection_before_new_results=True))
    print('Prepared fixed26 QA previews and2 native verification packs')
if __name__=='__main__':main()
