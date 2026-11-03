from pathlib import Path
import sys
from PIL import Image,ImageDraw

root=Path(__file__).resolve().parent/sys.argv[1]
out=root/'qa_visual';out.mkdir(exist_ok=False)
sources=sorted((root/'分步图像/06_最终叠加').glob('*.jpg'))
for f in sources:
    crops=[]
    for folder in ['01_原图','05_路径候选','06_最终叠加']:
        im=Image.open(root/'分步图像'/folder/f.name).convert('RGB')
        crop=im.crop((0,800,2048,1500))
        board=Image.new('RGB',(2048,730),'white');board.paste(crop,(0,30))
        ImageDraw.Draw(board).text((8,7),folder[:2]+' '+f.stem,fill='black');crops.append(board)
    board=Image.new('RGB',(2048,2190));
    for k,crop in enumerate(crops):board.paste(crop,(0,k*730))
    board.save(out/f.name,quality=96,subsampling=0)
print(len(sources))
