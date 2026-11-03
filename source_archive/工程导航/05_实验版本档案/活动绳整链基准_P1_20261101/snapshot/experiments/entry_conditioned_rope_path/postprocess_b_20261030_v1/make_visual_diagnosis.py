"""Read-only visual crops of the user-specified B comparison, for review only."""
from pathlib import Path
import json, hashlib
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'batch_ab_20261030_001'
IDS=['L_run_20260923_01__f000015__left','L_run_20260923_01__f000142__left',
     'R_run_20260923_01__f000055__right','dual_20260923_101517__pair0075__right',
     'dual_20260923_101643__pair0375__right','ruler_left_05',
     'R_run_stop_run_20260922_01__f002521__right']

def main():
    meta=json.loads((SOURCE/'jobs.json').read_text(encoding='utf-8-sig'))
    jobs={j['id']:j for j in meta['jobs']}
    out=ROOT/'visual_diagnosis'; out.mkdir(exist_ok=True)
    receipt=[]
    Image.MAX_IMAGE_PIXELS=200_000_000
    for ident in IDS:
        j=jobs[ident]
        original=Image.open(j['image_path']).convert('RGB')
        comparison=Image.open(j['output']).convert('RGB')
        for k in range(len(j['seeds'])):
            raw=original.crop((0,1050,4096,3000))
            heat=comparison.crop((8192,k*3190+1175,12288,k*3190+3125))
            board=Image.new('RGB',(2048,2010),'white')
            board.paste(raw.resize((2048,975)),(0,30))
            board.paste(heat.resize((2048,975)),(0,1035))
            d=ImageDraw.Draw(board);d.text((10,7),ident+' ORIGINAL',fill='black')
            d.text((10,1012),'B center response, entry '+str(k+1),fill='black')
            dest=out/(ident+f'__entry{k}.jpg')
            assert not dest.exists()
            board.save(dest,quality=96,subsampling=0)
            receipt.append({'id':ident,'entry':k,'source_comparison':j['output'],
                            'comparison_sha256':hashlib.sha256(Path(j['output']).read_bytes()).hexdigest(),
                            'output':str(dest),'crop_original':[0,1050,4096,3000],
                            'comparison_header':125,'display_scale':.5})
    (out/'crop_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(len(receipt))

if __name__=='__main__': main()
