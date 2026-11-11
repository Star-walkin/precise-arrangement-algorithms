from common20 import *

def main():
    out=[]
    for item in plan():
        id=item['id'];prev=None;changes=[]
        for f in sorted((ROOT/'geometry_v5'/id/'frames').glob('*.json.gz')):
            with gzip.open(f,'rt',encoding='utf8') as h:r=json.load(h)
            axis=r['axis'];i=r['frame']
            if axis is None:prev=None;continue
            d=2500 if item['camera']=='right' else 2100;a,b,c=axis['line_abc'];y=-(a*d+c)/b
            if prev and i==prev[0]+1 and abs(y-prev[1])>20:
                changes.append(dict(frame=i,dy_px=float(y-prev[1]),before_source=prev[2],after_source=axis['source'],d_diagnostic=d))
            prev=(i,y,axis['source'])
        changes.sort(key=lambda x:abs(x['dy_px']),reverse=True);out.append(dict(id=id,large_current_axis_steps=changes));print(id,changes[:5],flush=True)
    dump(ROOT/'axis_jump_audit.json',dict(videos=out,partial=not (ROOT/'geometry_v5_receipt.json').exists(),threshold_px=20.,not_independent_ground_truth=True))

if __name__=='__main__':main()
