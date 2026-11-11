import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import concurrent.futures as cf
from common22 import *

def process(item):
    id=item['id'];rows=read(R21/'delivery'/id/'frames.json')
    for r in rows:
        full=gzread(r['measurement_record_path']);c=full['measurement'].get('central') or {};b=c.get('boundary') or {}
        if c.get('active_y') is not None and b.get('y') is not None:
            r['motion_gap_px']=c['active_y']-b['y']
            r['motion_distance']=r['motion_gap_px']/23.29412648
        else:r['motion_gap_px']=None;r['motion_distance']=None
    dump(ROOT/'inputs'/id/'frames.json',rows);print('prepared',id,flush=True)
    return dict(id=id,frames=len(rows))

if __name__=='__main__':
    with cf.ThreadPoolExecutor(4) as pool:results=list(pool.map(process,plan()))
    dump(ROOT/'input_receipt.json',results)
