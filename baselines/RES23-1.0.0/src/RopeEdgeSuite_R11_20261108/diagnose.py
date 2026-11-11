"""Read-only diagnostic of frozen RES-1.0 region evidence."""
from pathlib import Path
import json
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'RopeEdgeSuite_R1_20261107'

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def main():
    allrows={}
    for session in ['dual_20260923_113745','dual_20260924_111143']:
        rows=[]
        for p in sorted((OLD/'transfer_runs'/session/'bg1/frames').glob('*.json')):
            r=read(p);g=r['region'];o=g.get('outer_rim',{});xy=np.array(o.get('points_xy',[]),float).reshape(-1,2)
            loc=xy[abs(xy[:,0]-2238.7)<3*23.2941]
            row=dict(frame=r['frame_index'],valid=g['valid'],reasons=g['reasons'],S1=r['s1']['valid'],
                edge=g.get('edge_y'),local_fraction=g.get('local_support_fraction'),mad=g.get('local_residual_MAD_px'),
                top_fraction=g.get('top_support_fraction'),coverage=g.get('count_neighborhood_surface_coverage'),
                outer_n=len(xy),outer_span=float(np.ptp(xy[:,0])) if len(xy) else 0,outer_local_n=len(loc),
                outer_local_bracket=bool(len(loc) and loc[:,0].min()<2238.7-23.2941 and loc[:,0].max()>2238.7+23.2941))
            rows.append(row)
        combos=Counter('|'.join(r['reasons']) or 'valid' for r in rows)
        summary=dict(frames=len(rows),reason_combinations=dict(combos),outer_only=sum(r['reasons']==['outer_flange_not_confirmed'] for r in rows),
                     active_and_region=sum(r['S1'] and r['valid'] for r in rows),outer_only_with_active=sum(r['S1'] and r['reasons']==['outer_flange_not_confirmed'] for r in rows))
        print(session,json.dumps(summary,ensure_ascii=False),flush=True)
        print('outer-only quantiles', {key:np.percentile([r[key] for r in rows if r['reasons']==['outer_flange_not_confirmed']],[0,10,50,90,100]).tolist() for key in ['outer_n','outer_span','outer_local_n']})
        allrows[session]=dict(summary=summary,rows=rows)
    (ROOT/'baseline_diagnosis.json').write_text(json.dumps(allrows,ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__':main()
