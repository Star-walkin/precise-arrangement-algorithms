from collections import Counter
from common import *
def main():
    out=[]
    for item in read(ROOT/'plan.json')['videos']:
        run=ROOT/'runs'/item['id'];bad=[]
        for p in sorted((run/'frames').glob('*.json.gz')):
            i=int(p.name.split('.')[0]);r=record(run,i);s=r['track']['diagnostics']['SG4'];c=s.get('fallback_check')
            if c and not c['valid']:bad.append(dict(frame=i,reasons=c['reasons'],metrics={k:v for k,v in c.items() if k!='witness'}))
        if bad:
            out.append(dict(id=item['id'],failures=bad))
            print(item['id'],len(bad),dict(Counter(w for r in bad for w in r['reasons'])),[r['frame'] for r in bad][:30],flush=True)
    dump(ROOT/'fallback_rejections_partial.json',out)
if __name__=='__main__':main()
