from common22 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    allstats=read(ROOT/'inspection/state_trial_B.json')
    for item in plan():
        rows=read(ROOT/'inputs'/item['id']/'frames.json');ev=read(ROOT/'events'/f"{item['id']}.json")
        if len(rows)<20:continue
        tt=[r['time_s'] for r in rows];nn=[r['N'] if r['available'] else float('nan') for r in rows]
        fig,ax=plt.subplots(3,1,figsize=(13,7),sharex=True)
        ax[0].plot(tt,nn,'.',ms=2,label='raw N');ax[0].plot(tt,[r['filtered'].get('N',float('nan')) for r in rows],lw=1,label='R21 filtered')
        ax[0].plot(tt,[r['motion_distance'] if r['motion_distance'] is not None else float('nan') for r in rows],lw=1,label='pixel gap / reference pitch')
        ax[0].legend();ax[0].set_ylabel('pitch distance')
        mapping={'unknown':0,'steady':1,'approaching':2,'near':3,'departing':4}
        ax[1].step(tt,[mapping[r['state']] for r in ev['frames']],where='post')
        ax[1].set_yticks(list(mapping.values()),list(mapping.keys()));ax[1].grid()
        ax[2].plot(tt,[r.get('trend',{}).get('slope',float('nan')) if r.get('trend') else float('nan') for r in ev['frames']])
        ax[2].axhline(0,color='k');ax[2].set_ylabel('robust speed');ax[2].set_xlabel('time / s')
        fig.suptitle(item['id']);fig.tight_layout();fig.savefig(ROOT/'inspection'/f"{item['id']}_state_B.png");plt.close(fig)
        s=next(x for x in allstats if x['id']==item['id'])
        print(item['id'],[(x['t'],x['state'],round(x['N'],2) if x['N'] is not None else None) for x in s['transitions']])
if __name__=='__main__':main()
