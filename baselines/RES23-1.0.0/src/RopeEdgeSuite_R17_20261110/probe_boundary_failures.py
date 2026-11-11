from common import *
def main():
    id='dual_20260923_113305__right';out=[]
    for i in [0,1,3,5,12,20,23,24,25,40,55,70]:
        r=record(ROOT/'runs'/id,i);b=r['boundary']
        out.append(dict(frame=i,boundary=b,camera=r['camera'],proposal_valid=r['proposal']['valid']))
        print(i,{k:v for k,v in b.items() if not isinstance(v,list)},'reasons',b['reasons'],flush=True)
    dump(ROOT/'boundary_initialization_probe.json',out)
if __name__=='__main__':main()
