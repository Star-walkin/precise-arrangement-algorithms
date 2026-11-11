"""Audit only newly committed frames, then finalize the standard audit."""
import argparse,time,sys
import audit_bg5 as a

def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);args=p.parse_args()
    run=a.ROOT/'runs'/args.session;dest=run/'audit_rows.json'
    rows=a.read(dest) if dest.exists() else []
    if rows and rows[0].get('audit_check_version')!=3:rows=[]
    for i in range(len(rows),600):
        deadline=time.monotonic()+600
        while not (run/'frames'/f'{i:06d}.json.gz').exists():
            assert time.monotonic()<deadline;time.sleep(.25)
        row=a.analyze(run,i);rows.append(row)
        if row['failures']:print('AUDIT_FAILURE',i,row['failures'],flush=True)
        if len(rows)%50==0:
            a.dump(dest,rows);print('AUDITED',args.session,len(rows),flush=True)
    a.dump(dest,rows);deadline=time.monotonic()+60
    while not (run/'receipt.json').exists():
        assert time.monotonic()<deadline;time.sleep(.25)
    sys.argv=[__file__,'--session',args.session];a.main()

if __name__=='__main__':main()
