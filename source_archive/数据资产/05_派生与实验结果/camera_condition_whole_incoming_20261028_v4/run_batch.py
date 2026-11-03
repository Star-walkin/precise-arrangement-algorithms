"""四个独立离线进程处理固定互斥分片；共用结果树，不重复计数。"""
import argparse,subprocess,sys,time,shutil
from collections import Counter
import whole_io as io
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--freeze-file',default='freeze_all.json');a=p.parse_args();run=io.HERE/a.run
 if run.exists():raise ValueError('New run required')
 run.mkdir();(run/'分片记录').mkdir();children=[];start=time.time()
 freeze_path=io.HERE/a.freeze_file
 frozen=io.read_json(freeze_path)
 for name,h in frozen['sources_sha256'].items():
  if io.sha(name)!=h:raise ValueError('Frozen source changed: '+name)
 io.dump(run/'batch_start.json',{'command':sys.argv,'freeze_sha256':io.sha(freeze_path),'freeze_source':str(freeze_path),'workers':a.workers,'input_count':len(io.selection('all'))})
 shutil.copy2(freeze_path,run/'freeze_all_snapshot.json');shutil.copy2(io.HERE/'run_batch.py',run/'run_batch_snapshot.py')
 for i in range(a.workers):
  log=(run/'分片记录'/f'part_{i:02d}.log').open('w',encoding='utf-8')
  cmd=[sys.executable,str(io.HERE/'run_whole.py'),'--mode','all','--run',a.run,'--shard',str(i),'--shards',str(a.workers)]
  child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));children.append((child,log,cmd))
 exits=[]
 for child,log,cmd in children:
  code=child.wait();log.close();exits.append({'command':cmd,'exit_code':code});print('shard complete',len(exits),code,flush=True)
 records=[];files=[];failures=[];starts=[]
 for i in range(a.workers):
  context=run/'分片记录'/f'part_{i:02d}';j=io.read_json(context/'summary.json');records+=j['records'];failures+=j['failures'];files+=io.read_json(context/'output_manifest.json')['files'];starts.append(io.read_json(context/'run_start.json'))
 assert all(s['sources_sha256']==starts[0]['sources_sha256'] for s in starts)
 expected={r['output_file'] for r in io.selection('all')};actual=[r['output_file'] for r in records]
 success=not failures and len(actual)==len(set(actual))==400 and set(actual)==expected and all(x['exit_code']==0 for x in exits)
 fullstart=dict(starts[0]);fullstart.update({'command':sys.argv,'inputs':[r['output_file'] for r in io.selection('all')],'shard_commands':exits})
 io.dump(run/'run_start.json',fullstart);shutil.copytree(run/'分片记录/part_00/source_snapshot',run/'source_snapshot')
 io.dump(run/'output_manifest.json',{'version':io.VERSION,'files':files,'case_count':len(records)})
 io.dump(run/'summary.json',{'version':io.VERSION,'completed':len(records),'failures':failures,'statuses':dict(Counter(r['status'] for r in records)),'records':records,'elapsed_seconds':time.time()-start,'exit_code':0 if success else 1,'shards':exits})
 print('complete',len(records),'failed',len(failures),'seconds',round(time.time()-start),flush=True);return 0 if success else 1
if __name__=='__main__':sys.exit(main())
