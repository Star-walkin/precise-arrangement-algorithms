"""One declared four-image posthoc structural regression, using the same pipeline.

This wrapper does not alter detector parameters or algorithms. It preserves
source/input hashes and renders original/past/new comparisons after prediction.
"""
import sys,time,traceback,platform,shutil
from collections import Counter
import cv2,numpy as np
import whole_io as io
from free_rope import refine_and_extend,dependencies
from boundary_occlusion import analyze_boundary_occlusion
from whole_path import trace_whole,build_search_surface
from run_whole import compact,render

def main():
    run=io.HERE/'repair_checks_001'
    if run.exists():raise ValueError('Preserve previous run; new directory required')
    addendum=io.read_json(io.HERE/'protocol_addendum_002.json')
    wanted=addendum['posthoc_cases'];available={r['output_file']:r for r in io.selection('all')}
    if len(wanted)!=4 or len(set(wanted))!=4:raise ValueError('Expected four distinct declared cases')
    rows=[available[name] for name in wanted]
    source_names=['whole_io.py','free_rope.py','boundary_occlusion.py','whole_path.py','run_whole.py',
                  'protocol.json','protocol_addendum_001.json','protocol_addendum_002.json','run_repair_checks.py']
    sources={str(io.HERE/name):io.sha(io.HERE/name) for name in source_names}
    sources.update({str(p):io.sha(p) for p in dependencies()})
    run.mkdir();snapshot=run/'source_snapshot';snapshot.mkdir()
    snapshot_index=[]
    for i,(path,digest) in enumerate(sources.items()):
        dest=snapshot/f'{i:02d}_{io.Path(path).name}'
        shutil.copy2(path,dest)
        if io.sha(dest)!=digest:raise ValueError('Snapshot changed')
        snapshot_index.append({'source':path,'snapshot':str(dest.relative_to(run)),'sha256':digest})
    io.dump(run/'run_start.json',{'version':io.VERSION,'command':sys.argv,'sources_sha256':sources,
      'source_snapshot_index':snapshot_index,'inputs':wanted,'input_scope':'explicit same400 open-development images',
      'selection_kind':'posthoc regression of known all001 structural failure; not blind test',
      'addendum_sha256':io.sha(io.HERE/'protocol_addendum_002.json'),'python':sys.version,
      'executable':sys.executable,'numpy':np.__version__,'opencv':cv2.__version__,'platform':platform.platform(),
      'equipment_connected':False,'reserved_data_access':False})
    old=io.HERE/'all_001';selection=io.read_json(old/'qa_postrun/difference_and_selection.json')
    crop_by_input={v['input']:v['native_crop_xyxy'] for v in selection['selected']}
    results=[];failures=[];files=[];start=time.time();cv2.setNumThreads(1)
    for index,row in enumerate(rows,1):
        tick=time.time();rel=io.Path(row['output_file']);role=rel.parent.name;stem=rel.stem
        try:
            data=io.inputs(row)
            free=refine_and_extend(data)
            search_surface,search_info=build_search_surface(data['surface'])
            occlusion=analyze_boundary_occlusion(data['original'],data['surface'],data['merged'],data['inferred'],
                data['curve_refs'],free['geometry'],data['exclusion'],data['pitch'],search_surface=search_surface)
            occlusion['diagnostics']['search_domain_construction']=search_info
            whole=trace_whole(data,free,occlusion)
            overlay=render(data['original'],whole['whole_mask'],whole['supported_mask'],whole)
            paths=[run/'01_整根候选掩膜'/role/rel.name,run/'02_原图叠加'/role/rel.name,
                   run/'03_实测支持掩膜'/role/rel.name,run/'04_接续待确认'/role/rel.name,
                   run/'记录'/role/(stem+'.json')]
            for p,a in zip(paths,[whole['whole_mask'],overlay,whole['supported_mask'],whole['review_mask']]):io.write_image(p,a)
            rec={'output_file':row['output_file'],'camera_role':row['camera_role'],'source_group':row['source_group'],
                 'source_time_s':row['source_time_s'],'dataset_use_status':row['dataset_use_status'],
                 'status':whole['status'],'elapsed_seconds':time.time()-tick,'input_dependency_sha256':data['input_dependency_sha256']}
            io.dump(paths[-1],{'record':rec,'free':compact(free),'occlusion':compact(occlusion),'whole':compact(whole)})
            # The shared native window was fixed by all001's posthoc diagnostics.
            x0,y0,x1,y1=crop_by_input[row['output_file']]
            prior=io.read_image(old/'02_原图叠加'/role/rel.name)
            panels=[]
            for title,img in [('Original',data['original']),('A4.0.0 - preserved all001',prior),('A4.0.1 - declared structural repair',overlay)]:
                crop=img[y0:y1,x0:x1].copy();head=np.full((34,crop.shape[1],3),15,np.uint8)
                cv2.putText(head,title,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
                panels.append(np.vstack([head,crop]))
            compare_path=run/'comparison'/role/(stem+'.jpg');io.write_image(compare_path,np.vstack(panels))
            points=np.argwhere((whole['whole_mask']>0)|(data['surface']>0))
            y0=max(0,int(points[:,0].min())-100) if len(points) else 1400
            x0=max(0,int(points[:,1].min())-100) if len(points) else 500
            y1=min(data['surface'].shape[0],int(points[:,0].max())+100) if len(points) else 3000
            x1=min(data['surface'].shape[1],int(points[:,1].max())+100) if len(points) else 3500
            qa_path=run/'qa'/role/(stem+'.jpg')
            io.write_image(qa_path,np.vstack([data['original'][y0:y1,x0:x1],overlay[y0:y1,x0:x1]]))
            files.extend([{'path':str(p.relative_to(run)),'size':p.stat().st_size,'sha256':io.sha(p)} for p in paths])
            results.append(rec)
            print(index,4,row['output_file'],whole['status'],round(time.time()-tick,2),flush=True)
        except Exception as e:
            failure={'output_file':row['output_file'],'error':str(e),'traceback':traceback.format_exc()};failures.append(failure)
            print(failure['traceback'],flush=True)
        io.dump(run/'progress.json',{'completed':len(results),'failed':len(failures),'total':4})
    unchanged={name:io.sha(name)==digest for name,digest in sources.items()}
    code=0 if not failures and len(results)==4 and all(unchanged.values()) else 1
    io.dump(run/'output_manifest.json',{'version':io.VERSION,'case_count':len(results),'files':files})
    io.dump(run/'summary.json',{'version':io.VERSION,'completed':len(results),'failures':failures,
       'statuses':dict(Counter(v['status'] for v in results)),'records':results,
       'elapsed_seconds':time.time()-start,'source_unchanged':all(unchanged.values()),'source_checks':unchanged,
       'exit_code':code,'evaluation_kind':'posthoc known-bug regression, not independent accuracy'})
    return code

if __name__=='__main__':sys.exit(main())
