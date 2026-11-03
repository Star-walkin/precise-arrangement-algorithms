"""Fixed16 geometry-only diagnostic using saved A4 free geometry, no new truth."""
import importlib.util,json,sys,time,shutil
from pathlib import Path
import cv2,numpy as np
from lane_candidates import generate_lane_candidates
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'camera_condition_whole_incoming_20261028_v4'
spec=importlib.util.spec_from_file_location('lane_probe_old_file_io',OLD/'whole_io.py');io=importlib.util.module_from_spec(spec);spec.loader.exec_module(io)

def main():
    run=HERE/'lane_probe_001'
    if run.exists():raise ValueError('No overwrite')
    run.mkdir();shutil.copy2(HERE/'lane_candidates.py',run/'lane_candidates_snapshot.py');cv2.setNumThreads(1)
    rows=io.selection('pilot');results=[];start=time.time()
    for row in rows:
        t=time.time();rel=Path(row['output_file']);data=io.inputs(row)
        old_path=OLD/'all_002/记录'/rel.parent.name/(rel.stem+'.json');saved=io.read_json(old_path)
        result=generate_lane_candidates(data,saved['free']);record={'input':row['output_file'],'seconds':time.time()-t,'status':result['status'],'candidates':len(result['candidates']),'lanes':len(result['lanes']),'diagnostics':result['diagnostics'],'free_geometry_source':str(old_path),'free_geometry_source_sha256':io.sha(old_path),'input_dependency_sha256':data['input_dependency_sha256']}
        results.append(record);io.dump(run/'records'/rel.parent.name/(rel.stem+'.json'),{'record':record,'generated':result})
        overlay=data['original'].copy();colors=[(40,230,100),(255,100,50),(50,200,255),(220,80,230),(200,220,30),(60,80,245)]
        for i,c in enumerate(result['candidates']):
            xy=np.rint(c['xy']).astype(np.int32);cv2.polylines(overlay,[xy],False,colors[i%len(colors)],3)
            q=xy[c['join_index']];cv2.drawMarker(overlay,tuple(q),colors[i%len(colors)],cv2.MARKER_CROSS,18,2)
            cv2.putText(overlay,str(i),tuple(xy[-1]),cv2.FONT_HERSHEY_SIMPLEX,.7,colors[i%len(colors)],2)
        cv2.rectangle(overlay,(12,12),(1540,75),(20,20,20),-1);cv2.putText(overlay,'GEOMETRY CANDIDATES ONLY / NOT ACTIVE-ROPE IDENTIFICATION',(25,55),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        io.write_image(run/'qa'/rel.parent.name/(rel.stem+'.jpg'),cv2.resize(overlay,(2048,1500)))
        print(len(results),len(rows),rel.name,len(result['candidates']),round(record['seconds'],2),flush=True)
    io.dump(run/'summary.json',{'scope':'Fixed16 pure geometry candidate diagnostic; saved A4 free geometry, no photometric scoring or same-material truth.','cases':results,'count':len(results),'seconds':time.time()-start,'source_sha256':io.sha(HERE/'lane_candidates.py'),'exit_code':0})
    return 0
if __name__=='__main__':sys.exit(main())
