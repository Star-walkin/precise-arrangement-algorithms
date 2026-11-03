"""Three named pilot diagnostics for the approved lane-geometry revision."""
from pathlib import Path
import time,shutil
import cv2,numpy as np
import whole_io as io
from lane_candidates import generate_lane_candidates

def main():
    run=io.HERE/'lane_revision2_probe_001'
    if run.exists():raise ValueError('Never overwrite')
    run.mkdir();shutil.copy2(io.HERE/'lane_candidates.py',run/'lane_candidates_snapshot.py')
    names=['L_run_20260923_01__f000015__left','ruler_left_05','ruler_right_05'];rows=io.selection('all');results=[];cv2.setNumThreads(1)
    for name in names:
        row=next(r for r in rows if Path(r['output_file']).stem==name);rel=Path(row['output_file']);data=io.inputs(row)
        old=io.read_json(io.HERE/'pilot_001/记录'/rel.parent.name/(name+'.json'));t=time.time();result=generate_lane_candidates(data,old['free'])
        rec={'input':row['output_file'],'candidate_count':len(result['candidates']),'lane_count':len(result['lanes']),'seconds':time.time()-t,'diagnostics':result['diagnostics'],'source_sha256':io.sha(io.HERE/'lane_candidates.py')}
        results.append(rec);io.dump(run/(name+'.json'),{'record':rec,'generated':result})
        im=data['original'].copy();colors=[(255,70,80),(40,220,70),(210,70,220),(20,200,250),(250,200,50),(230,60,40)]
        for i,c in enumerate(result['candidates']):
            q=np.rint(c['xy']).astype(np.int32);color=colors[i%len(colors)];cv2.polylines(im,[q],False,color,2);cv2.drawMarker(im,tuple(q[c['join_index']]),color,cv2.MARKER_CROSS,14,2)
            cv2.putText(im,str(i),tuple(q[-1]),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
        coords=np.argwhere(data['surface']>0);y0=max(0,int(coords[:,0].min())-200);y1=min(im.shape[0],int(coords[:,0].max())+100)
        im=im[y0:y1];io.write_image(run/(name+'.jpg'),cv2.resize(im,(2048,int(im.shape[0]/2))))
        print(name,len(result['candidates']),len(result['lanes']),round(rec['seconds'],2),flush=True)
    io.dump(run/'summary.json',{'scope':'Three open pilot cases, same saved free geometry; formal-pair geometry only, no score/identity claim.','exit_code':0,'cases':results})
    return 0
if __name__=='__main__':raise SystemExit(main())
