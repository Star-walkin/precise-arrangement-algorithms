from pathlib import Path
import sys,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from smoke_current import DEV,save_image
out=ROOT/"p1_runtime"/"entry_band_comparison"
for name,session,camera,video in DEV:
 cap=cv2.VideoCapture(video);ok,bgr=cap.read();cap.release();assert ok
 baseline=json.loads((ROOT/"p1_runtime"/"smoke_current_frame"/(name+"__f0.json")).read_text(encoding="utf-8"))
 band=json.loads((out/(name+"__f0_entry_band.json")).read_text(encoding="utf-8"))
 s1=json.loads((out/(name+"__f0_S1_current_A7_seed.json")).read_text(encoding="utf-8"))
 overlay=bgr.copy()
 for segment in s1["supported_segments_xy"]:
  q=np.rint(segment).astype(np.int32)
  if len(q)>1:cv2.polylines(overlay,[q],False,(255,150,0),7,cv2.LINE_AA)
 for record,color,dashed in [(baseline,(230,70,230),True),(band,(60,235,40),False)]:
  q=np.rint(record["path_xy"]).astype(np.int32)
  for i in range(1,len(q)):
   supported=record["diagnostics"]["path_segment_support"][i-1]
   if dashed:
    if i%5<2:cv2.line(overlay,tuple(q[i-1]),tuple(q[i]),color,3,cv2.LINE_AA)
   elif supported:cv2.line(overlay,tuple(q[i-1]),tuple(q[i]),color,3,cv2.LINE_AA)
   elif i%5<2:cv2.line(overlay,tuple(q[i-1]),tuple(q[i]),(0,140,255),3,cv2.LINE_AA)
 cv2.putText(overlay,f"{name} f0 full purple dash / band supported green / band weak orange dash / S1 blue",(40,80),cv2.FONT_HERSHEY_SIMPLEX,1.15,(255,255,255),3,cv2.LINE_AA)
 save_image(out/(name+"__f0_full_band_S1_overlay.jpg"),overlay)
print("Re-rendered actual-supported overlays from unchanged JSON; no inference rerun")

