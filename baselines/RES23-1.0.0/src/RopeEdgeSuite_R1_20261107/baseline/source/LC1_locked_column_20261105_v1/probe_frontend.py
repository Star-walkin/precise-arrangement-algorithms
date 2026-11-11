from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
from lc1_frontend import TraditionalFrontend,SOURCE

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
VIDEOS=Path(r'E:\study\program\济宁\双相机同步')
OUT=ROOT/'front_probe'; OUT.mkdir(exist_ok=True)
summary=[]
for session,camera,frames in [('dual_20260924_111500','left',[0,120,300]),('dual_20260923_113305','right',[0,240,360])]:
    mp=BASE/'camera_condition_sam2_20261009_v1/all_001/body_branches_002/masks'/f'{camera}_camera'/f'{session}__pair0015__{camera}.png'
    mask=cv2.imdecode(np.fromfile(mp,np.uint8),0)
    pp=BASE/'camera_condition_bright_rope_boundaries_20261023_v2/all_002/boundaries/records'/f'{camera}_camera'/f'{session}__pair0015__{camera}.json'
    pitch=float(json.loads(pp.read_text(encoding='utf-8'))['record']['pitch_px'])
    # No historical line, path, or target coordinates from that record are passed.
    flipped=camera=='left'
    if flipped:mask=cv2.flip(mask,1)
    front=TraditionalFrontend(mask,pitch_px=pitch,free_side='right')
    serial='DA9627065' if camera=='left' else 'DA9626975'
    cap=cv2.VideoCapture(str(VIDEOS/session/f'{camera}_{serial}.avi'))
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read();assert ok
        if flipped:im=cv2.flip(im,1)
        result=front.step(im,frame/10)
        result['probe_input']={'session':session,'camera':camera,'frame':frame,'horizontal_flip_applied':flipped,'static_surface_mask':str(mp),'pitch_record_only':str(pp)}
        stem=f'{session}_{camera}_f{frame:04d}'
        (OUT/f'{stem}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        overlay=im.copy()
        cont,_=cv2.findContours((mask>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,cont,-1,(0,130,0),2)
        pts=np.asarray(result['line_points_xy'],np.float32)
        for x,y in pts[::3]:cv2.circle(overlay,(round(float(x)),round(float(y))),4,(0,255,0),-1)
        path=np.asarray(result['path_xy'],np.float32)
        sup=result['diagnostics'].get('path_point_support',[True]*len(path))
        segsup=result['diagnostics'].get('path_segment_support',[True]*max(0,len(path)-1))
        for j in range(1,len(path)):
            color=(255,180,0) if sup[j] else (0,0,255)
            if not segsup[j-1]:
                if j%2:continue
                color=(0,0,255)
            cv2.line(overlay,tuple(np.round(path[j-1]).astype(int)),tuple(np.round(path[j]).astype(int)),color,3)
        text=f'LC1-T ONLY | f{frame} | whole_candidate={result["whole_path_candidate_supported"]}'
        cv2.putText(overlay,text,(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(20,255,255),4)
        cv2.imencode('.jpg',overlay,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(OUT/f'{stem}_overlay.jpg')
        yy,xx=np.where(mask>0)
        crop=overlay[max(0,yy.min()-400):min(overlay.shape[0],yy.min()+850),max(0,xx.min()-120):]
        cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95])[1].tofile(OUT/f'{stem}_crop.jpg')
        d=result['diagnostics']
        summary.append({'session':session,'camera':camera,'frame':frame,'line_abc':result['line_abc'],'whole_path_candidate_supported':result['whole_path_candidate_supported'],**{k:d.get(k) for k in ['status','line_candidate_count','free_line_supported','free_line_candidate_disagreement','trace_returned_points','trace_supported_fraction','trace_inside_span_px','trace_bend_slope_change','trace_candidate_disagreement','trace_distinct_endpoint_cost_margin_per_step','trace_truncated_for_missing_current_support']}})
    cap.release()
(OUT/'probe_summary.json').write_text(json.dumps({'source':SOURCE,'cases':summary},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
