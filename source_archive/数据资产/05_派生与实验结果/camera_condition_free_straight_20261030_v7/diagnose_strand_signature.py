"""固定旧AUTO入口的股纹机制诊断；不是训练/真值，不运行完整检测器。"""
from pathlib import Path
import sys,json,hashlib,datetime,csv
import cv2,numpy as np
from strand_signature import measure_strand_signature

HERE=Path(__file__).resolve().parent;ASSETS=HERE.parent
FILES=['right_camera/dual_20260923_101517__pair0075__right','right_camera/dual_20260923_101643__pair0375__right','left_camera/L_run_20260923_01__f000015__left','left_camera/ruler_left_05','right_camera/ruler_right_05']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 folder=HERE/sys.argv[1]
 if folder.exists():raise ValueError('Preserve previous diagnosis')
 folder.mkdir();data=ASSETS/'camera_condition_images_20261009_v2';old=ASSETS/'camera_condition_incoming_template_20261029_v6/all_001'
 rows={r['output_file']:r for r in csv.DictReader((data/'manifest.csv').open(encoding='utf-8-sig'))}
 manifest=json.loads((old/'output_manifest.json').read_text(encoding='utf-8'))
 hashes={str(Path(v['path'])):v['sha256'] for v in manifest['files']}
 result=[]
 for stem in FILES:
  rel='images/'+stem+'.png';row=rows[rel];p=data/rel
  assert sha(p)==row['output_sha256'] and '20260925' not in rel
  record=old/'记录'/(stem+'.json');assert sha(record)==hashes[str(record.relative_to(old))]
  im=cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_COLOR);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  j=json.loads(record.read_text(encoding='utf-8'))['result'];case={'output_file':rel,'image_sha256':sha(p),'A6_record_sha256':sha(record),'entries':[]}
  for entry in j.get('entries',[]):
   free=entry['free']['free_segment'];segment=np.asarray(free['observed_segment_xy']);width=float(free['width_px'])
   signature=measure_strand_signature(gray,segment,width)
   case['entries'].append({'entry_id':entry['entry_id'],'automatic_segment_xy':segment.tolist(),'width_px':width,'signature':signature})
   pad=int(max(80,3*width));a=np.floor(segment.min(axis=0)-pad).astype(int);b=np.ceil(segment.max(axis=0)+pad).astype(int);a=np.maximum(a,0);b=np.minimum(b,[im.shape[1],im.shape[0]])
   crop=im[a[1]:b[1],a[0]:b[0]].copy();q=np.rint(segment-a).astype(np.int32);cv2.line(crop,tuple(q[0]),tuple(q[1]),(0,200,255),1)
   if crop.shape[1]<1000:crop=cv2.resize(crop,None,fx=1000/crop.shape[1],fy=1000/crop.shape[1],interpolation=cv2.INTER_NEAREST)
   title=np.zeros((50,crop.shape[1],3),np.uint8);cv2.putText(title,entry['entry_id']+' '+signature['status'],(8,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
   cv2.imencode('.png',np.vstack([title,crop]))[1].tofile(str(folder/(Path(stem).name+'_'+entry['entry_id']+'.png')))
  result.append(case)
 # 平滑斜周期条纹、非周期粗糙纹理、过曝空白；旋转/端点反序不改变周期。
 yy,xx=np.mgrid[:320,:640].astype(float);rng=np.random.default_rng(7342);synthetics=[]
 for kind,angle in [('periodic',0),('periodic',23),('rough',0),('saturated',0)]:
  d=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))]);n=np.array([-d[1],d[0]]);origin=np.array([80.,80.]);s=(xx-origin[0])*d[0]+(yy-origin[1])*d[1];v=(xx-origin[0])*n[0]+(yy-origin[1])*n[1]
  image=np.full(xx.shape,255.) if kind=='saturated' else (120+55*np.cos(2*np.pi*(s/22+v/25)) if kind=='periodic' else cv2.GaussianBlur(rng.normal(120,25,xx.shape).astype(np.float32),(0,0),1.2))
  seg=np.array([origin,origin+400*d]);signature=measure_strand_signature(image.astype(np.float32),seg,20)
  reverse=measure_strand_signature(image.astype(np.float32),seg[::-1],20)
  assert signature==reverse
  if kind=='periodic':assert signature['status']=='periodic_strand_evidence' and abs(signature['period_px']-22)<1.5
  if kind=='saturated':assert signature['status']=='unknown'
  if kind=='rough':assert signature['status']!='periodic_strand_evidence'
  synthetics.append({'kind':kind,'angle_deg':angle,'signature':signature})
 report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Original pixels at saved automatic A6 outside segments; both competing0075 entries measured equally; qualitative known-tool diagnostic, not trained identity classifier.','code_sha256':sha(HERE/'strand_signature.py'),'originals_unchanged':True,'parameter_changes_per_file':False,'cases':result,'synthetic_checks':synthetics,'exit_code':0}
 (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps([{'id':c['output_file'],'entries':[{'entry':e['entry_id'],**{k:e['signature'].get(k) for k in ['status','period_px','median_peak_correlation','periodic_window_fraction','consistent_window_fraction','oblique_window_fraction','period_relative_mad']}} for e in c['entries']]} for c in result]))
if __name__=='__main__':main()
