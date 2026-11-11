import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import cProfile,pstats,io,cv2
from common20 import *
from roi21 import masks
cv2.setNumThreads(1)
item=next(x for x in plan() if x['id']=='dual_20260923_101517__right');i=380;r=record(item['id'],i)
cap=cv2.VideoCapture(item['source']);cap.set(1,i);ok,im=cap.read();cap.release();assert ok;im=cv2.flip(im,1)
_,mask,_,_=masks(im,r)
ref=load('_profile_exact',PARENT/'RopeEdgeSuite_R12_20261109/run_full.py')
p=cProfile.Profile();p.enable();result=ref.extract_current(im,mask,r['track']);p.disable();buf=io.StringIO();pstats.Stats(p,stream=buf).sort_stats('cumtime').print_stats(24)
(ROOT/'profile_f3.txt').write_text(buf.getvalue(),encoding='utf8');print(buf.getvalue())
