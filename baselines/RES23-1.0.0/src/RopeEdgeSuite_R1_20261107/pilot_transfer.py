import time
import cv2
import transfer_engine as e
def main():
    out=e.ROOT/'pilot';out.mkdir(exist_ok=True)
    source=e.ROOT.parents[3]/'双相机同步/dual_20260923_113745/left_DA9627065.avi'
    c=cv2.VideoCapture(str(source));ok,im=c.read();assert ok;c.release()
    tick=time.perf_counter();front=e.Frontend();tr,cam=front.step(im,0,0)
    print('frontend',tr['valid'],'secs',time.perf_counter()-tick,flush=True)
    r=e.spatial_frame(im,tr,cam,0,0,e.sha(source),out);e.dump(out/'000000.json',r)
    print('pilot',r['region']['valid'],r['observation'],r['ECW']['valid'],'spatial secs',r['spatial_elapsed_s'],flush=True)
if __name__=='__main__':main()
