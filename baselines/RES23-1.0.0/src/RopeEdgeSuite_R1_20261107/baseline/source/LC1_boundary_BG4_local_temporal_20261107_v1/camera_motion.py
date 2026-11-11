import cv2,numpy as np,math

class CameraContinuity:
    """Track stationary upper background, independent of the moving spool ROI."""
    def __init__(self):self.previous=None;self.unconfirmed=0;self.accum=np.eye(3);self.changed=False
    def step(self,image):
        gray=cv2.cvtColor(cv2.resize(image,(1024,750)),cv2.COLOR_BGR2GRAY)[:370]
        if self.previous is None:
            self.previous=gray;return dict(confirmed=True,camera_changed=False,translation_px=0.,rotation_deg=0.,unconfirmed_duration_s=0.)
        a=cv2.goodFeaturesToTrack(self.previous,250,.015,10)
        confirmed=False;shift=None;rotation=None;inliers=0
        if a is not None and len(a)>=20:
            b,st,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,a,None)
            aa,back,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,b,None)
            ok=(st[:,0]>0)&(back[:,0]>0)&(np.linalg.norm(aa-a,axis=2)[:,0]<.6)
            if np.sum(ok)>=20:
                M,mask=cv2.estimateAffinePartial2D(a[ok],b[ok],method=cv2.RANSAC,ransacReprojThreshold=.75)
                if M is not None:
                    inliers=int(np.sum(mask));confirmed=inliers>=20 and inliers/np.sum(ok)>=.70
                    if confirmed:
                        step=np.eye(3);step[:2]=M;self.accum=step@self.accum
                        shift=float(np.linalg.norm(self.accum[:2,2])*4)
                        rotation=math.degrees(math.atan2(self.accum[1,0],self.accum[0,0]))
                        if shift>8 or abs(rotation)>.2:self.changed=True
        self.unconfirmed=0 if confirmed else self.unconfirmed+1;self.previous=gray
        return dict(confirmed=confirmed,camera_changed=self.changed,translation_px=shift,rotation_deg=rotation,
            inliers=inliers,unconfirmed_duration_s=self.unconfirmed/10,method='upper_background_LK_RANSAC_accumulated')
