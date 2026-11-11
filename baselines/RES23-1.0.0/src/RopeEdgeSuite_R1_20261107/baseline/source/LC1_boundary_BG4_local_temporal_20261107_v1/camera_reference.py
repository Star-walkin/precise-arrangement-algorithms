"""Absolute registration to first-frame background avoids accumulated LK drift."""
import cv2,numpy as np,math

class CameraContinuity:
    def __init__(self):self.reference=None;self.points=None;self.unconfirmed=0;self.changed=False;self.suspicious=0
    def step(self,image):
        gray=cv2.cvtColor(cv2.resize(image,(1024,750)),cv2.COLOR_BGR2GRAY)[:370]
        if self.reference is None:
            self.reference=gray;self.points=cv2.goodFeaturesToTrack(gray,250,.015,10)
            return dict(confirmed=True,camera_changed=False,translation_px=0.,rotation_deg=0.,unconfirmed_duration_s=0.,method='fixed_first_frame_background')
        a=self.points;confirmed=False;shift=None;rotation=None;inliers=0;at_d=None
        if a is not None and len(a)>=20:
            b,st,_=cv2.calcOpticalFlowPyrLK(self.reference,gray,a,None)
            aa,back,_=cv2.calcOpticalFlowPyrLK(gray,self.reference,b,None)
            ok=(st[:,0]>0)&(back[:,0]>0)&(np.linalg.norm(aa-a,axis=2)[:,0]<.6)
            if np.sum(ok)>=20:
                M,mask=cv2.estimateAffinePartial2D(a[ok],b[ok],method=cv2.RANSAC,ransacReprojThreshold=.75)
                if M is not None:
                    inliers=int(np.sum(mask));confirmed=inliers>=20 and inliers/np.sum(ok)>=.70
                    if confirmed:
                        shift=float(np.linalg.norm(M[:,2])*4)
                        rotation=math.degrees(math.atan2(M[1,0],M[0,0]))
                        point=np.array([2238.7/4,2400/4]);at_d=float(np.linalg.norm(M[:,:2]@point+M[:,2]-point)*4)
                        suspect=shift>8 or at_d>8 or abs(rotation)>.2
                        self.suspicious=self.suspicious+1 if suspect else 0
                        if self.suspicious>=3:self.changed=True
        self.unconfirmed=0 if confirmed else self.unconfirmed+1
        return dict(confirmed=confirmed,camera_changed=self.changed,translation_px=shift,rotation_deg=rotation,
            displacement_at_measurement_roi_px=at_d,inliers=inliers,suspicious_frames=self.suspicious,
            unconfirmed_duration_s=self.unconfirmed/10,method='fixed_first_frame_background_LK_RANSAC')
