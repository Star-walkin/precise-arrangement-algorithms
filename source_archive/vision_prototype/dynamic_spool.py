"""Current-frame rope-region proposals. No reference image or actuator access."""
from dataclasses import dataclass
import cv2
import numpy as np
from geometry_measurement import flange_curve


@dataclass
class SpoolProposal:
    bbox: tuple
    pitch: float
    confidence: float
    edge_curve: tuple = ()


def find_spool(gray, bgr=None):
    """Bounded-resolution orientation/colour proposal + inner-interface fit.

    A proposal is NOT a tangent/edge measurement. Geometry must independently
    validate it; no old reference coordinate is substituted on failure.
    """
    h, w = gray.shape
    scale = min(1., 1024 / w)
    small = cv2.resize(gray, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    xx = cv2.boxFilter(gx*gx, -1, (17, 17))
    yy = cv2.boxFilter(gy*gy, -1, (17, 17))
    directional = (yy-xx)/(yy+xx+1.)
    if bgr is None:
        return None
    colour = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    saturation = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)[:, :, 1]
    # A scene-specific proposal, not a universal metal classifier. The later
    # inner-interface fit MUST confirm coloured flange above rope texture.
    mask = ((directional > .4) & (yy > 200) & (saturation < 65)).astype(np.uint8)*255
    mask[:int(.25*sh)] = 0
    mask[:, :int(.12*sw)] = 0
    mask[:, int(.86*sw):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for label in sorted(range(1, n), key=lambda i: -stats[i, 4])[:8]:
        x, y, bw, bh, area = stats[label]
        if bw < .12*sw or bh < 25 or area < .002*sw*sh:
            continue
        pitches = []
        for col in np.linspace(x+.2*bw, x+.8*bw, 7).astype(int):
            signal = small[y:y+bh, col-1:col+2].mean(axis=1).astype(float)
            if len(signal) < 30 or signal.std() < 3:
                continue
            signal -= np.convolve(np.pad(signal, 15, mode='edge'), np.ones(31)/31, 'valid')
            maxlag = min(19, len(signal)//4)
            ac = [np.dot(signal[:-k], signal[k:]) /
                  max(1e-8, np.linalg.norm(signal[:-k])*np.linalg.norm(signal[k:]))
                  for k in range(2, maxlag+2)]
            peaks = [i for i in range(1, len(ac)-1) if ac[i] > .45 and ac[i] > ac[i-1] and ac[i] >= ac[i+1]]
            if peaks:
                best = max(ac[i] for i in peaks)
                i = next(i for i in peaks if ac[i] >= .85*best)
                pitches.append(i+2)
        pitch = float(np.median(pitches)) if pitches else 6.
        # Weak periodicity can still propose an inner boundary; it NEVER
        # authorizes rope counts. Full geometry validates pitch separately.
        curve = []
        fit = flange_curve(small, colour, (labels == label).astype(np.uint8)*255,
                           pitch, x+.2*bw, sampled_curve=curve)
        if fit is None or fit[1] < .4:
            continue
        candidates.append((fit[1]*np.sqrt(area), SpoolProposal(
            tuple(int(round(v/scale)) for v in (x, y, x+bw, y+bh)), pitch/scale, fit[1],
            tuple((px/scale, py/scale) for px, py in curve))))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None
