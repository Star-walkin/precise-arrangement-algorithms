from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
R22=ROOT.parent/'RopeEdgeSuite_R22_20261111'
sys.path.append(str(R22))
from common22 import read,dump,sha,pixelsha,plan,np,R21,R20
