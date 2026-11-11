from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
R21=ROOT.parent/'RopeEdgeSuite_R21_20261110'
sys.path.append(str(R21))
from common21 import read,dump,sha,pixelsha,plan,gzread,frontend,R20
import numpy as np
