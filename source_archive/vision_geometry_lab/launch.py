"""Path-safe entry point for Windows double-click launcher."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_geometry_lab.app import main

if __name__ == '__main__':
    main()
