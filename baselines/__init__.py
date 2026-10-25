"""Baseline framework location and compatibility import path."""
from pathlib import Path
import sys
BASELINES_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BASELINES_ROOT.parent
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))
