"""Puts the repo root on sys.path for pytest, the way each entry-point script
does for itself (see e.g. src/study/pilot.py).

Every module in this repo is imported by its full path -- `tools.loglog`,
`src.generate.generate`, `models.srw` -- and nothing is ever reachable under
two different names. That needs exactly one directory on sys.path: this one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
