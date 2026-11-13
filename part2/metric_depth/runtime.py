"""Local dependency setup; does not modify system Python or upstream source."""
import sys
import os
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
scratch=ROOT/'.tmp'
scratch.mkdir(exist_ok=True)
os.environ['TMP']=str(scratch)
os.environ['TEMP']=str(scratch)
tempfile.tempdir=str(scratch)
for p in [ROOT/'vendor/UniDepth', ROOT/'vendor/python']:
    if p.is_dir() and str(p) not in sys.path: sys.path.insert(0, str(p))
