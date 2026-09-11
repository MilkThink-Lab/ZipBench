import sys
from pathlib import Path

# test the source tree, not a previously installed copy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
