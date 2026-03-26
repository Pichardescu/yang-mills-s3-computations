"""Root conftest for pytest — ensures yang_mills_s3 is importable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
