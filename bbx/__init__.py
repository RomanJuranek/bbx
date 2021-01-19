from pathlib import Path

from .boxes import Boxes
from .ops import non_maxima_suppression, set_aspect_ratio, resize, ARModify

__version__ = None

def _read_version():
    with open(Path(__file__).parent / "VERSION") as f:
        __version__ = f.read().strip()

_read_version()