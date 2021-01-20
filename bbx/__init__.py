from pathlib import Path

from .boxes import Boxes
from .ops import ARModify
from .ops import non_max_suppression, set_aspect_ratio, resize, concatenate, intersection, iou, sort_by_field

__version__ = None

def _read_version():
    with open(Path(__file__).parent / "VERSION") as f:
        __version__ = f.read().strip()

_read_version()