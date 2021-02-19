from pathlib import Path

from .boxes import Boxes
from .ops import ARModify, concatenate, intersection, iou,\
                  non_max_suppression, resize, shift, \
                  set_aspect_ratio, sort_by_field

__version__ = None

def _read_version():
    with open(Path(__file__).parent / "VERSION") as f:
        __version__ = f.read().strip()

_read_version()
