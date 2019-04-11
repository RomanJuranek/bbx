""" bbx - Simple operations over bounding boxes

Module provides simple primitives over bounding boxes often used in object
detection, like aspect ratio correction, resizing and distance computation.

Bounding boxes are represented as lists,tuples or np.ndarray with (x,y,w,h)
format. E.g.:

    bbs = [(1,1,10,10), (20,30,100,50)]

    or just

    bb = [1,1,10,10]

Operations in the module works on single or multiple bounding boxes, and always
returns 2d np.ndarray

Additionally, functions work always over first 4 element, so any elements after
them are always kept. This allows for storing additional data for each bounding
box ( ignore flag, etc.). Example:

    bb0 = [-5,0,10,10, -1, 0, 1]
    bb1 = bbx.set_aspect_ratio(bb0, 2)
    > [-10,0,20,10,-1,0,1]
"""

from .core import set_aspect_ratio, resize, scale, move, center
from .core import SHRINK, EXPAND, KEEP_AREA, KEEP_WIDTH, KEEP_HEIGHT
from .metrics import overlap, dist_matrix
from .nms import groups, nms
from .generate import randomize, empty
from os import path


with open(path.join(path.abspath(path.dirname(__file__)),"VERSION"),"r") as f:
    __version__ = f.read().strip()
