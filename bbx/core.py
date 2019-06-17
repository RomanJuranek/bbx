import numpy as np


EXPAND = 0
SHRINK = 1
KEEP_WIDTH = 2
KEEP_HEIGHT = 3
KEEP_AREA = 4


def __normalize_format(bbs):
    if not isinstance(bbs, np.ndarray) or bbs.size == 0:
        bbs = np.array(bbs, copy=True)
        if bbs.size == 0:
            bbs = np.empty((0,5), np.float32)
    return np.atleast_2d(bbs).astype(np.float32, copy=False)


def __width(bbs):
    return bbs[:,2]


def __height(bbs):
    return bbs[:,3]


def __aspect_ratio(bbs):
    return bbs[:,2] / bbs[:,3]


def __set_width(h, ar):
    return h*ar, h


def __set_height(w, ar):
    return w, w/ar


def __set_area(w,h,ar):
    area = w*h
    nw = np.sqrt(area * ar)
    nh = area / nw
    return nw, nh


def set_aspect_ratio(bbs, ar=1.0, type=KEEP_AREA):
    """
    Set apect ration without moving bb center
    Input:
        bbs     - Bounding boxes
        ar      - Target aspect ratio (width/height)
        type    - One of bbx.EXPAND, bbx.SHRINK, bbx.KEEP_WIDTH, bbx.KEEP_HEIGHT, bbx.KEEP_AREA
    Output:
        bbs width altered aspect ration set to ar
    """

    bbs = __normalize_format(bbs)
    w = __width(bbs)
    h = __height(bbs)

    if type is KEEP_AREA:
        nw, nh = __set_area(w, h, ar)
    elif type is KEEP_WIDTH:
        nw, nh = __set_height(w, ar)
    elif type is KEEP_HEIGHT:
        nw, nh = __set_width(h, ar)
    elif type is EXPAND:
        mask = w/h > ar
        nw = np.empty_like(w)
        nh = np.empty_like(h)
        nw[ mask], nh[ mask] = __set_height(w[mask], ar)
        nw[~mask], nh[~mask] = __set_width(h[~mask], ar)
    elif type is SHRINK:
        mask = w/h > ar
        nw = np.empty_like(w)
        nh = np.empty_like(h)
        nw[ mask], nh[ mask] = __set_width(h[mask], ar)
        nw[~mask], nh[~mask] = __set_height(w[~mask], ar)
    else:
        raise NotImplementedError

    sx, sy = (nw-w)/2, (nh-h)/2
    bbs[:,0] -= sx
    bbs[:,1] -= sy
    bbs[:,2] = nw
    bbs[:,3] = nh

    return bbs


def resize(bbs, ratio=1):
    """
    Resize all bounding boxes in bbs by ratio, keeping their center.
    Input:
        bbs - bounding boxes
        ratio - scalar, tuple or np.ndarray with resize ratio
    Output:
        np.ndarray with resized bounding boxes
    Raises:
        ValueError when wrong ratio is given

    There are different scenarios for various ratio inputs. When the ratio is
    scalar, all bounding boxes are resized by the same factor. In case of two
    scalars (tuple or array), all bbs are resized with different factor for width
    and height. Vector with length corresponding to the number of bbs, each bounding
    box is resized by its respective factor (same for width and height). The last
    case is when the ratio is matrix with two columns and rows corresponding to the
    number of bbs. Then each bb is resized by its respective factor different for
    width and height. In other cases, ValueError is raised.

    Example:
        bbs = [ [0,0,10,10], [10,10,10,10] ]
        bbx.resize(bbs, 2)      # [ [-5,-5,20,20],[5,5,20,20] ] - Just double the size of both
        bbx.resize(bbs, [1,2])  # [ [0,-5,10,20], [10,5,10,20] ] - Double the height
        bbx.resize(bbs, [[1,1],[2,2]])  # [ [0,0,10,10], [5,5,20,20] ] - Reisze only the second

    """
    bbs = __normalize_format(bbs)
    n = bbs.shape[0]

    r = np.array(ratio)
    if r.size == 1:
        rx = ry = r
    elif r.size == 2:
        rx,ry = r
    elif r.size == n:
        rx = ry = r.flatten()[:,None]
    elif r.ndim == 2 and r.shape == (n,2):
        rx = r[:,0]
        ry = r[:,1]
    else:
        raise ValueError("Wrong resize ratio")

    w = __width(bbs)
    h = __height(bbs)
    nw, nh = w*rx,  h*ry
    sx, sy = (nw-w)/2, (nh-h)/2

    bbs[:,0] -= sx
    bbs[:,1] -= sy
    bbs[:,2] = nw
    bbs[:,3] = nh

    return bbs


def scale(bbs, s=1):
    """
    Scale all bbs by the given factor
    """
    bbs = __normalize_format(bbs)
    bbs[:,:4] * s
    return bbs


def move(bbs, shift=0):
    bbs = __normalize_format(bbs)
    bbs[:,:2] += shift
    return bbs


def center(bbs):
    bbs = __normalize_format(bbs)
    return bbs[:,:2] + 0.5*bbs[:,2:4]
