import numpy as np


EXPAND = 0
SHRINK = 1
KEEP_WIDTH = 2
KEEP_HEIGHT = 3
KEEP_AREA = 4


def __normalize_format(bbs):
    bbs = np.array(bbs, copy=True)
    return np.atleast_2d(bbs).astype(np.float32)


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


def resize(bbs, ratio=(1, 1)):
    """
    Resize all bounding boxes in bbs by ratio, keeping their center.
    Input:
        bb - bounding boxes
        ratio - scalar or tuple with scale
    Output:
        np.ndarray with resized bounding boxes
    """
    if isinstance(ratio, tuple):
        rx, ry = ratio
        if ry is None:
            ry = rx
    else:
        rx = ry = ratio

    bbs = __normalize_format(bbs)
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
    bbs = __normalize_format(bbs)
    return bbs * s


def move(bbs, shift=(0, 0)):
    bbs = __normalize_format(bbs)
    bbs[:,:2] += shift
    return bbs


def center(bbs):
    bbs = __normalize_format(bbs)
    return bbs[:,:2] + 0.5*bbs[:,2:4]
