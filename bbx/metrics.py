import numpy as np
from .core import __normalize_format


# def __bb_to_scalespace(bbs):
#     res = np.empty_like(bbs[:,:4])
#     res[:,:2] = bbs[:,:2] + 0.5*bbs[:,2:4]  # center
#     res[:,3] = np.log(bbs[:,2] / bbs[:,3]) # aspect ratio
#     res[:,4] = np.sqrt(bbs[:,3] * bbs[:,4]) # area
#     return res


def overlap(bb0, bb1):
    x0,y0,w0,h0 = bb0[:4]
    x1,y1,w1,h1 = bb1[:4]
    xa = max(x0,x1)
    ya = max(y0,y1)
    xb = min(x0+w0,x1+w1)
    yb = min(y0+h0,y1+h1)
    if xa > xb or ya > yb:
        return 0
    i = (xb-xa) * (yb-ya)
    u = (w0*h0) + (w1*h1) - i
    return i / u


def dist_matrix(bbs0, bbs1, metric=overlap):
    """
    Compute pairwise metric between two sets of bounding boxes
    Input:
        bbs0, bbs1 - two sets of bounding boxes
    Output:
        D - (M,N) np.ndarray where D[i,j] = metrix(bbs0[i], bbs1[j])
    """
    bbs0 = __normalize_format(bbs0)
    bbs1 = __normalize_format(bbs1)
    u = bbs0.shape[0]
    v = bbs1.shape[0]

    D = np.empty((u,v), "f")
    for i,j in np.ndindex(u,v):
        D[i,j] = metric(bbs0[i],bbs1[j])

    return D
