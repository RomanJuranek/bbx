import numpy as np
from .core import __normalize_format, __width, __height, resize, move


def empty(n=0):
    return np.zeros((n,4),"f")


def randomize(bbs, scale_mu=0.1, scale_sigma=0.1, shift_sigma=0.1):
    """
    Randomly modify bb by moving and resize
    :param bb: Bounding box in [x,y,w,h] format
    :param scale_mu: Mean value of scale randomization
    :param scale_sigma: Variance of scale randomization
    :param shift_sigma: Variance in shift randomization
    :return:
    """
    bbs = __normalize_format(bbs)
    n = bbs.shape[0]
    scale = np.exp(scale_sigma * np.random.randn(n)) + scale_mu
    s = np.maximum(__width(bbs), __height(bbs)) * shift_sigma
    shift = s[...,None] * np.random.randn(n,2)
    return resize(move(bbs, shift), (scale, scale))
