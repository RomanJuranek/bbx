import numpy as np
from .core import __normalize_format
from .metrics import overlap


def groups(bbs, scores=None, min_overlap=0.5):
    bbs = __normalize_format(bbs)
    idx = np.argsort(scores)[::-1]
    groups = {}
    suppressed = np.zeros(idx.size, np.bool)
    for i,idx_i in enumerate(idx[:-1]):
        if suppressed[idx_i]: continue
        groups[i] = [idx_i]
        for j,idx_j in enumerate(idx[i+1:]):
            if suppressed[idx_j]: continue
            if overlap(bbs[idx_i], bbs[idx_j]) > min_overlap:
                groups[i].append(idx_j)
                suppressed[idx_j] = True
    return groups


def nms(bbs, scores=None, min_group=1, min_overlap=0.5):
    bbs = __normalize_format(bbs)
    scores = np.array(scores)
    gs = groups(bbs, scores, min_overlap)

    res_bbs = []
    res_score = []
    for gid, group in gs.items():
        if len(group) >= min_group:
            res_bbs.append(np.mean(bbs[group,:4], axis=0))
            res_score.append(np.max(scores[group]))

    return np.array(res_bbs), np.array(res_score)
