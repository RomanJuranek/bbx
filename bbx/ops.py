from enum import Enum
from typing import Collection, List

import numpy as np

from .boxes import Boxes, empty


class ARModify(Enum):
    EXPAND = 0
    SHRINK = 1
    KEEP_WIDTH = 2
    KEEP_HEIGHT = 3
    KEEP_AREA = 4


def set_aspect_ratio(
    boxes:Boxes,
    ar:float=1,
    action:ARModify=ARModify.KEEP_WIDTH) -> Boxes:

    if action not in set(ARModify):
        raise ValueError("Wrong action")

    center = boxes.center()
    width = boxes.width()
    height = boxes.height()

    # Calculate new width and height according to action
    if action == ARModify.KEEP_AREA:
        area = boxes.area()
        new_width = np.sqrt(area * ar)
        new_height = area / new_width
    elif action == ARModify.KEEP_WIDTH:
        new_width = width
        new_height = width / ar
    elif action == ARModify.KEEP_HEIGHT:
        new_width = height * ar
        new_height = height
    elif action == ARModify.EXPAND or action == ARModify.SHRINK:
        if action == ARModify.EXPAND:
            mask = boxes.aspect_ratio() > ar
        else:
            mask = boxes.aspect_ratio() < ar
        new_width = np.empty_like(width)
        new_height = np.empty_like(height)
        new_width[ mask], new_height[ mask] = width, width/ar  # keep width
        new_width[~mask], new_height[~mask] = height*ar, height  # keep height
        pass
    
    # Compose new boxes
    new_width = np.expand_dims(new_width, axis=1)
    new_height = np.expand_dims(new_height, axis=1)
    cx,cy = np.split(center, 2, axis=1)
    x1,x2 = cx-0.5*new_width,  cx+0.5*new_width
    y1,y2 = cy-0.5*new_height, cy+0.5*new_height
    new_boxes = Boxes(np.hstack([x1,y1,x2,y2]), **boxes.fields)

    return new_boxes


def resize(boxes:Boxes, scale=1) -> Boxes:
    return boxes.resize(scale)


def shift(boxes:Boxes, shift=0, relative=False):
    return boxes.shift(shift, relative)


def concatenate(boxes_list:List[Boxes], fields:Collection[str]=None) -> Boxes:
    """Merge multiple boxes to a single instance
    B = A[:10]
    C = A[10:]
    D = concatenate([A, B])
    D should be equal to A
    """
    if not boxes_list:
        if fields is None:
            fields = []
        return empty(*fields)

    if fields is None:
        # Get fields common to all sub-boxes
        common_fields = set.intersection( *[set(x.get_fields()) for x in boxes_list] )
    else:
        common_fields = fields

    coords = np.concatenate([x.get() for x in boxes_list], axis=0)
    new_fields = dict()
    for f in common_fields:
        new_fields[f] = np.concatenate([x.get_field(f) for x in boxes_list], axis=0)
    return Boxes(coords, **new_fields)


def intersection(a:Boxes, b:Boxes) -> np.ndarray:
    """Calculate intersection of two sets of boxes"""
    ax1,ay1,ax2,ay2 = a.coordinates()
    bx1,by1,bx2,by2 = b.coordinates()

    min_ymax = np.minimum(ay2, by2.T)
    max_ymin = np.maximum(ay1, by1.T)
    h = np.maximum(min_ymax-max_ymin, 0)

    min_xmax = np.minimum(ax2, bx2.T)
    max_xmin = np.maximum(ax1, bx1.T)
    w = np.maximum(min_xmax-max_xmin, 0)

    return h * w


def iou(a:Boxes, b:Boxes) -> np.ndarray:
    """Calculate pairwise IoU of two sets of boxes"""
    intersect = intersection(a,b)
    area_a = np.expand_dims(a.area(),axis=1)
    area_b = np.expand_dims(b.area(),axis=0)
    union = area_a + area_b - intersect
    return intersect / union


def ioa(a:Boxes, b:Boxes) -> np.ndarray:
    """
    Calculate pairwise overlaps normalized by the area of 'a'
    """
    intersect = intersection(a,b)
    area_a = np.expand_dims(a.area(),axis=1)
    return intersect / area_a


def boxes_in_window(a:Boxes, window:Boxes, min_overlap=1.0) -> np.ndarray:
    """
    Get boxes from 'a' which are in any of 'window' boxes with at least 'min_overlap' area
    """
    return ioa(a, window).max(axis=1) >= min_overlap


def sort_by_field(b:Boxes, field:str, descending=True) -> Boxes:
    """Return boxes sorted by the value of specified field"""
    order = np.argsort(b.get_field(field))
    if descending:
        order = order[::-1]
    return b[order]


def overlapping_groups(boxes:Boxes, iou_threshold=1.0, order_by="scores"):
    b = sort_by_field(boxes, order_by)
    if len(b) == 0:
        return b
    assigned = np.full(len(b), False, np.bool)
    for i in range(len(b)):
        if assigned[i]: continue
        unassigned_indices = np.where(~assigned)[0]
        if unassigned_indices.size == 0: break
        first = unassigned_indices[0]
        add_to_group = iou(b[first], b[unassigned_indices])[0] >= iou_threshold
        group_indices = unassigned_indices[add_to_group]
        assigned[group_indices] = True
        yield b[group_indices]


def _softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def non_max_suppression(boxes:Boxes, iou_threshold=0.5, min_score=0, score_field="scores", reduction="max", max_groups=None, min_group_size=1) -> Boxes:
    """
    Performs non maximum suppression of the input boxes

    Inputs
    ------
    boxes: Boxes
    iou_threshold: float
    min_score: float
    score_field: str
    reduction: str
    max_groups: int
    min_group_size: int
        Only groups with at least min_group_size boxes are emited
    """
    if not reduction in ["max", "mean"]:
        raise ValueError("Reduction must be 'max' or 'mean'")
    idx = boxes.get_field(score_field) > min_score
    nms_boxes = []
    for group_boxes in overlapping_groups(boxes[idx], iou_threshold=iou_threshold, order_by=score_field):
        if len(group_boxes) < min_group_size:
            continue
        if reduction == "mean":
            group_scores = group_boxes.get_field(score_field)
            group_weights = _softmax(group_scores)
            group_coords = np.average(group_boxes.get(), axis=0, weights=group_weights)
            nms_boxes.append(Boxes(group_coords, scores=group_scores.max(), size=len(group_boxes)))
        elif reduction == "max":
            group_scores = group_boxes.get_field(score_field)
            k = np.argmax(group_scores)
            group_coords = group_boxes[k].get()
            nms_boxes.append(Boxes(group_coords, scores=group_scores.max(), size=len(group_boxes)))
        if max_groups is not None and len(nms_boxes) == max_groups:
            break
    return concatenate(nms_boxes, ["scores", "size"])
