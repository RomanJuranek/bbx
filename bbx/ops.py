import numpy as np
from enum import Enum

from .boxes import Boxes


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
    else:
        raise ValueError("Wrong action")
    
    # Compose new boxes
    new_width = np.expand_dims(new_width, axis=1)
    new_height = np.expand_dims(new_height, axis=1)
    cx,cy = np.split(center, 2, axis=1)
    x1,x2 = cx-0.5*new_width,  cx+0.5*new_width
    y1,y2 = cy-0.5*new_height, cy+0.5*new_height
    new_boxes = Boxes(np.hstack([x1,y1,x2,y2]), **boxes.fields)

    return new_boxes


def resize(boxes:Boxes, scale=1) -> Boxes:
    if isinstance(scale, tuple):
        sx, sy = scale
    else:
        sx, sy = scale, scale
    cx,cy = np.split(boxes.center(), 2, axis=1)
    new_width = sx * np.expand_dims(boxes.width(), axis=1)
    new_height = sy * np.expand_dims(boxes.height(), axis=1)
    x1,x2 = cx-0.5*new_width,  cx+0.5*new_width
    y1,y2 = cy-0.5*new_height, cy+0.5*new_height
    new_boxes = Boxes(np.hstack([x1,y1,x2,y2]), **boxes.fields)
    return new_boxes


def concatenate(boxes_list) -> Boxes:
    """Merge multiple boxes to a single instance
    B = A[:10]
    C = A[10:]
    D = concatenate([A, B])
    D should be equal to A
    """
    # Get fields common to all sub-boxes
    common_fields = set.intersection( *[set(x.get_fields()) for x in boxes_list] )

    coords = np.concatenate([x.get() for x in boxes_list], axis=0)
    fields = dict()
    for f in common_fields:
        fields[f] = np.concatenate([x.get_field(f) for x in boxes_list], axis=0)
    return Boxes(coords, **fields)


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


def sort_by_field(b:Boxes, field:str, descending=True) -> Boxes:
    order = np.argsort(b.get_field(field))
    if descending:
        order = order[::-1]
    return b[order]

def non_max_suppression(boxes:Boxes, iou_threshold=0.5, score_threshold=0) -> Boxes:
    idx = boxes.get_field("scores") > score_threshold
    b = sort_by_field(boxes[idx], "scores")
    if len(b) == 0:
        return b
    valid = np.full(len(b), 1, np.bool)
    selected = []
    for i in range(len(b)):
        if valid[i]:
            selected.append(i)
            valid[i] = False
            valid_indices = np.where(valid)[0]
            if valid_indices.size == 0:
                break
            metric = iou(b[i], b[valid_indices])
            valid[valid_indices] = np.logical_and(
                valid[valid_indices],
                metric[0] < iou_threshold
            )
    return b[np.array(selected,"i")]
