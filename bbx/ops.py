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
        print("d")
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
    cx,cy = np.split(center, 2, axis=1)
    x1,x2 = cx-0.5*new_width,  cx+0.5*new_width
    y1,y2 = cy-0.5*new_height, cx+0.5*new_height
    new_boxes = Boxes(np.hstack([x1,y1,x2,y2]), **boxes.fields)

    return new_boxes


def resize(boxes:Boxes, scale=1) -> Boxes:
    cx,cy = np.split(boxes.center(), 2, axis=1)
    new_width = scale * boxes.width()
    new_height = scale * boxes.height()
    x1,x2 = cx-0.5*new_width,  cx+0.5*new_width
    y1,y2 = cy-0.5*new_height, cx+0.5*new_height
    new_boxes = Boxes(np.hstack([x1,y1,x2,y2]), **boxes.fields)
    return new_boxes


def non_maxima_suppression(boxes:Boxes, iou_threshold=0.5, score_threshold=0):
    if not boxes.has_field("score"):
        raise ValueError("Boxes most have score field")
    raise NotImplementedError
