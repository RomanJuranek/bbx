# Package *bbx*

Simple operations over bounding boxes.
_Bounding boxes_ are represented in XYWH format where `(X,Y)` is coordinate of top-left corner, and `(W,H)` its size. I.e. `bb = [0,0,64,16]` represents a rectangle of size 64x16 located at coordinate `(0,0)`. The units are always abstract - the interpretation is left for the user (usually _pixels_ are used).

The package defines several common operations - resize, aspect ratio changes, randomization.

```python
import bbx

bb = (-5,-5,10,10)   # Center at (0,0)
bb = bbx.set_aspect_ratio(bb, 2, bbx.KEEP_WIDTH)
print(bb)  # Prints [[-5.0, -10.0, 10, 20]]
```

# Installation

The package is in PyPI so just use `pip`
```
pip install bbx
```

# Documentation

Bounding boxes are defined in x,y,w,h format, but they can hold any additional information (e.g. ignore flag). All functions always work on first four elements.

## `bbx.set_aspect_ratio(bbs, ar, type)`
Set aspecr ratio `ar` to all bounding boxes in `bbs` using `type` method. The `type` can be one of:
* `bbx.EXPAND`
* `bbx.SHRINK`
* `bbx.KEEP_WIDTH`
* `bbx.KEEP_HEIGHT`
* `bbx.KEEP_AREA`

For unknown type, the function raises `NotImplementedError`


## `bbx.resize(bbs, scale)`
Resize `bbs` by `ratio` without moving center. The `ratio` can be:
* scalar - all boxes are resized by the same factor
* tuple of two scalars - resize all boxes with different scale for width and height

## `bbx.scale`

## `bbx.move`

## `bbx.center`

## `bbx.overlap`

## `bbx.dist_matrix`

## `bbx.nms`

## `bbx.groups`

## `bbx.randomize`


# TODOs
* Documentation
* Tests

# License

This code is published under [MIT License](LICENSE)
