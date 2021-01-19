# Package *bbx*

Simple operations over bounding boxes.
The package provides a class `Boxes` encapsulating a set of bounding boxes. Internally, the boxes are represented in an Nx4 matrix with x1,y1,x2,y2 coordinates. Instances of `Boxes` support arbitraty user-defined fields which can hold other properties of boxes like e.g. score.

```python
import bbx

B = bbx.Boxes(np.array([[0,0,10,10]]))  # New instance. Explicit conversion to array required
B.set_field("score", np.array([1]))  # Set the field
B.width()  # [[10]]
C = bbx.resize(B, 2)  # Resize and make new instance
C.get() # [[-5,-5,15,15]]
```

# Installation

The package is in PyPI so just use `pip`
```
pip install bbx
```

# Contribute

If you find the package useful, contribute with code. You can also submit an issue if something is broken.

# License

This code is published under [MIT License](LICENSE)
