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

# But why?

... for the glory of s... But seriously, I use bounding boxes in every other project and there is no decent small package for this. Yes I can use TF obj detection api (and I did for a while, this pkg have similar interface to it) or structures from imgaug or others. I did not find anything that suits my needs (if you know about something just let me know). You either install a big non-standard package with tons of functionality you do not need or you implement it by yourself. So I decided to make a very small package that does precisely what I need and nothing more, is portable (just numpy needed, and you already have it!).

Yes, I just implemented a non-standard package, so I so not need to use other non-standard packages... yes, I know...

# Installation

The package is in PyPI so just use `pip`
```
pip install bbx
```

# Contribute

Feelin' brave? Contribute with code! You can also submit an issue if something is broken.

# License

This code is published under [MIT License](LICENSE)
