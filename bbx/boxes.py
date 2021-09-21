from typing import Any, Iterable

import numpy as np


def expand_parameter(x):
    if isinstance(x, Iterable):
        if len(x) != 2:
            raise ValueError("Expected iterable with 2 items")
        if isinstance(x, np.ndarray):
            assert(x.ndim == 1)
        a, b = x
        return a, b
    else:
        assert(isinstance(x, (int, float)))
        return x, x


class Boxes:
    """
    A collection of axis-aligned bounding boxes with their properties.
    """
    def __init__(self, C:np.ndarray, **kwargs):
        C = np.atleast_2d(C)
        if not isinstance(C,np.ndarray):
            raise TypeError("Coordinates must be a numpy array")
        if C.ndim != 2 or C.shape[1] != 4:
            raise ValueError("Coordinates must be a matrix with 4 columns")
        a,b,c,d = np.split(C, 4, axis=1)
        x1, x2 = np.minimum(a,c), np.maximum(a,c)
        y1, y2 = np.minimum(b,d), np.maximum(b,d)
        self.C = np.hstack([x1,y1,x2,y2])
        self.fields = dict()
        self.set_fields(**kwargs)
    def __len__(self) -> int:
        return len(self.C)
    def __getitem__(self, indices) -> "Boxes":
        k = [indices] if isinstance(indices, int) else indices
        B = Boxes(self.C[k])  # New instance from coords
        for field, val in self.fields.items():
            B.set_field(field, np.atleast_1d(val[k]))
        return B

    @staticmethod
    def from_numpy(x:np.ndarray) -> "Boxes":
        return Boxes(x)

    @staticmethod
    def from_points(iterable:Iterable[Any]) -> "Boxes":
        boxes = []
        for points in iterable:
            pts = np.array(points, np.float)
            pts:np.ndarray
            assert(pts.ndim==2 and pts.shape[1]==2)
            x1,y1 = pts.min(0)
            x2,y2 = pts.max(0)
            boxes.append([x1,y1,x2,y2])
        return Boxes(boxes)

    # Modifiers
    def resize(self, scale=1) -> "Boxes":
        """Resize boxes and keep center"""
        sx, sy = expand_parameter(scale)
        cx,cy = np.split(self.center(), 2, axis=1)
        new_width = sx * np.expand_dims(self.width(), axis=1)
        new_height = sy * np.expand_dims(self.height(), axis=1)
        x1,x2 = cx-0.5*new_width,  cx+0.5*new_width
        y1,y2 = cy-0.5*new_height, cy+0.5*new_height
        return Boxes(np.hstack([x1,y1,x2,y2]), **self.fields)

    def shift(self, shift, relative=False) -> "Boxes":
        sx, sy = expand_parameter(shift)
        cx, cy = np.split(self.center(), 2, axis=1)
        w = np.expand_dims(self.width(), axis=1)
        h = np.expand_dims(self.height(), axis=1)
        new_cx = cx + w*sx if relative else cx + sx
        new_cy = cy + h*sy if relative else cy + sy
        x1,x2 = new_cx-w/2, new_cx+w/2
        y1,y2 = new_cy-h/2, new_cy+h/2
        return Boxes(np.hstack([x1,y1,x2,y2]), **self.fields)

    def scale(self, scale=1) -> "Boxes":
        sx, sy = expand_parameter(scale)
        x1, y1, x2, y2 = self.coordinates()
        return Boxes(np.hstack([sx*x1,sy*y1,sx*x2,sy*y2]), **self.fields)

    def normalized(self, shift=0, scale=1) -> "Boxes":
        return self.shift(shift).scale(scale)

    # Properties
    def numpy(self) -> np.ndarray:
        """Get coordinates as (N,4) matrix"""
        return self.C
    get = numpy  # backward compatibility
    def coordinates(self):
        """Returns x1,y1,x2,y2 as 4 arrays (N,1)"""
        return np.split(self.C, 4, axis=1)
    def top_left(self) -> np.ndarray:
        return self.C[:,:2]
    def bottom_right(self) -> np.ndarray:
        return self.C[:,2:]
    def corners(self) -> np.ndarray:
        """ Get cordinates of box corners
        
        Output
        ------
        tl,tr,br,bl : ndarray
            Each is (N,2) matrix with (x,y) coordinates for
            top-left, top-right, bottom-right, bottom-left corners
        """
        x1, y1, x2, y2 = self.coordinates()
        return np.hstack([x1, y1]),\
               np.hstack([x2, y1]),\
               np.hstack([x2, y2]),\
               np.hstack([x1, y2])
    def center(self) -> np.ndarray:
        """Get central points of lines"""
        return (self.top_left() + self.bottom_right()) / 2
    def width(self) -> np.ndarray:
        x1,_,x2,_ = self.coordinates()
        return (x2 - x1).flatten()
    def height(self) -> np.ndarray:
        _,y1,_,y2 = self.coordinates()
        return (y2 - y1).flatten()
    def area(self) -> np.ndarray:
        x1,y1,x2,y2 = self.coordinates()
        return ((x2 - x1) * (y2 - y1)).flatten()
    def aspect_ratio(self) -> np.ndarray:
        x1,y1,x2,y2 = self.coordinates()
        return ((x2 - x1) / (y2 - y1)).flatten()
    
    # Field management
    def _validate_field(self, v:np.ndarray) -> bool:
        if not isinstance(v, np.ndarray):
            raise TypeError("Only numpy arrays are supported for fields")
        if len(v) != len(self):
            raise ValueError(f"Expected {len(self)} items, {len(v)} passed")
    # Setters
    def set_fields(self, overwrite=True, **fields):
        for k,v in fields.items():
            self.set_field(k, v, overwrite=overwrite)
    add_fields = set_fields

    def set_field(self, field, value, overwrite=True):
        value = np.atleast_1d(value)
        self._validate_field(value)
        if not overwrite and field in self.fields:
            raise KeyError(f"Field {field} already present")
        self.fields[field] = value.copy()

    # Getters
    def get_field(self, field) -> np.ndarray:
        return self.fields[field]
    
    # Deletion
    def del_field(self, field):
        self.fields.pop(field)

    # Information
    def has_field(self, field) -> bool:
        return field in self.fields
    def fields_names(self):
        return self.fields.keys()
    get_fields = fields_names  # compatibility

    def __repr__(self):
        def box_info(k:int):
            x1, y1, x2, y2 = self.C[k]
            s = f"(x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}) "
            for key, val in self.fields.items():
                v = val[k]
                if isinstance(v, np.ndarray):  # For arrays
                    s += f"{key}=[{v.shape}, {v.dtype}] "
                else:
                    s += f"{key}={v} "
            return s

        header = f"{self.__class__} at {id(self):x}, n={len(self)} "
        if self.fields:
            f = ", ".join(self.fields.keys())
            header += f"with keys: {f}"
        if len(self) < 20:
            box_lines = [box_info(k) for k in range(len(self))]
        else:
            box_lines = [box_info(k) for k in range(3)]
            box_lines.append("...")
            box_lines.append(box_info(len(self)-1))
        lines = [header] + box_lines
        return "\n".join(lines)

        
def empty(*fields) -> Boxes:
    """
    Create empty boxes
    """
    coords = np.empty((0,4), np.float32)
    extra_fields = {name:[] for name in fields}
    return Boxes(coords, **extra_fields)


def empty_like(other:Boxes) -> Boxes:
    """
    New empty Boxes with same fields as `other`
    """
    return empty(*other.fields_names())
