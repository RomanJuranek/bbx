import numpy as np


class Boxes:
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
        self.add_fields(**kwargs)
    def __len__(self) -> int:
        return self.C.shape[0]
    def __getitem__(self, indices) -> "Boxes":
        B = Boxes(self.C[indices])  # New instance from coords
        for field, val in self.fields.items():
            B.set_field(field, np.atleast_1d(val[indices]))
        return B

    # Modifiers
    def normalized(self, scale=1, shift=(0,0)) -> "Boxes":
        """Scale and shift line segments"""
        # Check scale
        if isinstance(scale, (int, float)):
            scale = np.atleast_2d([scale,scale]).astype("f")
        elif isinstance(scale, (list, tuple, np.ndarray)):
            sx,sy = scale
            scale = np.atleast_2d([sx,sy]).astype("f")
        else:
            raise TypeError("Scale must be scalar or two element vector")

        # Check shift
        if isinstance(shift, (list, tuple, np.ndarray)):
            sx,sy = shift
            shift = np.atleast_2d([sx,sy]).astype("f")
        else:
            raise TypeError("Shift must be scalar or two element vector")

        scale = np.tile(scale, 2)
        shift = np.tile(shift, 2)
        #print(scale, shift)
        B = Boxes((self.get()+shift)*scale)
        B.add_fields(**self.fields)
        return B

    # Properties
    def get(self) -> np.ndarray:
        """Get coordinates as (N,4) matrix"""
        return self.C
    def coordinates(self):
        """Returns x1,y1,x2,y2 as 4 arrays (N,1)"""
        return np.split(self.C, 4, axis=1)
    def top_left(self) -> np.ndarray:
        return self.C[:,:2]
    def bottom_right(self) -> np.ndarray:
        return self.C[:,2:]
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
        if v.shape[0] != len(self):
            raise ValueError(f"Expected {len(self)} items, {v.shape[0]} passed")
    def add_fields(self, **fields):
        """Add multiple fields to the instance"""
        for field,value in fields.items():
            self.set_field(field, value)
    def get_field(self, field) -> np.ndarray:
        return self.fields[field]
    def set_field(self, field, value, overwrite=True):
        value = np.atleast_1d(value)
        self._validate_field(value)
        if not overwrite and field in self.fields:
            raise KeyError(f"Field {field} already present")
        self.fields[field] = value.copy()
    def del_field(self, field):
        self.fields.pop(field)
    def has_field(self, field) -> bool:
        return field in self.fields
    def get_fields(self):
        return self.fields.keys()