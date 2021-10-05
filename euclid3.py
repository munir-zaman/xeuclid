from euclid.euclid_math import *
from euclid.euclid2 import dist, mid, GObject, Line, Segment, Ray


class GObject3(GObject):
    """ base class for 3d geometric objects """
    def __init__(self):
        pass


class Line3(Line, GObject3):
    def __init__(self, A, B):
        Line.__init__(self, A, B)
        self.type = "line3"

    def __repr__(self):
        return f"[{self.A[0,0]}, {self.A[1,0]}, {self.A[2,0]}] + [{self.v[0,0]}, {self.v[1,0]}, {self.v[2,0]}] * t"

    def __str__(self):
        return f"[{self.A[0,0]}, {self.A[1,0]}, {self.A[2,0]}] + [{self.v[0,0]}, {self.v[1,0]}, {self.v[2,0]}] * t"

    def fx(self, x):
        raise NotImplementedError("Line3.fx(x) is not defined")

    def fy(self, y):
        raise NotImplementedError("Line3.fy(y) is not defined")

    def __add__(self, vector):
        return Line3(self.A+vector, self.B+vector)

    def __radd__(self, vector):
        return Line3(self.A+vector, self.B+vector)

    def __sub__(self, vector):
        return Line3(self.A-vector, self.B-vector)

    def __rsub__(self, vector):
        return Line3(vector-self.A, vector-self.B)

    def __mul__(self, value):
        return Line3(value*self.A, value*self.B)

    def __rmul__(self, value):
        return Line3(value*self.A, value*self.B)

    def __truediv__(self, value):
        return Line3(self.A/value, self.B/value)

    def matmul(self, matrix):
        return Line3(matmul(matrix, self.A), matmul(matrix, self.B))

    def parallel_line(self,P):
        return Line3(P, self.v+P)

    def perpendicular_line(self,P):
        return None


class Segment3(Segment, GObject3):
    def __init__(self, A: np.ndarray, B: np.ndarray):
        Segment.__init__(self, A, B)
        self.type="segment3"
        self.line=Line3(self.A, self.B)

    def __add__(self,vector):
        return Segment3(self.A+vector, self.B+vector)

    def __radd__(self,vector):
        return Segment3(self.A+vector, self.B+vector)

    def __sub__(self,vector):
        return Segment3(self.A-vector, self.B-vector)

    def __rsub__(self,vector):
        return Segment3(vector-self.A, vector-self.B)

    def __mul__(self,value):
        return Segment3(value*self.A, value*self.B)

    def __rmul__(self,value):
        return Segment3(value*self.A, value*self.B)

    def __truediv__(self,value):
        return Segment3(self.A/value, self.B/value)

