from euclid.euclid_math import *
from euclid.euclid2 import dist, mid, GObject, Line, Segment, Ray
import numpy as np
from numpy.core.numeric import isclose
from numpy.lib.shape_base import row_stack


class GObject3(GObject):
    """ base class for 3d geometric objects """
    def __init__(self):
        pass

def intersection_line3_line3(line1, line2):
    dx, dy, dz = row_vector(line2.A - line1.A)
    v1, v2, v3 = row_vector(line1.v)
    u1, u2, u3 = row_vector(line2.v)
    
    D = np.array([[dx], [dy]])
    C = np.array([[v1, -u1], 
                  [v2, -u2]])
    T = system(C, D)
    t1, t2 = row_vector(T)
    
    if isclose(dz, v3*t1 - u3*t2):
        out = line1(t1)
    else:
        out = None
    return out

def intersection_line3_segment3(line, segment):
    int_=intersection_line3_line3(line, segment.line)
    if not int_ is None:
        out=int_ if (int_ in segment) else None
    else:
        out=None
    return out

def intersection_segment3_segment3(segment1, segment2):
    int_=intersection_line3_line3(segment1.line, segment2.line)
    if not isnone(int_):
        out=int_ if (int_ in segment1) and (int_ in segment2) else None
    else:
        out=None
    return out

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


def intersection_plane_plane_plane(plane1, plane2, plane3):
    C = col_vector([plane1.c, plane2.c, plane3.c])
    n1x, n1y, n1z = row_vector(plane1.n)
    n2x, n2y, n2z = row_vector(plane2.n)
    n3x, n3y, n3z = row_vector(plane3.n)

    N = np.array([[n1x, n1y, n1z],
                  [n2x, n2y, n2z],
                  [n3x, n3y, n3z]])
    
    X = system(N, C)

    return X


def intersection_plane_plane(plane1, plane2):
    v = col_vector(np.cross(row_vector(plane1.norm()), row_vector(plane2.norm())))
    return None


class Plane(GObject3):
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        self.u = self.B - self.A
        self.v = self.C - self.A
        self.n = self.norm()
        self.c = np.dot(row_vector(self.n), row_vector(self.A))
        self.type = "plane"

    def __call__(self, s, t):
        return (self.A + self.u*s + self.v*t)

    def normst(self, s, t):
        return (self.A + norm(self.u)*s + norm(self.v)*t)

    def norm(self):
        U, V = row_vector(self.u), row_vector(self.v)
        N = col_vector(np.cross(U, V))
        n = N/dist(col_vector([0, 0, 0]), N)
        return n

    def fxy(self, x, y):
        n, c = self.n, self.c
        nx, ny, nz = row_vector(n)
        if not isclose(nz, 0):
            z = (c - (nx*x + ny*y))/nz
        elif isclose((c - (nx*x + ny*y)), 0):
            z = 1
        else:
            z = None
        return z


def impl_param_plane(n, c):
    pass
