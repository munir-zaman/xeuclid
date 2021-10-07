from euclid.utils.math import *
from euclid.euclid2 import dist, mid, GObject, Line, Segment, Ray
import numpy as np


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


class Line3(GObject3):
    def __init__(self, A, B):
        self.A=A
        self.B=B
        self.v=self.B-self.A
        self.type = "line3"

    def __call__(self,t):
        return (self.A+self.v*t)

    def normt(self, t):
        return self.A + norm(self.v) * t

    def __repr__(self):
        return f"[{self.A[0,0]}, {self.A[1,0]}, {self.A[2,0]}] + [{self.v[0,0]}, {self.v[1,0]}, {self.v[2,0]}] * t"

    def __str__(self):
        return f"[{self.A[0,0]}, {self.A[1,0]}, {self.A[2,0]}] + [{self.v[0,0]}, {self.v[1,0]}, {self.v[2,0]}] * t"
    
    def inv(self, point):
        out = None
        P = row_vector(point - self.A)
        V = row_vector(self.v)
        for i in range(0, len(V)):
            if not isclose(V[i], 0):
                t = P[i]/V[i]
        out = t if np.all(P == V*t) else None
        return out

    def fxy(self, x, y):
        pass

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

    def intersection(self,obj):
        if obj.type=="line3":
            out=intersection_line3_line3(self, obj)
        else:
            out=None

        return out

    def __and__(self, obj):
        return self.intersection(obj)


class Segment3(GObject3):
    def __init__(self, A: np.ndarray, B: np.ndarray):
        self.A=A
        self.B=B
        self.v=self.B-self.A
        self.type="segment3"
        self.line=Line3(self.A, self.B)
        self.mid=mid(self.A, self.B)
        self.length=dist(self.A, self.B)

    def __call__(self, t):
        return (self.A+ self.v*t)

    def normt(self, t):
        return self.A + norm(self.v) * t

    def __contains__(self, point):
        return (not self.inv(point) is None)

    def inv(self,P):
        out_=self.line.inv(P)
        if out_!=None:
            out= out_ if (0 <= round(out_, 8) <= 1) else None
        else:
            out=None

        return out

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


class Ray3(GObject3):
    def __init__(self, A, B):
        self.A=A
        self.B=B
        self.v=self.B-self.A
        self.type="ray3"
        self.line=Line3(self.A, self.B)

    def __call__(self, t):
        return self.A+ self.v*t

    def normt(self, t):
        return self.A + norm(self.v) * t

    def __repr__(self):
        return f"[{self.A[0,0]}, {self.A[1,0]}] +[{self.v[0,0]}, {self.v[1,0]}]*t, t >= 0"

    def __str__(self):
        return str(f"[{self.A[0,0]}, {self.A[1,0]}] +[{self.v[0,0]}, {self.v[1,0]}]*t, t >= 0")

    def __contains__(self, point):
        return (not self.inv(point) is None)

    def inv(self, point):
        out_=self.line.inv(point)
        if out_!=None:
            out= out_ if (round(out_, 8) >= 0) else None
        else:
            out=None

        return out

    def __add__(self,vector):
        return Ray3(self.A+vector, self.B+vector)

    def __radd__(self,vector):
        return Ray3(self.A+vector, self.B+vector)

    def __sub__(self,vector):
        return Ray3(self.A-vector, self.B-vector)

    def __rsub__(self,vector):
        return Ray3(vector-self.A, vector-self.B)

    def __mul__(self,value):
        return Ray3(value*self.A, value*self.B)

    def __rmul__(self,value):
        return Ray3(value*self.A, value*self.B)

    def __truediv__(self,value):
        return Ray3(self.A/value, self.B/value)

    def intersection(self,obj):
        out = None
        return out


def intersection_line3_plane(line, plane):
    A1 = line.A
    u1x, u1y, u1z = row_vector(line.v)

    A2 = plane.A
    u2x, u2y, u2z = row_vector(plane.u)
    v2x, v2y, v2z = row_vector(plane.v)

    B = A1 - A2
    A = np.array([[u2x, v2x, -u1x ],
                  [u2y, v2y, -u1y ],
                  [u2z, v2z, -u1z]])

    X = system(A, B)
    s, t, r = row_vector(X)
    P = line(r)

    return P

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
