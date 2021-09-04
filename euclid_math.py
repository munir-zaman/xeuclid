import math as mth
import numpy as np

abs_tol=10**-5
def eq(a,b):
    return mth.isclose(a,b,abs_tol=abs_tol)

def array(*n):
    return np.array(n)

def sqrt(n):
    if isinstance(n, Array):
        out=[mth.sqrt(x) for x in n]
    else:
        out=mth.sqrt(n)

    return out

def sin(a):
    return mth.sin(mth.radians(a))

def asin(a):
    return mth.degrees(mth.asin(a))%180

def cos(a):
    return mth.cos(mth.radians(a))

def acos(a):
    return mth.degrees(mth.acos(a))%180

def tan(a):
    return mth.tan(mth.radians(a))

def atan(a):
    return mth.degrees(mth.atan(a))%180

def atan2(x,y):
    return mth.degrees(mth.atan2(y,x))

def Vectorize(func):
    return lambda A: [func(x) for x in A]


class Array(list):
    def __init__(self,values):
        super().__init__(values)
        self.as_list=list(values)
        self.as_tuple=tuple(values)

    def __add__(self,n):
        if isinstance(n, float) or isinstance(n, int):
            n=[n for i in range(len(self))]
        A=[x+y for x,y in zip(self,n)]
        return Array(A)

    def __radd__(self,n):
        if isinstance(n, float) or isinstance(n, int):
            n=[n for i in range(len(self))]
        A=[x+y for x,y in zip(self,n)]
        return Array(A)

    def __sub__(self,n):
        if isinstance(n, float) or isinstance(n, int):
            n=[n for i in range(len(self))]
        A=[x-y for x,y in zip(self,n)]
        return Array(A)

    def __rsub__(self,n):
        if isinstance(n, float) or isinstance(n, int):
            n=[n for i in range(len(self))]
        A=[y-x for x,y in zip(self,n)]
        return Array(A)

    def __mul__(self,n):
        if isinstance(n, float) or isinstance(n, int):
            n=[n for i in range(len(self))]
        A=[x*y for x,y in zip(self,n)]
        return Array(A)

    def __rmul__(self,n):
        if isinstance(n, float) or isinstance(n, int):
            n=[n for i in range(len(self))]
        A=[x*y for x,y in zip(self,n)]
        return Array(A)

    def __truediv__(self,n):
        if isinstance(n, float) or isinstance(n, int):
            n=[n for i in range(len(self))]
        A=[x/y for x,y in zip(self,n)]
        return Array(A)

    def __rtruediv__(self,n):
        if isinstance(n, float) or isinstance(n, int):
            n=[n for i in range(len(self))]
        A=[y/x for x,y in zip(self,n)]
        return Array(A)

    def __pow__(self,n):
        if isinstance(n, float) or isinstance(n, int):
            n=[n for i in range(len(self))]
        A=[x**y for x,y in zip(self,n)]
        return Array(A)

    def __rpow__(self,n):
        if isinstance(n, float) or isinstance(n, int):
            n=[n for i in range(len(self))]
        A=[y**x for x,y in zip(self,n)]
        return Array(A)

    def apply_func(self,func):
        return Array(Vectorize(func)(self))

    def __eq__(self,other):
        out=all([eq(x,y) for x,y in zip(self,other)])
        return out


def system2(A1,A2):
    """ returns the solution to the following system of linear equations, 
        A1[0]x+A1[1]y+A1[2]=0
        A2[0]x+A2[1]y+A2[2]=0

        A1,A2: list of length 3

    """
    a1,b1,c1=A1
    a2,b2,c2=A2
    x=(b1*c2-c1*b2)/(a1*b2-b1*a2)
    y=(c1*a2-a1*c2)/(a1*b2-b1*a2)
    return [x,y]


