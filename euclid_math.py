import math as mth
import numpy as np


abs_tol=10**-5

eq=lambda a,b: mth.isclose(a,b,abs_tol=abs_tol)

sqrt=lambda n: mth.sqrt(n) 

sin=lambda a: mth.sin(mth.radians(a))

asin=lambda a: mth.degrees(mth.asin(a))%180

cos=lambda a: mth.cos(mth.radians(a))

acos=lambda a: mth.degrees(mth.acos(a))%180

tan=lambda a: mth.tan(mth.radians(a))

atan=lambda a: mth.degrees(mth.atan(a))%180

atan2=lambda x,y: mth.degrees(mth.atan2(y,x))


def system2(A1,A2):
    """ returns the solution to the following system of linear equations, 
        A1[0]x+A1[1]y+A1[2]=0
        A2[0]x+A2[1]y+A2[2]=0

        A1,A2: list of length 3

    """

    a1,b1,c1=A1
    a2,b2,c2=A2
    if (a1*b2-b1*a2)!=0:
        x=(b1*c2-c1*b2)/(a1*b2-b1*a2)
        y=(c1*a2-a1*c2)/(a1*b2-b1*a2)
        out=[x,y]
    else:
        print("solution does not exist")
        out=None
    return out

in_interval=lambda x,a,b: (a <= x) and (x <= b)

def quad(a,b,c):
    d=b**2-4*a*c
    if b >= 0:
        out=[(-b+d)/2*a,(-b-d)/2*a]
    else:
        print("solution does not exist in R")
        out=None 
    return out

col_vector=lambda A: np.reshape(A,(len(A),1))

def system(A,B):
    """ returns the solution, `x`, of the linear system, `Ax=B` 
        where `x` and `B` are column vectors and `A` is a matrix. 
    """
    return np.linalg.solve(A, B)

def matmul(A,B):
    return np.linalg.matmul(A,B)

def det(A):
    """ returns the determinant,`np.linalg.det(A)`, of the matrix A """
    return np.linalg.det(A)

vectorize=lambda func: np.vectorize(func)

