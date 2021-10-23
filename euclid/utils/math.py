import math as mth
import numpy as np
import scipy.interpolate


def isnone(obj):
    return str(type(obj))=="<class 'NoneType'>"

round_val=8
abs_tol=10**-7

isclose=lambda a,b: mth.isclose(a,b,abs_tol=abs_tol,rel_tol=0)
rnd=lambda x: round(x,round_val) if not isnone(x) else None
rndv=np.vectorize(rnd)

sqrt=lambda n: mth.sqrt(n)

sin=lambda a: round(mth.sin(mth.radians(a)), 16)

asin=lambda a: mth.degrees(mth.asin(a))%180

cos=lambda a: round(mth.cos(mth.radians(a)), 16)

acos=lambda a: mth.degrees(mth.acos(a))%180

tan=lambda a: round(mth.tan(mth.radians(a)), 16)

atan=lambda a: mth.degrees(mth.atan(a))%180

atan2=lambda x,y: mth.degrees(mth.atan2(y,x))

def get_rid_of_multiple_points(l):
    L=l.copy()
    RND=np.vectorize(lambda x: round(x, 12))
    L_=set([tuple(RND(row_vector(point))) for point in L])
    out=[col_vector(p) for p in L_]

    return out

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

in_interval=lambda x,a,b: (rnd(a) <= rnd(x)) and (rnd(x) <= rnd(b))

def quad(a,b,c):
    """returns the list of real solutions to the quadratic equation, 
        a**2 * x+ b * x + c = 0
        returns an empty list if no solution exists
        a : float
        b : float
        c : float

        returns:   
            out : list of floats
    """
    d=b**2-4*a*c
    if round(d, 8) > 0:
        out=[(-b+d**(1/2))/(2*a),(-b-d**(1/2))/(2*a)]
    elif isclose(d, 0):
        out=[(-b)/(2*a)]
    else:
        out=[]
    return out

col_vector=lambda A: np.reshape(A,(len(A),1))
row_vector=lambda A: np.reshape(A,(len(A),))

def polar(r, theta, center=col_vector([0, 0])):
    V = col_vector( [r * cos(theta), r * sin(theta)] )
    v = V + center
    return v

def system(A,B):
    """ returns the solution, `x`, of the linear system, `Ax=B`
        where `x` and `B` are column vectors and `A` is a matrix.
        returns `None` if no solution exists.
        A : np.ndarray
        B : np.ndarray

        returns 
            out : None or np.ndarray
    """
    if np.linalg.det(A)!=0:
        out=np.linalg.solve(A, B)
    else:
        #print("solution does not exist in R")
        out=None
    return out

def matmul(A,B):
    return np.matmul(A,B)

def det(A):
    """ returns the determinant,`np.linalg.det(A)`, of the matrix A """
    return np.linalg.det(A)

vectorize=lambda func: np.vectorize(func)

strv=np.vectorize(str)

def norm(vector):
    vx,vy=row_vector(vector)
    r=mth.sqrt(vx**2+ vy**2)
    Vx,Vy=vx/r, vy/r
    V=col_vector([Vx, Vy])
    return V

def dist(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """ Returns the distance between points `p1` and `p2` """
    if p1.shape != p2.shape:
        raise ValueError("`p1` and `p2` must have the same shape")

    return round(mth.sqrt(sum((p1 - p2)**2)), 16)

def mid(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """ Returns the midpoint of points `p1` and `p2` """
    if p1.shape != p2.shape:
        raise ValueError("`p1` and `p2` must have the same shape")

    return (p1 + p2)/2

_CHOOSE_CACHE = {}

def choose(n: int, k: int, save=True) -> int:
    """ returns the value of n - choose - k = n(n-1)(n-2)...(n-k+1)/k! 
        if `save` is set to `True` then the `out` value will be saved to `_CHOOSE_CACHE`
    """
    if k>n:
        raise ArithmeticError("`k` should be less than or equal to `n`.")

    if (n,k) not in _CHOOSE_CACHE.keys():
        X = 1
        for i in range(0, k):
            X*=(n-i)
        out = X/mth.factorial(k)

        if save:
            _CHOOSE_CACHE.update({(n,k): out})

    else:
        out = _CHOOSE_CACHE[(n,k)]
    return out


def get_lagrange_polynomial_as_func(points):
    X = [point[0] for point in points]
    Y = [point[1] for point in points]

    return (lambda x: scipy.interpolate.lagrange(X, Y)(x))
