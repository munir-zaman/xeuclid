from xeuclid.utils.math import *
from xeuclid.euclid2 import convex_hull
from scipy.special import comb


def bernstein_poly_func(degree, index, interval=(0, 1)):
    """ computes the bernstein polynomial function with given degree, index and interval
        default interval is set to (0, 1):
    """
    n = degree
    i = index
    t0 = interval[0]
    t1 = interval[1]
    dt = t1 - t0
    # B_{[t_0, t_1]}(t, n, i) = {n \choose i} * (\frac{t_1 - t}{t_1 - t_0})^{n - i} (\frac{t -t_0}{t_1 - t_0})^{n}
    func = lambda t: comb(n, i)*((t1-t)**(n-i))*((t-t0)**i)*(1/dt**n)
    return func

def increment_degree(curve: BezierCurve) -> BezierCurve:
    """ increments the degree of the given bezier curve `curve` by one and
        returns the control points of the new bezier curve.
    """
    old_degree = curve.degree

    new_controls = []
    for i in range(0, old_degree + 1):
        new_control = (i/(old_degree + 1))*curve.controls[i-1] + (1 - (i/(old_degree + 1)))*curve.controls[i]
        new_controls.append(new_control)

    new_controls.append(curve.controls[-1])

    return new_controls

def elevate_degree(curve, n=1) -> BezierCurve:
    new_curve = curve
    new_controls = curve.controls[::]
    for i in range(0, n):
        new_controls = increment_degree(new_curve)
        new_curve = BezierCurve(new_controls)

    return new_controls

class BezierCurve:

    def __init__(self, *controls):
        """ base class for all bezier curves """
        self.controls = controls
        self.degree = len(self.controls) - 1
        self.interval = (0, 1)
        self.hull = convex_hull(self.controls)

    def __call__(self, t):
        return sum([bernstein_poly_func(self.degree, i, interval=self.interval)(t) * self.controls[i] for i in range(0, self.degree + 1)])

    def elevate(self, n=1):
        return elevate_degree(self, n=n)

class QuadBezier(BezierCurve):

    def __init__(self, p0, p1, p2):
        super().__init__(p0, p1, p2)


class CubicBezier(BezierCurve):

    def __init__(self, p0, p1, p2, p3):
        super().__init__(p0, p1, p2, p3)

