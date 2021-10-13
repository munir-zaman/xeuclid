from euclid.utils.math import *


def bernstein_poly_func(degree, index):
    n = degree
    i = index
    func = lambda t: choose(n, i) * ((1-t)**(n-i)) * (t**i)
    return func


class BezierCurve:

    def __init__(self, *controls):
        """ base class for all bezier curves """
        self.controls = controls
        self.degree = len(self.controls) - 1

    def __call__(self, t):
        return sum([bernstein_poly_func(self.degree, i)(t) * self.controls[i] for i in range(0, self.degree + 1)])


class QuadBezier(BezierCurve):

    def __init__(self, p0, p1, p2):
        super().__init__(p0, p1, p2)


class CubicBezier(BezierCurve):

    def __init__(self, p0, p1, p2, p3):
        super().__init__(p0, p1, p2, p3)


def elevate_degree(curve: BezierCurve, degree: int) -> BezierCurve:
    old_degree = curve.degree
    new_degree = degree

    new_controls = []
    for i in range(0, old_degree + 1):
        new_control = (i/(old_degree + 1))*curve.controls[i-1] + (1 - (i/(old_degree + 1)))*curve.controls[i]
        new_controls.append(new_control)

    new_controls.append(curve.controls[-1])

    return BezierCurve(*new_controls)

