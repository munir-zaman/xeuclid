from xeuclid.utils.math import *

to_tuple = lambda vector: tuple(row_vector(vector))

class GObject(object):
    """Base class for all geometric objects """
    def __init__(self):
        pass


origin=np.array([[0],[0]])

def rotation_matrix(theta):
    """Returns the matrix,

    .. math::
        \\begin{bmatrix}
            \\cos(\\theta) & -\\sin(\\theta) \\\\
            \\sin(\\theta) &  \\cos(\\theta)
        \\end{bmatrix}

    Parameters
    ----------
    theta : float, int
        The value of :math:`\\theta`

    Returns
    -------
    np.ndarray
        Returns the matrix.

    Examples
    --------
    >>> from xeuclid import *
    >>> rotation_matrix(30)
    array([[ 0.8660254, -0.5      ],
           [ 0.5      ,  0.8660254]])
    """
    return np.array([[cos(theta),-1*sin(theta)],
                     [sin(theta),   cos(theta)]])

def rotate(point, center, angle):
    """Rotates ``point`` around ``center`` by an angle of ``angle``.

    Parameters
    ----------
    point: np.ndarray
        The point to rotate
    center: np.ndarray
        Center of rotation
    angle: float
        Angle of rotation

    Returns
    -------
    np.ndarray
        The rotated point

    Examples
    --------
    >>> from xeuclid import *
    >>> P = col_vector([1, 3])
    >>> O = col_vector([1, 1])
    >>> Q = rotate(P, O, 30)
    >>> Q
    array([[2.22044605e-16],
           [2.73205081e+00]])
    """
    point, center = col_vector(point), col_vector(center)
    return np.matmul(rotation_matrix(angle),point-center)+center

def dilate(point, center, factor):
    return center + (point - center) * factor

def reflect_about_point(point, center):
    return rotate(point, center, 180)

def reflect_about_line(point, line):
    center = line & line.perpendicular_line(point)
    return rotate(point, center, 180)


def angle(A,B,C) -> float:
    """Returns the angle :math:`\\angle ABC`. The equation used to compute the angle is,

    .. math::
        \\angle ABC =
        \\mathrm{atan2}\\left( {C_y - B_y},\\ {C_x - B_x} \\right) -
        \\mathrm{atan2}\\left( {A_y - B_y},\\ {A_x - B_x} \\right)
        \\pmod{360^{\\circ}}

    where :math:`A = \\left( A_x,\\ A_y \\right)`, :math:`B = \\left( B_x,\\ B_y \\right)` and 
    :math:`C = \\left( C_x,\\ C_y \\right)`

    Parameters
    ----------
    A : np.ndarray, tuple or list
        The point :math:`A`
    B : np.ndarray, tuple or list
        The point :math:`B`
    C : np.ndarray, tuple or list
        The point :math:`C`

    Returns
    -------
    float
        The angle :math:`\\angle ABC`

    Examples
    --------
    >>> from xeuclid import *
    >>> A = [1, 2]
    >>> B = [0, -2]
    >>> C = [-2, 2]
    >>> theta = angle(A, B, C)
    >>> print(theta)
    40.601294645004
    """
    a1,a2=row_vector(A)
    b1,b2=row_vector(B)
    c1,c2=row_vector(C)
    #x,y coordinates

    A1,A2=a1-b1,a2-b2
    B1,B2=0,0
    C1,C2=c1-b1,c2-b2
    #translate [x,y] -> [x-b1,y-b2]

    t1=atan2(A1,A2)
    t2=atan2(C1,C2)

    return round((t2-t1)%360, 12)

def angle_between_vectors(v1,v2):
    return angle(v1, origin, v2)

def angle_bisector(A,B,C):
    """Returns the angle bisector of :math:`\\angle ABC`

    Parameters
    ----------
    A: np.ndarray
        The point :math:`A`
    B: np.ndarray
        The point :math:`B`
    C: np.ndarray
        The point :math:`C`

    Returns
    -------
    line : Line
        The angle bisector
    """
    a1,a2=row_vector(A)
    b1,b2=row_vector(B)
    c1,c2=row_vector(C)
    #x,y coordinates

    A1,A2=a1-b1,a2-b2
    B1,B2=0,0
    C1,C2=c1-b1,c2-b2
    #translate [x,y] -> [x-b1,y-b2]

    t1=atan2(A1,A2)
    t2=atan2(C1,C2)
    T=(((t2-t1)%360)/2)+t1
    R=np.array([[cos(T)],
                [sin(T)]])

    line=Line(np.array([[B1],[B2]]),R)
    line=line+np.array([[b1],[b2]])

    return line

def intersection_line_line(line1,line2):
    """Returns the intersection point of ``line1`` and ``line2``.

    Parameters
    ----------
    line1 : Line
        Line 1
    line2 : Line
        Line 2

    Returns
    -------
    np.ndarray, None
        Returns the intersection point if it exists. Otherwise returns ``None``.

    """
    if not (line1 | line2):
        v1x,v1y=line1.v[0,0],line1.v[1,0]
        v2x,v2y=line2.v[0,0],line2.v[1,0]
        V=np.array([[v1x,-1*v2x],
                    [v1y,-1*v2y]])
        A=line2.A-line1.A
        T=system(V, A)
        t1=T[0,0]
        out=line1(t1)
    else:
        out=None

    return out

def intersection_line_segment(line,segment):
    int_=intersection_line_line(line, segment.line)
    if not isnone(int_):
        out=int_ if (int_ in segment) else None
    else:
        out=None
    return out

def intersection_line_ray(line,ray):
    int_=intersection_line_line(line, ray.line)
    if not isnone(int_):
        out=int_ if (int_ in ray) else None
    else:
        out=None
    return out

def intersection_segment_ray(segment, ray):
    int_=intersection_line_line(segment.line, ray.line)
    if not isnone(int_):
        out=int_ if (int_ in ray) and (int_ in segment) else None
    else:
        out=None
    return out

def intersection_ray_ray(ray1, ray2):
    int_=intersection_line_line(ray1.line, ray2.line)
    if not isnone(int_):
        out=int_ if (int_ in ray1) and (int_ in ray2) else None
    else:
        out=None

    return out

def intersection_segment_segment(segment1,segment2):
    int_=intersection_line_line(segment1.line, segment2.line)
    if not isnone(int_):
        out=int_ if (int_ in segment1) and (int_ in segment2) else None
    else:
        out=None
    return out


class Line(GObject):
    """A ``GObject`` for representing lines. 
    Lines are represented using the parametric equation,

    .. math::
        L(t) = (1-t)\\overline{A} + t\\overline{B}

    where :math:`A` and :math:`B` are two points that are on the line :math:`L(t)`.

    Attributes
    ----------
    A : np.ndarray
        The vector :math:`\\overline{A}`
    B : np.ndarray
        The vector :math:`\\overline{B}`
    v : np.ndarray
        The vector :math:`\\overline{v} = \\overline{B} - \\overline{A}`

    Examples
    --------
    >>> from xeuclid import *
    >>> l1 = Line([0, 0], [1, 2]) # line through points (0,0) and (1,2)
    >>> l1
    [0, 0] +[1, 2]*t
    >>> l1(1.5) # l1(t) = (1 - t)*(0,0) + t*(1,2)
    array([[1.5],
           [3. ]])
    >>> newl1 = l1 + col_vector([1,1]) # translates the line by (1,1)
    >>> newl1
    [1, 1] +[1, 2]*t
    >>> l2 = Line([1, 3], [-1, -2])
    >>> l2 | l1 # checks if l1 and l2 are parallel
    False
    >>> l1 & l2 # the intersection point of l1 and l2
    array([[-1.],
           [-2.]])
    >>> l1 ^ l2 # checks if l1 and l2 are perpendicular
    False
    """
    def __init__(self,A,B):
        self._A = col_vector(A)
        self._B = col_vector(B)
        self._update_attrs()

        if isclose(self.v[0,0], 0) and isclose(self.v[1,0], 0):
            raise ValueError("`self.A` and `self.B` cannot be equal")

    def _update_attrs(self):
        self._v = self.B - self.A

    @property
    def v(self):
        return self._v

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @A.setter
    def A(self, value):
        value = col_vector(value)
        self._A = value
        self._update_attrs()

    @B.setter
    def B(self, value):
        value = col_vector(value)
        self._B = value
        self._update_attrs()

    @property
    def type(self):
        return 'line'

    def __call__(self,t):
        """Evaluates the value of the function,

        .. math::
            L(t) = \\overline{A} + \\overline{v}t

        Parameters
        ----------
        t : float,int
            The value of :math:``t.

        Returns
        -------
        float,int
            The evaluated value of :math:`L(t)`
        """
        return (self.A+self.v*t)

    def normt(self, t):
        """Evaluates the value of the function,

        .. math::
            L'(t) = \\overline{A} + \\frac{\\overline{v}}{|v|}t

        Parameters
        ----------
        t : float, int
            The value of :math:`t`.

        Returns
        -------
        float,int
            The evaluated value of :math:`L'(t)`
        """
        return self.A + norm(self.v) * t

    def __repr__(self):
        return f"[{self.A[0,0]}, {self.A[1,0]}] +[{self.v[0,0]}, {self.v[1,0]}]*t"

    def __str__(self):
        return str(f"[{self.A[0,0]}, {self.A[1,0]}] +[{self.v[0,0]}, {self.v[1,0]}]*t")

    def __contains__(self, point) -> bool:
        """Checks if ``point`` is on the line.

        Parameters
        ----------
        point : np.ndarray
            The point

        Returns
        -------
        bool
            returns ``True`` if ``point`` is on line else returns ``False``
        """
        return (not self.inv(point) is None)

    def inv(self, point):
        """Returns the value :math:`L^{-1}(\\overline{p})` where 
        :math:`\\overline{p}` is the point ``point``.

        Parameters
        ----------
        point : np.ndarray
            The point :math:`\\overline{p}`.

        Returns
        -------
        float, int
            The evaluated value :math:`L^{-1}(\\overline{p})`.
        """
        out = None
        point = col_vector(point)
        P = row_vector(point - self.A)
        V = row_vector(self.v)
        for i in range(0, len(V)):
            if not isclose(V[i], 0):
                t = P[i]/V[i]
        out = t if np.allclose(P, V*t, rtol=0) else None
        return out

    def fx(self, x):
        """Returns the :math:`y` value at :math:`x`.

        Parameters
        ----------
        x : float, int
            :math:`x`

        Returns
        -------
        float, int, None
            :math:`y`
        """
        vx, vy = row_vector(self.v)
        Ax, Ay = row_vector(self.A)

        if not isclose(vx, 0):
            y = Ay + vy * ((x - Ax)/vx)
        elif isclose(vx, 0) and isclose(x - Ax, 0):
            y = Ay
        else:
            y = None

        return y

    def fy(self, y):
        """Returns the :math:`x` value at :math:`y`.

        Parameters
        ----------
        y : float, int
            :math:`y`

        Returns
        -------
        float, int, None
            :math:`x`
        """
        vx, vy = row_vector(self.v)
        Ax, Ay = row_vector(self.A)

        if not isclose(vy, 0):
            x = Ax + vx * ((y - Ay)/vy)
        elif isclose(vy, 0) and isclose(y - Ay, 0):
            x = Ax
        else:
            x = None

        return x

    def __eq__(self, line):
        """Checks if ``self`` and ``line`` are equal."""
        out = False
        if self | line:
            A, B = self.A, self.B
            out = (A in line) and (B in line)
        return out

    def __add__(self, vector):
        return Line(self.A+vector, self.B+vector)

    def __radd__(self, vector):
        return Line(self.A+vector, self.B+vector)

    def __sub__(self, vector):
        return Line(self.A-vector, self.B-vector)

    def __rsub__(self, vector):
        return Line(vector-self.A, vector-self.B)

    def __mul__(self, value):
        return Line(value*self.A, value*self.B)

    def __rmul__(self, value):
        return Line(value*self.A, value*self.B)

    def __truediv__(self, value):
        return Line(self.A/value, self.B/value)

    def matmul(self, matrix):
        return Line(matmul(matrix, self.A), matmul(matrix, self.B))

    def intersection(self,obj):
        if obj.type=="line":
            out=intersection_line_line(self, obj)
        elif obj.type=="segment":
            out=intersection_line_segment(self, obj)
        elif obj.type=="ray":
            out=intersection_line_ray(self, obj)
        elif obj.type=="circle":
            out=intersection_line_circle(self, obj)
        elif obj.type=="polygon":
            out=intersection_line_poly(self, obj)
        else:
            out=None

        return out

    def rotate(self, point, angle):
        """Rotates the line around ``point`` by an angle of ``angle``.

        Parameters
        ----------
        point : np.ndarray
            The center of rotation
        angle : float, int
            The angle of rotation

        Returns
        -------
        Line
            Returns the rotated line.
        """
        point = col_vector(point)
        A_,B_=rotate(self.A, point, angle),rotate(self.B, point, angle)
        return Line(A_, B_)

    def reflect(self, center):
        out=None
        if isinstance(center, GObject):
            if center.type=="line":
                out=Line(reflect_about_line(self.A, center), reflect_about_line(self.B, center))

        elif isinstance(center, np.ndarray):
            out=Line(reflect_about_point(self.A, center), reflect_about_point(self.B, center))

        return out

    def parallel_line(self,P):
        """Returns the parallel line through point ``P``.

        Parameters
        ----------
        P : np.ndarray
            The point through which the parallel line will go.

        Returns
        -------
        Line
            Returns the parallel line through point ``P``.
        """
        P = col_vector(P)
        return Line(P, self.v+P)

    def perpendicular_line(self,P):
        """Returns the perpendicular line through point ``P``.

        Parameters
        ----------
        P : np.ndarray
            The point through which the perpendicular line will go.

        Returns
        -------
        Line
            The perpendicular line through point ``P``.
        """
        P = col_vector(P)
        return Line(P, rotate(self.v,origin,90)+P)

    def __or__(self,line):
        """Checks if ``self`` and `line`` are parallel.

        Parameters
        ----------
        line : Line
            The other line

        Returns
        -------
        bool
            Returns ``True`` if ``self`` and ``line`` are parallel line. 
            Otherwise returns ``False``.
        """
        v1x,v1y=self.v[0,0],self.v[1,0]
        v2x,v2y=line.v[0,0],line.v[1,0]
        V=np.array([[v1x,-1*v2x],
                    [v1y,-1*v2y]])

        return isclose(det(V),0.0)

    def __and__(self,obj):
        """Returns the intersection point of ``self`` and ``obj``

        Parameters
        ----------
        obj : GObject
            The other geometric object.

        """
        return self.intersection(obj)

    def __xor__(self,line):
        """Checks if ``self`` and ``line`` are perpendicular.

        Parameters
        ----------
        line : Line
            The other line.

        Returns
        -------
        bool
            Returns ``True`` if ``self`` and ``line`` are perpendicular lines. 
            Otherwise returns ``False``.
        """
        v1=row_vector(self.v)
        v2=row_vector(line.v)
        return isclose(np.dot(v1,v2),0)

    def isperp(self,line):
        """Checks if ``self`` and ``line`` are perpendicular.

        Parameters
        ----------
        line : Line
            The other line.

        Returns
        -------
        bool
            Returns ``True`` if ``self`` and ``line`` are perpendicular lines. 
            Otherwise returns ``False``.
        """
        return self ^ line

    def isparallel(self,line):
        """Checks if ``self`` and `line`` are parallel.

        Parameters
        ----------
        line : Line
            The other line.

        Returns
        -------
        bool
            Returns ``True`` if ``self`` and ``line`` are parallel line. 
            Otherwise returns ``False``.
        """
        return self | line

    def angle(self,line):
        """Returns the angle between ``self`` and ``line``.

        Parameters
        ----------
        line : Line
            The other line.

        Returns
        -------
        float, int
            The value of the angle.
        """
        v1=self.v
        v2=line.v
        theta=angle_between_vectors(v1, v2)%180
        return theta

    def distance(self, obj):
        """Returns the perpendicular distance between ``self`` and ``obj``.

        Parameters
        ----------
        obj : np.ndarray, Line
            Point or a line

        Returns
        -------
        float, int, None
            The perpendicular distance between ``self`` and ``obj``
        """
        out=None
        if isinstance(obj, (np.ndarray, tuple, list)):
            perp_line=self.perpendicular_line(obj)
            A=self.intersection(perp_line)
            out=dist(A, obj)

        elif isinstance(obj, Line) and (obj | self):
            perp_line = self.perpendicular_line(obj.A)
            A=self.intersection(perp_line)
            out=dist(A, obj.A)

        return out


def param_to_impl_line(line : Line) -> tuple:
    """Returns the tuple :math:(A, B, C)`` where :math:`A, B, C` 
    are constants such that all the points :math:`(x,y)` that satisfy 
    the equation,

    .. math::
        Ax + By + C = 0
    are on the line ``line``.

    Parameters
    ----------
    line : Line
        The line whose implicit form you want to find

    Returns
    -------
    tuple
        The tuple :math:`(A, B, C)`.
    """
    Ax,Ay=row_vector(line.A)
    vx,vy=row_vector(line.v)
    return (-1*vy, vx, vy*Ax-vx*Ay)

def impl_to_param_line(coeff):
    a, b, c = coeff
    vy, vx = -a, b

    V1 = np.array([[-c/a], [0]]) if not isclose(a, 0) else None
    V2 = np.array([[0], [-c/b]]) if not isclose(b, 0) else None

    if isnone(V1):
        A=V2
    elif isnone(V2):
        A=V1
    else:
        A=V1
    V=col_vector([vx, vy])

    return Line(A, A + V)

x_vect=np.array([[1],[0]])
y_vect=np.array([[0],[1]])
zero_vect=np.array([[0],[0]])
y_axis=Line(zero_vect,y_vect)
x_axis=Line(zero_vect,x_vect)

def collinear(*points: np.ndarray):
    """Returns the line through the points ``points`` if all the points 
    in ``points`` are collinear. Otherwise returns ``None``.

    Parameters
    ----------
    points : np.ndarray
        The points

    Returns
    -------
    Line, None
        Returns a ``Line`` if the points are collinear. Otherwise returns ``None``.
    """
    A, B = points[0],points[1]
    AB = Line(A, B)
    out=True

    for p in points:
        out=out and (p in AB)
        if not out:
            break

    if out:
        line=AB
    else:
        line=None

    return line

def concurrent(*lines: Line):
    """ checks if the given set of lines are concurrent """
    ints=[]
    out=True

    for l in range(1,len(lines)):

        if not isinstance(lines[l],Line):
            lines[l]=points_to_line(*lines[l])
        if not isinstance(lines[l-1],Line):
            lines[l]=points_to_line(*lines[l-1])

        int_=lines[l].intersection(lines[l-1])
        ints.append(int_)

        if not len(ints)<2:
            out=out and (ints[-1]==ints[-2])

        if not out:
            break

    if not out:
        ints[-1]=None

    print(out)
    return ints[-1]


class Segment(GObject):
    def __init__(self, A, B):
        self._A = col_vector(A)
        self._B = col_vector(B)
        self._update_attrs()

    def _update_attrs(self):
        self._v = self.B - self.A
        self._line = Line(self.A, self.B)
        self._mid = mid(self.A, self.B)
        self._length = dist(self.A, self.B)

    @property
    def type(self):
        return 'segment'

    @property
    def v(self):
        return self._v

    @property
    def line(self):
        return self._line

    @property
    def mid(self):
        return self._mid

    @property
    def length(self):
        return self._length

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @A.setter
    def A(self, value):
        value = col_vector(value)
        self._A = value
        self._update_attrs()

    @B.setter
    def B(self, value):
        value = col_vector(value)
        self._B = value
        self._update_attrs()

    def __call__(self, t):
        return (self.A+ self.v*t)

    def normt(self, t):
        return self.A + norm(self.v) * t

    def __repr__(self):
        return f"[{self.A[0,0]}, {self.A[1,0]}] +[{self.v[0,0]}, {self.v[1,0]}]*t, t in [0, 1]"

    def __str__(self):
        return str(f"[{self.A[0,0]}, {self.A[1,0]}] +[{self.v[0,0]}, {self.v[1,0]}]*t, t in [0, 1]")

    def __contains__(self, point):
        return (not self.inv(point) is None)

    def __len__(self):
        return self.length

    def inv(self,P):
        out_=self.line.inv(P)
        if out_!=None:
            out= out_ if (0 <= round(out_, 8) <= 1) else None
        else:
            out=None

        return out

    def __eq__(self, segment):
        out = False
        if self.line == segment.line:
            mid1 = (self.A + self.B)/2
            mid2 = (segment.A + segment.B)/2
            same_mid = np.allclose(mid1, mid2, rtol=0, atol=1e-8)
            # check if they have the same midpoint
            common_end = ( np.allclose(self.A, segment.A, rtol=0, atol=1e-8)
                        or np.allclose(self.A, segment.B, rtol=0, atol=1e-8) )
            # check if they share at least one common endpoint 
            out = same_mid and common_end
            # two segments will be the same
            # if and only if they have the same midpoint and 
            # share at least one common endpoint.
        return out

    def __add__(self,vector):
        return Segment(self.A+vector, self.B+vector)

    def __radd__(self,vector):
        return Segment(self.A+vector, self.B+vector)

    def __sub__(self,vector):
        return Segment(self.A-vector, self.B-vector)

    def __rsub__(self,vector):
        return Segment(vector-self.A, vector-self.B)

    def __mul__(self,value):
        return Segment(value*self.A, value*self.B)

    def __rmul__(self,value):
        return Segment(value*self.A, value*self.B)

    def __truediv__(self,value):
        return Segment(self.A/value, self.B/value)

    def intersection(self,obj):
        if obj.type=="line":
            out=intersection_line_segment(obj, self)
        elif obj.type=="segment":
            out=intersection_segment_segment(self, obj)
        elif obj.type=="ray":
            out=intersection_segment_ray(self, obj)
        elif obj.type=="circle":
            out=intersection_ray_circle(self, obj)
        elif obj.type=="polygon":
            out=intersection_segment_poly(self, obj)
        else:
            out=None

        return out

    def __and__(self, obj):
        return self.intersection(obj)

    def rotate(self, point, angle):
        A_, B_= rotate(self.A, point, angle), rotate(self.B, point, angle)
        return Segment(A_, B_)



class Ray(GObject):
    def __init__(self, A, B):
        self._A = col_vector(A)
        self._B = col_vector(B)
        self._update_attrs()

    def _update_attrs(self):
        self._v = self.B - self.A
        self._line = Line(self.A, self.B)

    @property
    def line(self):
        return self._line

    @property
    def v(self):
        return self._v

    @property
    def type(self):
        return 'ray'

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @A.setter
    def A(self, value):
        value = col_vector(value)
        self._A = value
        self._update_attrs()

    @B.setter
    def B(self, value):
        value = col_vector(value)
        self._B = value
        self._update_attrs()

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

    def __eq__(self, ray):
        out = False
        if self.line == self.ray:
            out = np.allclose(self.A, ray.A, rtol=0, atol=1e-8)
        return out

    def __add__(self,vector):
        return Ray(self.A+vector, self.B+vector)

    def __radd__(self,vector):
        return Ray(self.A+vector, self.B+vector)

    def __sub__(self,vector):
        return Ray(self.A-vector, self.B-vector)

    def __rsub__(self,vector):
        return Ray(vector-self.A, vector-self.B)

    def __mul__(self,value):
        return Ray(value*self.A, value*self.B)

    def __rmul__(self,value):
        return Ray(value*self.A, value*self.B)

    def __truediv__(self,value):
        return Ray(self.A/value, self.B/value)

    def intersection(self,obj):
        if obj.type=="line":
            out=intersection_line_ray(obj, self)
        elif obj.type=="segment":
            out=intersection_segment_ray(obj, self)
        elif obj.type=="ray":
            out=intersection_ray_ray(self, obj)
        elif obj.type=="circle":
            out=intersection_segment_circle(self, obj)
        elif obj.type=="polygon":
            out=intersection_ray_poly(self, obj)
        else:
            out=None

        return out

    def __and__(self, obj):
        return self.intersection(obj)

    def rotate(self, point, angle):
        A_,B_=rotate(self.A, point, angle), rotate(self.B, point, angle)
        return Ray(A_, B_)


def intersection_line_poly(line, poly):
    A, B, C, D = poly.bbox[::]
    AB = Segment(A, B)
    BC = Segment(B, C)
    CD = Segment(C, D)
    DA = Segment(D, A)

    # first check if line intersects the bounding box of the polygon
    bbox_int = False
    for segment in [AB, BC, CD, DA]:
        if not (line & segment) is None:
            bbox_int = True
            break
    out = []
    if bbox_int:
        # calc intersection
        verts = poly.vertices[::]
        verts.append(poly.vertices[0])

        for i in range(0, len(verts)-1):
            edge = Segment(verts[i], verts[i+1])
            intersection = line & edge
            if not intersection is None:
                out.append(intersection)
    return out

def intersection_segment_poly(segment, poly):
    ints_ = intersection_line_poly(segment.line, poly)
    out = [point for point in ints_ if (point in segment)]
    return out

def intersection_ray_poly(ray, poly):
    ints_ = intersection_line_poly(ray.line, poly)
    out = [point for point in ints_ if (point in ray)]
    return out

def intersection_circle_poly(circle, poly):
    verts = poly.vertices[::]
    verts.append(poly.vertices[0])

    for i in range(0, len(verts)-1):
        edge = Segment(verts[i], verts[i+1])
        intersection = circle & edge
        if (not intersection is None) and (not intersection==[]):
            out += intersection

    return out

def intersection_poly_poly(poly1, poly2):
    out = []

    verts1= poly1.vertices[::]
    verts2= poly2.vertices[::]

    verts1.append(poly1.vertices[0])
    verts2.append(poly2.vertices[0])

    for r in range(0, len(verts1) - 1):
        edge1 = Segment(verts1[r], verts1[r+1])
        for s in range(0, len(verts2) - 1):
            edge2 = Segment(verts2[s], verts2[s+1])
            ints = (edge1 & edge2)
            if not ints is None:
                out.append(ints)

    return out


class Polygon(GObject):

    def __init__(self, *vertices):
        self.vertices=list(vertices)

    @property
    def area(self):
        return self.get_area()

    @property
    def bbox(self):
        return self.get_bbox()

    @property
    def type(self):
        return 'polygon'

    def isinside(self, point) -> bool:
        """
        Parameters
        ----------
        point : np.ndarray, list or tuple


        Returns
        -------
        bool
            Returns ``True`` if ``point`` is inside the Polygon. Returns ``False`` otherwise.

        Examples
        --------
        """
        A = col_vector(point)
        B = A + 1
        ray = Ray(A, B)
        ints = intersection_ray_poly(ray, self)
        return True if len(ints)%2==1 else False

    def get_area(self):
        V=self.vertices[::]
        V.append(self.vertices[0])
        out = (0.5)*sum([ np.linalg.det(np.array([[V[i][0,0], V[i+1][0,0]],
                                                  [V[i][1,0], V[i+1][1,0]]])) for i in range(len(V)-1) ])
        return out

    def get_bbox(self):
        """ returns the vertices of the bounding rectangle of the polygon
            returns: list of np.ndarray
        """
        vertices = self.vertices[::]

        max_px = max(vertices, key=lambda p: p[0,0])
        min_px = min(vertices, key=lambda p: p[0,0])
        max_py = max(vertices, key=lambda p: p[1,0])
        min_py = min(vertices, key=lambda p: p[1,0])

        down_line = y_axis.perpendicular_line(min_py)
        up_line = y_axis.perpendicular_line(max_py)
        left_line = x_axis.perpendicular_line(min_px)
        right_line = x_axis.perpendicular_line(max_px)

        A, B, C, D = down_line & left_line, down_line & right_line, right_line & up_line, left_line & up_line

        return [A, B, C, D]

    def rotate(self, point, angle):
        vertices=[rotate(p, point, angle) for p in self.vertices]
        return Polygon(vertices)

    def intersection(self,obj):
        if obj.type=="line":
            out=intersection_line_poly(obj, self)
        elif obj.type=="segment":
            out=intersection_segment_poly(obj, self)
        elif obj.type=="ray":
            out=intersection_ray_poly(obj, self)
        elif obj.type=="circle":
            out=intersection_circle_poly(obj, self)
        elif obj.type=="polygon":
            out=intersection_poly_poly(self, obj)
        else:
            out=None

        return out

    def __and__(self, obj):
        return self.intersection(obj)


def convex_hull(*points) -> list:
    """Returns the convex hull of ``points``.

    Parameters
    ----------
    *points : np.ndarray, list or tuple
        The points whose convex hull you want to find.

    Returns
    -------
    list
        The list of all the points on the convex hull.

    Examples
    --------
    >>> from xeuclid import *
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> points = [col_vector(point) for point in rng.random((100, 2))*10]
    >>> print(points)
    [array([[2.00756707],
           [7.91470412]]), array([[1.01901632],
           [9.43917478]]), array([[8.63472305],
           ...,
           [9.55533233]]), array([[7.87207801],
           [5.4060051 ]]), array([[6.02948716],
           [5.29350243]])]
    >>> hull = convex_hull(*points)
    >>> print(hull)
    [array([[5.69873368],
           [0.07473919]]), array([[9.46265076],
           [0.08784139]]), array([[9.89536912],
           ...,
           [0.76708439]]), array([[3.07070786],
           [0.08520319]])]
    """
    if any([type(obj)!=np.ndarray for obj in points]):
        points = [col_vector(obj_) for obj_ in points]

    #find the point `P` with min y - coordinate
    P = min(points, key=lambda p: p[1,0])
    # get the list of remaining points `Q`
    Q = [point for point in points if not np.all(point == P)]
    # sort the remaining points by polar 
    # angle in counter clockwise order around `P`
    sorted_Q = sorted(Q, key= lambda p: atan2(p[0,0] - P[0,0], p[1,0] - P[1,0]))
    hull = [P, sorted_Q[0], sorted_Q[1]]
    # perform Graham-Scan
    for i in range(2, len(sorted_Q)):
        while np.cross(row_vector(hull[-2] - hull[-1]), row_vector(sorted_Q[i] - hull[-1])) > 0:
            hull.pop()
        hull.append(sorted_Q[i])

    return hull

def intersection_line_circle(line, circle):
    R = circle.radius
    center = circle.center
    Ax, Ay = row_vector(line.A - center)
    vx, vy = row_vector(line.v)
    out = quad((vx**2 + vy**2), 2*(Ax*vx + Ay*vy), (Ax**2 + Ay**2 - R**2))
    return [line(t) for t in out]

def intersection_segment_circle(segment, circle):
    int_=intersection_line_circle(segment.line, circle)
    out=[point for point in int_ if (point in segment) and (point in circle)]
    return out

def intersection_ray_circle(ray, circle):
    int_=intersection_line_circle(ray.line, circle)
    out=[point for point in int_ if (point in ray) and (point in circle)]
    return out

def intersection_circle_circle(circle1, circle2):
    d=dist(circle1.center, circle2.center)
    r1,r2= circle1.radius, circle2.radius

    out=[]
    if round(d, 8) <= r2 + r1:
        theta= acos((r1**2 + d**2 - r2**2)/(2*r1*d))

        c1c2=circle1.center - circle2.center
        x_angle=angle_between_vectors(x_vect, c1c2)

        R1=Ray(circle1.center, circle2.center).rotate(circle1.center, theta)
        R2=Ray(circle1.center, circle2.center).rotate(circle1.center, -theta)

        out=intersection_ray_circle(R1, circle1) + intersection_ray_circle(R2, circle1)
    out = [point for point in out if (point in circle1) and (point in circle2)]
    return get_rid_of_multiple_points(out)


class Circle(GObject):

    def __init__(self, center, radius):
        self.center=center
        self.radius=radius

    @property
    def type(self):
        return 'circle'

    def __repr__(self):
        return f"[{self.center[0,0]}, {self.center[1,0]}] + [cos(x), sin(x)]* {self.radius}"

    def __call__(self, theta):
        return self.center+ self.radius* np.array([[cos(theta)], [sin(theta)]])

    def __contains__(self, point):
        Px, Py=row_vector(point - self.center)
        r=self.radius
        return isclose( (Px)**2 + (Py)**2 , r**2 )

    def inv(self, point):
        out=None
        if point in self:
            Px, Py = row_vector(point - self.center)
            out=atan2(Px, Py)
        return out

    def power(self, point):
        return dist(point, self.center)**2 - self.radius**2

    def tangent(self, point):
        out=[]
        if point in self:
            line=Line(point, self.center)
            perp_line=line.perpendicular_line(point)
            out=[perp_line]
        elif round(self.power(point), 8) < 0:
            out=[]
        else:
            ABmid=mid(self.center, point)
            Circ=Circle(ABmid, dist(ABmid, self.center))
            int_=intersection_circle_circle(self, Circ)
            out=[Line(point, q) for q in int_]
        return out

    def intersection(self,obj):
        if obj.type=="line":
            out=intersection_line_circle(obj, self)
        elif obj.type=="segment":
            out=intersection_segment_circle(obj, self)
        elif obj.type=="ray":
            out=intersection_ray_circle(obj, self)
        elif obj.type=="circle":
            out=intersection_circle_circle(obj, self)
        else:
            out=None

        return out

    def __and__(self, obj):
        return self.intersection(obj)

    def rotate(self, point, angle):
        return Circle(rotate(self.center, point, angle), self.radius)



def common_tangents(circle1 ,circle2):
    out=[]

    if circle2.radius > circle1.radius:
        c2=circle2
        c1=circle1
    else:
        c2=circle1
        c1=circle2

    r2=c2.radius
    r1=c1.radius
    d=dist(c1.center, c2.center)

    if isclose(circle1.radius,circle2.radius):
        c1c2Line= Line(circle1.center, circle2.center)
        perp_c1c2Line=c1c2Line.perpendicular_line(circle1.center)
        int_=intersection_line_circle(perp_c1c2Line, circle1)
        l1=perp_c1c2Line.perpendicular_line(int_[0])
        l2=perp_c1c2Line.perpendicular_line(int_[1])
        out= out+ [l1, l2]

    elif rnd(d) >= (r2-r1):
        b1=((r1)/(r2-r1))* d
        c1c2=c1.center - c2.center
        x_angle=angle_between_vectors(x_vect, c1c2)
        P=polar(b1, x_angle, center=c1.center)
        out= out + c1.tangent(P)

    if rnd(d) >= (r2+r1):
        B1= ((r1)/(r2+r1)) * d
        c2c1=c2.center - c1.center
        X_angle=angle_between_vectors(x_vect, c2c1)
        P_=polar(B1, X_angle, center=c1.center)
        out= out+ c1.tangent(P_)

    return out

def points_to_circle(point1, point2, point3):
    p1, p2, p3= point1, point2, point3
    p1p2=Segment(p1, p2)
    p1p3=Segment(p1, p3)
    p1p2mid=p1p2.mid
    p1p3mid=p1p3.mid

    p1p2_perp=p1p2.line.perpendicular_line(p1p2mid)
    p1p3_perp=p1p3.line.perpendicular_line(p1p3mid)

    center=p1p2_perp & p1p3_perp
    radius=dist(center, p1)

    return Circle(center, radius)

