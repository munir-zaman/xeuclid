from euclid_math import *


class GObject(object):
    """ base class for geometric objects """
    def __init__(self):
        pass


origin=np.array([[0],[0]])

rotation_matrix=lambda theta: np.array([[cos(theta),-1*sin(theta)],
                                        [sin(theta),   cos(theta)]])

rotate=lambda A,B,theta: np.matmul(rotation_matrix(theta),A-B)+B

def angle(A,B,C):
    """ returns the angle <ABC
        A,B,C : np.array([[x],
                          [y]])
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
    """ returns the angle bisector of angle <ABC"""
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

    if not (line1 | line2):
        # if the lines are not parallel
        # then, they intersect

        v1x,v1y=line1.v[0,0],line1.v[1,0]
        v2x,v2y=line2.v[0,0],line2.v[1,0]
        V=np.array([[v1x,-1*v2x],[v1y,-1*v2y]])

        A=line2.A-line1.A
        T=system(V, A)
        #print(T)
        t1=T[0,0]
        out=line1(t1)
    else:
        #print("The Lines are Parallel. Returned None")
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
    def __init__(self,A,B):
        self.A=A
        self.B=B
        self.v=self.B-self.A
        # v is the velocity vector
        self.type='line'

        if isclose(self.v[0,0], 0) and isclose(self.v[1,0], 0):
            print('WARNING V=[0,0]')

    def __call__(self,t):
        return (self.A+self.v*t)

    def normt(self, t):
        return self.A + norm(self.v) * t

    def __repr__(self):
        return f"[{self.A[0,0]}, {self.A[1,0]}] +[{self.v[0,0]}, {self.v[1,0]}]*t"

    def __str__(self):
        return str(f"[{self.A[0,0]}, {self.A[1,0]}] +[{self.v[0,0]}, {self.v[1,0]}]*t")

    def __contains__(self,P):
        return self.inv(P)!=None

    def inv(self,P):
        Ax,Ay=row_vector(self.A)
        vx,vy=row_vector(self.v)
        Px,Py=row_vector(P)

        if (not isclose(vx,0)) and (not isclose(vy,0)):
            t1=(Px-Ax)/vx
            t2=(Py-Ay)/vy
        elif (isclose(vx,0)):
            t2=(Py-Ay)/vy
            t1=t2 if isclose(Px,Ax) else None
        elif (isclose(vy,0)):
            t1=(Px-Ax)/vx
            t2=t1 if isclose(Py,Ay) else None

        if not (isnone(t1) or isnone(t2)):
            out=t1 if isclose(t1,t2) else None
        else:
            out=None
        return out

    def __add__(self,vector):
        return Line(self.A+vector, self.B+vector)

    def __radd__(self,vector):
        return Line(self.A+vector, self.B+vector)

    def __sub__(self,vector):
        return Line(self.A-vector, self.B-vector)

    def __rsub__(self,vector):
        return Line(vector-self.A, vector-self.B)

    def __mul__(self,value):
        return Line(value*self.A, value*self.B)

    def __rmul__(self,value):
        return Line(value*self.A, value*self.B)

    def __truediv__(self,value):
        return Line(self.A/value, self.B/value)

    def matmul(self,matrix):
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
        else:
            out=None

        return out

    def rotate(self, point, angle):
        A_,B_=rotate(self.A, point, angle),rotate(self.B, point, angle)
        return Line(A_, B_)

    def parallel_line(self,P):
        return Line(P, self.v+P)

    def perpendicular_line(self,P):
        return Line(P, rotate(self.v,origin,90)+P)

    def __or__(self,line):
        """ Checks if `self` and `line` are parallel """
        v1x,v1y=self.v[0,0],self.v[1,0]
        v2x,v2y=line.v[0,0],line.v[1,0]
        V=np.array([[v1x,-1*v2x],
                    [v1y,-1*v2y]])

        return isclose(det(V),0.0)

    def __and__(self,obj):
        """Returns the intersection point of `self` and `line`"""
        return self.intersection(obj)

    def __xor__(self,line):
        """ Checks if `self` and `line` are perpendicular """
        v1=row_vector(self.v)
        v2=row_vector(line.v)
        return isclose(np.dot(v1,v2),0)

    def isperp(self,line):
        return self ^ line

    def isparallel(self,line):
        return self | line

    def angle(self,line):
        v1=self.v
        v2=line.v
        theta=angle_between_vectors(v1, v2)%180
        return theta

    def distance(self, obj):
        out=None
        if isinstance(obj, np.ndarray):
            perp_line=self.perpendicular_line(obj)
            A=self.intersection(perp_line)
            out=dist(A, obj)

        elif isinstance(obj, Line) and (obj | self):
            perp_line = self.perpendicular_line(obj.A)
            A=self.intersection(perp_line)
            out=dist(A, obj.A)

        return out


def param_to_impl_line(line):
    Ax,Ay=row_vector(line.A)
    vx,vy=row_vector(line.v)
    print(f"{-1*vy}*x+ {vx}*y+ {vy*Ax-vx*Ay}= 0")
    return np.array([-1*vy, vx, vy*Ax-vx*Ay])

def impl_to_param_line(line):
    return NotImplemented

def dist(p1,p2):
    """ Return distance between p1 and p2 """
    return mth.sqrt((p1[0,0]-p2[0,0])**2+(p1[1,0]-p2[1,0])**2)

def mid(p1,p2):
    """ Return midpoint of p1 and p2 """
    return col_vector([(p1[0,0]+p2[0,0])/2,(p1[1,0]+p2[1,0])/2])


x_vect=np.array([[1],[0]])
y_vect=np.array([[0],[1]])
zero_vect=np.array([[0],[0]])
y_axis=Line(zero_vect,y_vect)
x_axis=Line(zero_vect,x_vect)

def collinear(*points):
    """ collinear(*points)

        checks if the given set of points are collinear
        If,the given set of points are collinear:
            returns the line through the given set of points.
        Else,
            returns None
    """
    A,B=points[0],points[1]
    AB=points_to_line(A,B)
    out=True

    for p in points:
        out=out and (p in AB)
        if not out:
            break

    if out:
        print(True)
        line=AB
    else:
        print(False)
        line=None

    return line

def concurrent(*lines):
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
        self.A=A
        self.B=B
        self.v=self.B-self.A
        self.type="segment"
        self.line=Line(self.A, self.B)

        self.mid=mid(self.A, self.B)
        self.length=dist(self.A, self.B)

    def __call__(self, t):
        return (self.A+ self.v*t)

    def normt(self, t):
        return self.A + norm(self.v) * t

    def __repr__(self):
        return f"[{self.A[0,0]}, {self.A[1,0]}] +[{self.v[0,0]}, {self.v[1,0]}]*t, t in [0, 1]"

    def __str__(self):
        return str(f"[{self.A[0,0]}, {self.A[1,0]}] +[{self.v[0,0]}, {self.v[1,0]}]*t, t in [0, 1]")

    def __contains__(self,P):
        return self.inv(P)!=None

    def inv(self,P):
        out_=self.line.inv(P)
        if out_!=None:
            out= out_ if (0 <= round(out_, 8) <= 1) else None
        else:
            out=None

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
        else:
            out=None

        return out

    def rotate(self, point, angle):
        A_, B_= rotate(self.A, point, angle), rotate(self.B, point, angle)
        return Segment(A_, B_)



class Ray(GObject):
    def __init__(self, A, B):
        self.A=A
        self.B=B
        self.v=self.B-self.A
        self.type="ray"
        self.line=Line(self.A, self.B)

    def __call__(self, t):
        return self.A+ self.v*t

    def normt(self, t):
        return self.A + norm(self.v) * t
        
    def __repr__(self):
        return f"[{self.A[0,0]}, {self.A[1,0]}] +[{self.v[0,0]}, {self.v[1,0]}]*t, t >= 0"

    def __str__(self):
        return str(f"[{self.A[0,0]}, {self.A[1,0]}] +[{self.v[0,0]}, {self.v[1,0]}]*t, t >= 0")

    def __contains__(self, point):
        return self.inv(point)!=None

    def inv(self, point):
        out_=self.line.inv(point)
        if out_!=None:
            out= out_ if (round(out_, 8) >= 0) else None
        else:
            out=None

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
        else:
            out=None

        return out

    def rotate(self, point, angle):
        A_,B_=rotate(self.A, point, angle), rotate(self.B, point, angle)
        return Ray(A_, B_)


class Polygon(GObject):

    def __init__(self,*vertices):

        self.vertices=list(vertices)
        self.type="polygon"

    def area(self):
        V=self.vertices[::]
        V.append(self.vertices[0])
        return (0.5)*sum([ np.linalg.det(np.array([[V[i][0,0], V[i+1][0,0]],
                                                   [V[i][1,0], V[i+1][1,0]]])) for i in range(len(V)-1) ])

    def rotate(self, point, angle):
        vertices=[rotate(p, point, angle) for p in self.vertices]
        return Polygon(vertices)



def intersection_line_circle(line, circle):
    D=line.distance(circle.center)
    out=[]
    if round(D, 8) <= circle.radius:
        dx, dy= row_vector(circle.center - line.A)
        vx, vy= row_vector(line.v)
        r= circle.radius
        V=sqrt(vx**2 + vy**2)

        a=(r**2)*(V**2)
        b=2*vy*r*(vy*dx - vx*dy)
        c=(vy*dx - vx*dy)**2 - (vx**2)*(r**2)

        u=quad(a, b, c)

        t=[mth.degrees(mth.acos(x)) for x in u] + [-mth.degrees(mth.acos(x)) for x in u]

        out=[(circle(theta)) for theta in t if (circle(theta) in line)]

    return get_rid_of_multiple_points(out)


def intersection_segment_circle(segment, circle):
    int_=intersection_line_circle(segment.line, circle)
    out=[point for point in int_ if (point in segment)]
    return out

def intersection_ray_circle(ray, circle):
    int_=intersection_line_circle(ray.line, circle)
    out=[point for point in int_ if (point in ray)]
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

    return get_rid_of_multiple_points(out)


class Circle(GObject):

    def __init__(self, center, radius):
        self.center=center
        self.radius=radius
        self.type="circle"

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


