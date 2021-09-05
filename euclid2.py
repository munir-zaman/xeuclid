from euclid_math import *


class GObject(object):
    """ base class for geometric objects """
    def __init__(self):
        pass


origin=col_vector(np.array([0,0]))

rotation_matrix=lambda theta: np.array([[cos(theta),-1*sin(theta)],
                                        [sin(theta),   cos(theta)]])

rotate=lambda A,B,theta: np.matmul(rotation_matrix(theta),A-B)+B



def p2l(P1,P2):
    """ Returns the line through P1 and P2 """
    x1,y1=P1
    x2,y2=P2
    A,B,C=(y1-y2),(x2-x1),(y2-y1)*x1-(x2-x1)*y1 
    return Line(A, B, C)

def points_to_line(p1,p2):
    return p2l(p1, p2)

def points_to_segment(p1,p2):
    return Segment(p1, p2)

def points_to_polygon(*p):
    return Polygon(*p)

def points_to_gobject(points,obj_type):
    return eval(f"points_to_{obj_type.lower()}(*{points})")

def apply_transformation(obj,func):

    Func=lambda L: [func(*l) for l in L]
    out=[]

    if not isinstance(obj, list):
        out=points_to_gobject(Func(obj.get_unique_points()),obj.obj_type)
    else:
        out=[points_to_gobject(Func(gobj.get_unique_points()),gobj.obj_type) for gobj in obj]

    return out

def angle(A,B,C):
    """ returns the angle <ABC 
        A,B,C : List of length 2 [x,y]
    """
    a1,a2=A
    b1,b2=B
    c1,c2=C
    #x,y coordinates

    A1,A2=a1-b1,a2-b2
    B1,B2=0,0
    C1,C2=c1-b1,c2-b2
    #translate [x,y] -> [x-b1,y-b2]

    t1=atan2(A1,A2)
    t2=atan2(C1,C2)

    return (t2-t1)%360

def angle_bisector(A,B,C):
    """ returns the angle bisector of angle <ABC"""
    a1,a2=A
    b1,b2=B
    c1,c2=C
    #x,y coordinates

    A1,A2=a1-b1,a2-b2
    B1,B2=0,0
    C1,C2=c1-b1,c2-b2
    #translate [x,y] -> [x-b1,y-b2]

    t1=atan2(A1,A2)
    t2=atan2(C1,C2)
    T=(((t2-t1)%360)/2)+t1
    R=[cos(T),sin(T)]

    line=p2l([B1,B2],R)
    line=line.shift([b1,b2])

    return line

def dist(p1,p2):
    """ Return distance between p1 and p2 """
    return mth.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def mid(p1,p2):
    """ Return midpoint of p1 and p2 """
    return Point([(p1[0]+p2[0])/2,(p1[1]+p2[1])/2])


class Line(GObject):

    def __init__(self,a,b,c):
        """ Line ax+by+c=0
        """

        self.a,self.b,self.c=a,b,c

        if b!=0:
            self.yint=-c/b
            self.slope=-a/b
        else:
            self.yint="inf"
            self.slope="inf"

        if a!=0:
            self.xint=-c/a
        else:
            self.xint="inf"

        # f(0)=y-intercept
        # f(x-intercept)=0
        
        if self.slope!="inf":
            self.xtheta=atan(self.slope)
        else:
            self.xtheta=90

        if self.slope!=0 and self.slope!="inf":
            self.ytheta=atan(self.slope**(-1))
        elif self.slope=="inf":
            self.ytheta=0
        elif self.slope==0:
            self.ytheta=90

        self.obj_type="Line"

    def __repr__(self):
        return f"{self.a}*x+{self.b}*y+{self.c}=0"

    def __str__(self):
        return str(f"{self.a}*x+{self.b}*y+{self.c}=0") 
    
    def explicit(self):
        print(f"{self.slope}*x+{self.yint}")
        return [self.slope,self.yint]

    def __call__(self,x):
        if self.slope!="inf":
            out=self.slope*x+self.yint
        else:
            print(str(self))
            if x==-self.c/self.a:
                out=1
            else:
                out="inf"

        return out

    def __eq__(self,line):

        if self.intersection(line,list) and len(line)==3:
            line=Line(*line)

        try:
            out=(self.angle(line)==0)
        except:
            out=False

        return out

    def __and__(self,line):
        """ returns the intersection of line and self """
        return self.intersection(line)

    def __xor__(self,line):
        """ returns the angle between self and line """
        return self.angle(line)

    def __add__(self,other):
        out=None
        if isinstance(other, list):
            other=Array(other)
        p1,p2=self.get_unique_points()
        out=p2l(p1+other,p2+other)

        return out

    def __radd__(self,other):
        return self+other

    def __sub__(self,other):
        if isinstance(other, list):
            other=-Array(other)
        else:
            other=-other

        return self+other

    def __rsub__(self,other):
        out=None
        if isinstance(other, list):
            other=Array(other)
        p1,p2=self.get_unique_points()
        out=p2l(other-p1,other-p2)

        return out

    def contains(self,p):
        """ checks if the point "p" is on the line "self" """
        px,py=p
        out=(self.a*px+self.b*py+self.c==0)
        return out

    def shift(self,*dv):
        """ Translates/Shifts the line.
            [dx,dy]: list of len 2
            Applies the translation:
            [x,y] --> [x+dx,y+dy]
        """
        if len(dv)==1:
            dv=dv[0]
        dx,dy=dv
        return Line(self.a, self.b, self.c-self.a*dx-self.b*dy)

    def goto_point(self,point):
        """ shifts the point to point "point" """
        p=point
        if self.b!=0:
            A=-self.slope
            B=1
            C=-(A*p[0]+B*p[1])
        else:
            A=self.a
            B=0
            C=-(A*p[0]+B*p[1])

        return Line(A, B, C)

    def parallel(self,point):
        """ returns the parallel line through point "point" """
        line=self
        px,py=point
        if line.b !=0:
            A=-line.slope
            B=1
        else:
            B=0
            A=line.a
        C=-(A*px+B*py)

        return Line(A, B, C)

    def perpendicular(self,point):
        """ returns the perpendicular line through point "point" """
        line=self
        p=point
        xtheta=(line.xtheta+90)%180
        A=-tan(xtheta)
        B=1
        C=-(A*p[0]+B*p[1])
        return Line(A, B, C)

    def angle(self,line):
        """ returns the angle between line "self" and line "line" """
        line1=self
        line2=line
        angle=(line1.xtheta-line2.xtheta)%180
        return angle

    def distance(self,point):
        ''' returns the distance between point "point" and line "self" '''

        line=self
        px,py=point
        perp=line.perpendicular(point)
        intersect=perp & line

        return dist(point, intersect)

    def intersection(self,line):
        """ returns the intersection point of "self" and line "line" """
        l1,l2=self,line
        return Point(system2([l1.a,l1.b,l1.c],[l2.a,l2.b,l2.c]))

    def rotate(self,point,angle):
        """ rotates the line "self" around point "pont" by an angle "angle" """
        func=lambda x,y: Point([x,y]).rotate(point,angle)
        line=apply_transformation(self, func)
        return line

    def get_unique_points(self):
        p1,p2=None,None
        if self.b==0:
            p1=Point([-self.c/self.a,1])
            p2=Point([-self.c/self.a,2])
        else:
            p1=Point([1,-(self.a/self.b)*1-(self.c/self.b)])
            p2=Point([2,-(self.a/self.b)*2-(self.c/self.b)])

        return p1,p2 

yaxis=Line(1,0,0)
xaxis=Line(0,1,0)


class Segment(GObject):
    """docstring for Segment"""
    def __init__(self, A, B):

        if not isinstance(A, Point):
            A=Point(A)
        if not isinstance(B, Point):
            B=Point(B)

        self.A=A
        self.B=B
        self.D=self.B-self.A
        self.line=p2l(self.A, self.B)
        self.obj_type="Segment"

    def __call__(self,t):
        return self.A+self.D*t

    def __repr__(self):
        return f"{self.A}+{self.D}*t"

    def contains(self,point):
        Ax,Ay=self.A
        dx,dy=self.D
        x,y=point
        t=(x-Ax)/dx
        out=eq(Ay+dy*t,y) and ((0 <= t) and (t <= 1))  
        return out

    def rotate(self,point,angle):
        A_=self.A.rotate(point, angle)
        B_=self.B.rotate(point, angle)
        return Segment(A_, B_)

    def intersection(self,other):
        out=None
        if isinstance(other, Segment):
            A,B=self.A,self.D
            C,D=other.A,other.D
            Ax,Ay=A
            Bx,By=B
            Cx,Cy=C
            Dx,Dy=D
            A1=[Bx,-Dx,Ax-Cx]
            A2=[By,-Dy,Ay-Cy]
            u,v=system2(A1, A2)
            exists=((0 <= u) and (u <= 1)) and ((0 <= v) and (v <= 1))
            if exists:
                print(f"t value: {u}, intersection point: {A+B*u}")
                out=Point(A+B*u)
                
        elif isinstance(other, Line):
            x,y=other.intersection(self.line)
            A,B=self.A,self.D
            Ax,Ay=A
            Bx,By=B
            t=(x-Ax)/Bx
            exists=((0 <= t) and (t <= 1)) and eq(t,(y-Ay)/By)
            if exists:
                print(f"t value: {t}, intersection point: {A+B*t}")
                out=Point(A+B*t)
        
        return out

    def get_unique_points(self):
        return self.A,self.B


def collinear(*points):
    """ collinear(*points)

        checks if the given set of points are collinear 
        If,the given set of points are collinear: 
            returns the line through the given set of points. 
        Else, 
            returns None

        Examples:
            In [1]: collinear([1,-1/2],[2,0],[4,1],[6,2])
            True
            Out[1]: -0.5*x+1*y+1.0=0

            In [2]: collinear([1,2],[1,3],[1,6],[1,7])
            True
            Out[2]: -1*x+0*y+1=0

            In [3]: collinear([1,-1/2],[2,0],[4,1],[6,0])
            False
    """
    A,B=points[0],points[1]
    AB=p2l(A,B)
    out=True

    for p in points:
        out=out and AB.contains(p)
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
            lines[l]=p2l(*lines[l])
        if not isinstance(lines[l-1],Line):
            lines[l]=p2l(*lines[l-1])

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



class Polygon(GObject):

    def __init__(self,*vertices):

        self.vertices=list(vertices)
        self.area=self.area()
        self.obj_type="Polygon"

    def area(self):

        vertices=self.vertices[::]
        vertices.append(self.vertices[0])

        V=[[vertices[p],vertices[p+1]] for p in range(0,len(vertices)-1)]

        Det=lambda l: l[0][0]*l[1][1]-l[1][0]*l[0][1]
        Area=abs(sum([Det(l) for l in V]))/2

        return Area

    def shift(self,*dv):
        if len(dv)==1:
            dv=dv[0]

        vertices_=[]

        for p in self.vertices:
            vertices_.append(list(array(p)+array(dv)))

        return Polygon(*vertices_)

    def rotate(self,point,angle):
        func=lambda x,y: Point([x,y]).rotate(point,angle)
        poly=apply_transformation(self, func)
        return poly

    def get_unique_points(self):
        return self.vertices[::]

