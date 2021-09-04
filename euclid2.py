from euclid_math import *


class Point(Array):
    def __init__(self,coordinates,polar=False):

        if polar:
            r,t=coordinates
            coordinates=[r*cos(t),r*sin(t)]
        super().__init__(coordinates)

        self.x,self.y=self
        self.polar=Array([sqrt(self.x**2+self.y**2),atan2(self.x, self.y)])
        self.r,self.t=self.polar

    def rotate(self,point,angle):
        p1,p2=self
        q1,q2=point

        P1,P2=p1-q1,p2-q2
        line=p2l([P1,P2],[0,0])
        xtheta=line.xtheta

        R=dist([P1,P2],[0,0])
        Rx=R*cos(angle+xtheta)
        Ry=R*sin(angle+xtheta)

        p1_,p2_=Rx+q1,Ry+q2
        return Point([p1_,p2_])


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
    return [(p1[0]+p2[0])/2,(p1[1]+p2[1])/2]

def p2l(P1,P2):
    """ Returns the line through P1 and P2 """
    x1,y1=P1
    x2,y2=P2
    A,B,C=(y1-y2),(x2-x1),(y2-y1)*x1-(x2-x1)*y1 
    return Line(A, B, C)


class Line():

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

    def __repr__(self):
        return f"{self.a}*x+{self.b}*y+{self.c}=0"

    def __str__(self):
        return str(f"{self.a}*x+{self.b}*y+{self.c}=0") 
    
    def slope_int(self):
        return f"{self.slope}*x+{self.yint}"

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

        if intersection(line,list) and len(line)==3:
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
        return system2([l1.a,l1.b,l1.c],[l2.a,l2.b,l2.c])

    def rotate(self,point,angle):
        """ rotates the line "self" around point "pont" by an angle "angle" """
        px,py=point
        xtheta=self.xtheta
        xtheta_= xtheta + angle
        m_=tan(xtheta_)

        B_,A_=1,-m_
        C_=-(A_*px+B_*py)

        line=Line(A_,B_,C_)

        if not self.contains(point):
            #px,py=[]
            R=self.distance(point)
            Rx=R*cos(90+xtheta_)
            Ry=R*sin(90+xtheta_)
            line=line.shift([Rx,Ry])
        else:
            line=line

        return line

yaxis=Line(1,0,0)
xaxis=Line(0,1,0)


class Segment():
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

    def __call__(self,t):
        return self.A+self.D*t

    def __repr__(self):
        return f"{self.A}+{self.D}*t"

    def contains(self,point):
        Ax,Ay=self.A
        dx,dy=self.D
        x,y=point
        t=(x-Ax)/dx
        out=(Ay+dy*t==y) and ((0 <= t) and (t <= 1))  
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
            t=(Cx-Ax)/(Bx-Dx)
            exists=((0 <= t) and (t <= 1)) and t==(Cy-Ay)/(By-Dy)
            if exists:
                print(f"t value: {t}, intersection point: {A+B*t}")
                out=Point(A+B*t)
                
        elif isinstance(other, Line):
            x,y=other.intersection(self.line)
            A,B=self.A,self.D
            Ax,Ay=A
            Bx,By=B
            t=(x-Ax)/Bx
            exists=((0 <= t) and (t <= 1)) and t==(y-Ay)/By
            if exists:
                print(f"t value: {t}, intersection point: {A+B*t}")
                out=Point(A+B*t)
        
        return out

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



class Polygon():

    def __init__(self,*vertices):

        self.vertices=list(vertices)
        self.area=self.area()

    def area(self):

        vertices=self.vertices
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
        vertices_=[]
        for p in self.vertices:
            vertices_.append(rotate(p, point, angle))
        return Polygon(*vertices_)

