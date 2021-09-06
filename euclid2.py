from euclid_math import *


class GObject(object):
    """ base class for geometric objects """
    def __init__(self):
        pass


origin=np.array([[0],[0]])

rotation_matrix=lambda theta: np.array([[cos(theta),-1*sin(theta)],
                                        [sin(theta),   cos(theta)]])

rotate=lambda A,B,theta: np.matmul(rotation_matrix(theta),A-B)+B

def points_to_line(P1,P2):
    """ Returns the line through P1 and P2 """
    A=P1
    v=P2-P1
    return Line(A, v)


class Line(GObject):
    def __init__(self,A,v):
        self.A=A
        self.v=v
        self.B=self.A+self.v

    def __call__(self,t): #ok
        return self.A+self.v*t

    def __contains__(self,P): #ok
        v1,v2=self.v[0,0],self.v[1,0]
        V=np.array([[v1,0],
                    [0,v2]],dtype=np.float64)
        P_=P-self.A
        T=system(V,P_)
        return mth.isclose(T[0,0],T[1,0],abs_tol=abs_tol)

    def intersection(self,line): #ok
        v1x,v1y=self.v[0,0],self.v[1,0]
        v2x,v2y=line.v[0,0],line.v[1,0]
        V=np.array([[v1x,-1*v2x],
                    [v1y,-1*v2y]])

        A=line.A-self.A
        T=system(V, A)
        print(T)
        t1=T[0,0]
        out=self(t1)

        return out

    def rotate(self,P,theta): #ok
        A_,B_=rotate(self.A, P, theta),rotate(self.B, P, theta)
        v_=B_-A_
        return Line(A_, v_)
        

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


x_vect=np.array([[1],[0]])
y_vect=np.array([[0],[1]])
zero_vect=np.array([[0],[0]])
y_axis=Line(zero_vect,x_vect)
x_axis=Line(zero_vect,y_vect)


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

