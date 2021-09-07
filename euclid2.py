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

    return round((t2-t1)%360, 8)

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



class Line(GObject):
    def __init__(self,A,B):
        self.A=A
        self.B=B
        self.v=self.B-self.A

    def __call__(self,t):
        """ evaluates l(t)= A+ v*t
            input(s):
                t: type= float or int
            output(s):
                P: type= np.array([[Px],
                                   [Py]])
         """
        return self.A+self.v*t

    def __repr__(self):
        return f"[{self.A[0,0]}, {self.A[1,0]}] +[{self.v[0,0]}, {self.v[1,0]}]*t"

    def __str__(self):
        return str(f"[{self.A[0,0]}, {self.A[1,0]}] +[{self.v[0,0]}, {self.v[1,0]}]*t")

    def __contains__(self,P):
        return self.inv(P)!=None

    def inv(self,P):
        v1,v2=self.v[0,0],self.v[1,0]
        V=np.array([[v1,0],
                    [0,v2]],dtype=np.float64)
        P_=P-self.A
        T=system(V,P_)
        if not mth.isclose(T[0,0],T[1,0],abs_tol=abs_tol):
            print(f"({P[0,0]},{P[1,0]}) not in {str(self)}")
            T=np.array([[None],[None]])
        return T[0,0]

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

    def matmul(self,vector):
        return Line(matmul(self.A, vector), matmul(self.B, vector))

    def intersection(self,line,get_t_value=False):
        v1x,v1y=self.v[0,0],self.v[1,0]
        v2x,v2y=line.v[0,0],line.v[1,0]

        V=np.array([[v1x,-1*v2x],[v1y,-1*v2y]])

        if not (self | line):
            A=line.A-self.A
            T=system(V, A)
            print(T)
            t1=T[0,0]
            out=self(t1) if not get_t_value else T
        else:
            print("The Lines are Parallel")
            out=None

        return out

    def rotate(self,P,theta):
        A_,B_=rotate(self.A, P, theta),rotate(self.B, P, theta)
        return Line(A_, B_)

    def parallel_line(self,P):
        return Line(P, self.v+P)

    def perpendicular_line(self,P):
        return Line(P, rotate(self.v,origin,90)+P)

    def __or__(self,line):       
        v1x,v1y=self.v[0,0],self.v[1,0]
        v2x,v2y=line.v[0,0],line.v[1,0]
        V=np.array([[v1x,-1*v2x],
                    [v1y,-1*v2y]])

        return isclose(det(V),0.0)

    def __and__(self,line):
        return self.intersection(line)

    def __xor__(self,line):
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
        return round(theta,8)


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
    return Point([(p1[0,0]+p2[0,0])/2,(p1[1,0]+p2[1,0])/2])


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

