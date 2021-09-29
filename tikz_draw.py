import os
from euclid2 import *

RND8=np.vectorize(lambda x: round(x, 8))

def create_file(file_name):
    with open(file_name,"x") as file:
        pass

def write_to_file(file_name,text):
    with open(file_name,"a") as file:
        file.write(text+"\n")


def_preamble="""%tikz_draw
\\documentclass[11pt,a4paper]{article}
\\usepackage[utf8]{inputenc}
\\usepackage[english]{babel}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage[left=2cm,right=2cm,top=2cm,bottom=2cm]{geometry}
\\usepackage{tikz}
%tikzlibrary
\\usetikzlibrary{arrows.meta}
%preamble

"""

def_editor="vim"
pdflatex_command='pdflatex -shell-escape'

def_vector_config="black, thick"
def_ray_config="black , thick, -Stealth"
def_arrow_tip="-Stealth"
def_grid_config="gray, opacity=0.75, dashed"
def_axis_arrow_tip="Stealth-Stealth"
def_point_config="fill=cyan!20!black, draw=black"
def_path_config="black, thick"
def_path_fill_config="cyan, opacity=0.3"
def_line_config="black, thick, Stealth-Stealth"
def_arc_fill_config="cyan, opacity=0.3"
def_arc_config=""
def_node_draw_config=""
def_node_config="anchor=north"
def_circle_config="cyan!20!black"



class Tikz():
    def __init__(self,file_name, preamble=def_preamble):
        try:
            create_file(file_name)
            #creates the file
        except:
            print('WARNING: FILE ALREADY EXISTS')
        if preamble!=None:
            write_to_file(file_name,preamble)
            #writes the preamble

        self.file_name=file_name

    def write(self,text):
        write_to_file(self.file_name,text)

    def read(self):
        with open(self.file_name) as file:
            print(file.read())

    def edit(self,editor=def_editor):
        os.system(f'{editor} {self.file_name}')

    def begin(self,env,config=None):
        Config=f"[{config}]" if (not isnone(config) and config!="") else ""
        self.write('\\begin{'+env+'}'+f"{Config}"+'\n')

    def end(self,env):
        self.write('\\end{'+env+'}')

    def pdf(self):
        os.system(f'{pdflatex_command} {self.file_name}')

    def clip(self, x_range=[-5,5], y_range=[-5,5]):
        xmin,xmax=x_range
        ymin,ymax=y_range
        clip_code=f"\\clip {str((xmin, ymin))} rectangle {str((xmax, ymax))};"
        self.write(clip_code)

    def draw_axis(self, x_range=[-5,5], y_range=[-5,5], arrow_tip=def_axis_arrow_tip ,tick_labels=False):
        xmin,xmax=x_range
        ymin,ymax=y_range

        Tip=f"[{arrow_tip}]" if (not isnone(arrow_tip) and arrow_tip!="") else ""

        axis_code=f"""
    %axis
    \\draw{Tip} ({xmin},0) -- ({xmax},0);
    \\draw{Tip} (0, {ymin}) -- (0, {ymax});\n"""

        axis_ticks_code="""
    %axis ticks
    \\foreach \\x in {"""+f"""{xmin+1},...,{xmax-1}"""+"""}
        \\draw (\\x,-2pt) -- (\\x,2pt);\n""" + """%\n
    \\foreach \\x in {"""+f"""{ymin+1},...,{ymax-1}"""+"""}
        \\draw (-2pt,\\x) -- (2pt,\\x);\n"""

        #TODO
        code=axis_code+axis_ticks_code
        self.write(code)

    def draw_grid(self, x_range=[-5,5], y_range=[-5,5], config=def_grid_config):
        xmin,xmax=x_range
        ymin,ymax=y_range

        Config=f"[{config}]" if (not isnone(config) and config!="") else ""
        grid_code=f"\\draw{Config} {str((xmin,ymin))} grid {str((xmax,ymax))};"
        self.write(grid_code)

    def draw_point(self, point, config=def_point_config, radius=2):
        X,Y=row_vector(point)
        X,Y=round(X, 8),round(Y, 8)

        Config=f"[{config}]" if (not isnone(config) and config!="") else ""
        draw_point_code=f"\\filldraw{Config} ({X},{Y}) circle ({radius}pt);"
        self.write(draw_point_code)

    def draw_vector(self,vector,start=origin, config=def_vector_config, arrow_tip=def_arrow_tip):
        X,Y=row_vector(vector)
        X,Y=round(X, 8),round(Y, 8)

        Config=f"[{config},{arrow_tip}]" if (not isnone(config) and config!="") else f"[{Tip}]"
        code=f"""
    %vector [{X}, {Y}]
    \\draw{Config} {(start[0,0], start[1,0])} -- {str((X,Y))};
    """
    
        self.write(code)

    def draw_path(self,*points, config=def_path_config, cycle=False):
        points_xy=[(round(p[0,0], 8), round(p[1,0], 8)) for p in points]
        path_string=""

        for i in range(0,len(points_xy)-1):
            path_string=path_string+f"{str(points_xy[i])} -- "

        path_string=path_string+f"{str(points_xy[-1])};" if not cycle else path_string+f"{str(points_xy[-1])} -- cycle;" 

        Config=f"[{config}]" if (not isnone(config) and config!="") else ""

        draw_path_code=f"\\draw{Config}  "+path_string
        self.write(draw_path_code)

    def fill_path(self, *points, fill_config=def_path_fill_config, cycle=False):
        points_xy=[(round(p[0,0], 8), round(p[1,0], 8)) for p in points]
        path_string=""

        for i in range(0,len(points_xy)-1):
            path_string=path_string+f"{str(points_xy[i])} -- "

        path_string=path_string+f"{str(points_xy[-1])};" if not cycle else path_string+f"{str(points_xy[-1])} -- cycle;" 

        Config=f"[{fill_config}]" if (not isnone(fill_config) and fill_config!="") else ""

        draw_path_code=f"\\fill{Config}  "+path_string
        self.write(draw_path_code)        

    def draw_points(self, *points, config=def_point_config, radius=2):
        for point in points:
            self.draw_point(point, config=config, radius=radius)

    def draw_line(self, line, config=def_line_config, x_range=[-5,5], y_range=[-5, 5]):
        xmin, xmax=x_range
        ymin, ymax=y_range
        
        A=col_vector([xmin, ymin])
        B=col_vector([xmax, ymin])
        C=col_vector([xmax, ymax])
        D=col_vector([xmin, ymax])

        up=Line(C, D)
        right=Line(C, B)

        down=Line(A, B)
        left=Line(A, D)

        int1=[line.intersection(down), line.intersection(left)]
        int2=[line.intersection(up), line.intersection(right)]

        in_range=lambda p: (xmin <= round(p[0,0], 12) <= xmax) and (ymin <= round(p[1,0], 12) <= ymax) if not isnone(p) else False
        #might cause trouble due to limitations of floating point arithmetic

        int1_=[p for p in int1 if in_range(p)]
        int2_=[q for q in int2 if in_range(q)]

        if len(int1_)==len(int2_):
            p1=int1_[0]
            p2=int2_[0]
        elif len(int1_)==2 and int2_==0:
            p1,p2=int1_
        elif len(int1_)==0 and len(int2_)==2:
            p1,p2=int2_
        else:
            p1=int1_[0]
            p2=int2_[0]

        p1=tuple(RND8(row_vector(p1)))
        p2=tuple(RND8(row_vector(p2)))
        
        draw_Config=f"[{config}]" if (not isnone(config) and config!="") else ""
        line_draw_code=f"\\draw{draw_Config} {p1} -- {p2};"

        self.write(line_draw_code)

    def draw_lines(self, *lines, config=def_line_config, x_range=[-5,5], y_range=[-5,5]):
        for line in lines:
            self.draw_line(line, config=config, x_range=x_range, y_range=y_range)

    def draw_ray(self, ray, config=def_ray_config, x_range=[-5,5], y_range=[-5, 5]):
        #ray.A cannot be outside given x y range
        xmin, xmax=x_range
        ymin, ymax=y_range
        
        A=col_vector([xmin, ymin])
        B=col_vector([xmax, ymin])
        C=col_vector([xmax, ymax])
        D=col_vector([xmin, ymax])

        up=Line(C, D)
        right=Line(C, B)

        down=Line(A, B)
        left=Line(A, D)

        int1=[ray.intersection(down), ray.intersection(left)]
        int2=[ray.intersection(up), ray.intersection(right)]

        in_range=lambda p: (xmin <= round(p[0,0], 12) <= xmax) and (ymin <= round(p[1,0], 12) <= ymax) if not isnone(p) else False
        #might cause trouble due to limitations of floating point arithmetic

        int1_=[p for p in int1 if in_range(p)]
        int2_=[q for q in int2 if in_range(q)]

        P = int1_[0] if len(int1_)!=0 else int2_[0]

        p1=tuple(RND8(row_vector(ray.A)))
        p2=tuple(RND8(row_vector(P)))

        draw_Config=f"[{config}]" if (not isnone(config) and config!="") else ""
        ray_draw_code=f"\\draw{draw_Config} {p1} -- {p2};"

        self.write(ray_draw_code)

    def draw_angle(self, A, B, C, config=def_arc_config, radius=1, fill_config=def_arc_fill_config, right_angle=True):
        
        Angle=angle(A, B, C)
        Bx, By=RND8(row_vector(B))
        
        start_angle=atan2((A-B)[0,0], (A-B)[1,0])
        end_angle= start_angle + Angle

        Angle, start_angle, end_angle = round(Angle, 8), round(start_angle, 8), round(end_angle, 8)

        draw_Config=f"[{config}]" if (not isnone(config) and config!="") else ""
        fill_Config=f"[{fill_config}]" if (not isnone(fill_config) and fill_config!="") else ""

        if (not right_angle) or (not isclose(Angle, 90)):
            draw_angle_code=f"\\draw{draw_Config}  ([shift=({start_angle}:{radius})]{Bx},{By}) arc[start angle={start_angle}, end angle={end_angle}, radius={radius}];"
            fill_angle_code=f"\\fill{fill_Config} {Bx,By} -- ([shift=({start_angle}:{radius})]{Bx},{By}) arc[start angle={start_angle}, end angle={end_angle}, radius={radius}] -- cycle;"
        else:
            draw_angle_code=f"\\draw{draw_Config} {Bx, By} -- ([shift=({end_angle}:{radius/sqrt(2)})]{Bx}, {By}) -- ([shift=({(start_angle+Angle/2)%360}:{radius})]{Bx}, {By}) -- ([shift=({start_angle}:{radius/sqrt(2)})]{Bx}, {By}) -- cycle;"
            fill_angle_code=f"\\fill{fill_Config} {Bx, By} -- ([shift=({end_angle}:{radius/sqrt(2)})]{Bx}, {By}) -- ([shift=({(start_angle+Angle/2)%360}:{radius})]{Bx}, {By}) -- ([shift=({start_angle}:{radius/sqrt(2)})]{Bx}, {By}) -- cycle;"

        self.write(fill_angle_code)
        self.write(draw_angle_code)

    def draw_circle(self, circle, config=def_circle_config):
        Cx, Cy= RND8(row_vector(circle.center))
        radius= round(circle.radius, 8)

        draw_Config=f"[{config}]" if (not isnone(config) and config!="") else ""

        draw_circle_code=f"\\draw{draw_Config} ({Cx}, {Cy}) circle ({radius});"
        self.write(draw_circle_code)

    def node(self, position, node_config=def_node_config , config=def_node_draw_config, text=""):
        X,Y=row_vector(position)
        X,Y=round(X, 8), round(Y, 8)

        Config=f"[{config}]" if (not isnone(config) and config!="") else ""
        node_Config=f"[{node_config}]" if (not isnone(node_config) and node_config!="") else ""
        
        node_code=f"\\draw{Config} {X,Y} node {node_Config} "+"{"+f"{text}"+"};"
        
        self.write(node_code)

