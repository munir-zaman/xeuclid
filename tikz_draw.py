import os
from euclid2 import *

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
%preamble
"""

def_editor="vim"
pdflatex_command='pdflatex -shell-escape'

def_vector_config="blue, thick"
def_grid_config="cyan"
def_point_config="fill=cyan!20!black, draw=black"
def_path_config="black, thick"
def_line_config="black, thick"
def_arc_fill_config="red, opacity=0.3"
def_arc_config=""
def_node_draw_config=""
def_node_config="anchor=north"


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

    def clip(self, min_val, max_val):
        clip_code=f"\\clip {str((min_val,max_val))} rectangle {str((max_val,min_val))};"
        self.write(clip_code)

    def draw_axis(self,xy_range=[-5,5],tick_labels=False):
        axis_code=f"""
    %axis
    \\draw[<->] ({xy_range[0]},0) -- ({xy_range[1]},0);
    \\draw[<->] (0, {xy_range[0]}) -- (0, {xy_range[1]});\n"""

        axis_ticks_code="""
    %axis ticks
    \\foreach \\x in {"""+f"""{xy_range[0]+1},{xy_range[0]+2},...,{xy_range[1]-2},{xy_range[1]-1}"""+"""}
        \\draw (\\x,-2pt) -- (\\x,2pt);\n""" + """%\n
    \\foreach \\x in {"""+f"""{xy_range[0]+1},{xy_range[0]+2},...,{xy_range[1]-2},{xy_range[1]-1}"""+"""}
        \\draw (-2pt,\\x) -- (2pt,\\x);\n"""

        #TODO
        code=axis_code+axis_ticks_code
        self.write(code)

    def draw_grid(self, xy_range=[-5,5],config=def_grid_config):
        xmin, xmax=xy_range
        Config=f"[{def_grid_config}]" if (not isnone(config) and config!="") else ""
        grid_code=f"\\draw{Config} {str((xmin,xmax))} grid {str((xmax,xmin))};"
        self.write(grid_code)

    def draw_point(self, point, config=def_point_config, radius=2):
        X,Y=row_vector(point)
        Config=f"[{config}]" if (not isnone(config) and config!="") else ""
        draw_point_code=f"\\filldraw{Config} ({X},{Y}) circle ({radius}pt);"
        self.write(draw_point_code)

    def draw_vector(self,vector,start=(0,0), config=def_vector_config):
        X,Y=row_vector(vector)

        Config=f"[{config},->]" if (not isnone(config) and config!="") else "[->]"
        code=f"""
    %vector [{X}, {Y}]
    \\draw{Config} {str(tuple(start))} -- {str((X,Y))};
    """
        self.write(code)

    def draw_path(self,*points, config=def_path_config, cycle=False):
        points_xy=[(p[0,0], p[1,0]) for p in points]
        path_string=""

        for i in range(0,len(points_xy)-1):
            path_string=path_string+f"{str(points_xy[i])} -- "

        path_string=path_string+f"{str(points_xy[-1])};" if not cycle else path_string+f"{str(points_xy[-1])} -- cycle;" 

        Config=f"[{config}]" if (not isnone(config) and config!="") else ""

        draw_path_code=f"\\draw{Config}  "+path_string
        self.write(draw_path_code)

    def draw_points(self, *points, config=def_point_config, radius=2):
        for point in points:
            self.draw_point(point, config=config, radius=radius)

    def draw_line(self, line, config=def_line_config, t_range=[-10,10]):
        A=line(t_range[0])
        B=line(t_range[1])
        Config=f"[{config},<->]" if (not isnone(config) and config!="") else "[<->]"
        
        self.draw_path(A,B, config=Config, cycle=False)

    def draw_angle(self, A, B, C, config=def_arc_config, radius=1, fill_config=def_arc_fill_config):
        
        Angle=angle(A, B, C)
        Bx, By=row_vector(B)
        
        start_angle=atan2((A-B)[0,0], (A-B)[1,0])
        end_angle=atan2((C-B)[0,0], (C-B)[1,0])

        draw_Config=f"[{config}]" if (not isnone(config) and config!="") else ""
        fill_Config=f"[{fill_config}]" if (not isnone(fill_config) and fill_config!="") else ""

        draw_angle_code=f"\\draw{draw_Config}  ([shift=({start_angle}:{radius})]{Bx},{By}) arc[start angle={start_angle}, end angle={end_angle}, radius={radius}];"
        fill_angle_code=f"\\fill{fill_Config} {Bx,By} -- ([shift=({start_angle}:{radius})]{Bx},{By}) arc[start angle={start_angle}, end angle={end_angle}, radius={radius}] -- cycle;"
        
        self.write(fill_angle_code)
        self.write(draw_angle_code)

    def node(self, position, node_config=def_node_config , config=def_node_draw_config, text=""):
        X,Y=row_vector(position)
        
        Config=f"[{config}]" if (not isnone(config) and config!="") else ""
        node_Config=f"[{node_config}]" if (not isnone(node_config) and node_config!="") else ""
        
        node_code=f"\\draw{Config} {X,Y} node {node_Config} "+"{"+f"{text}"+"};"
        
        self.write(node_code)

