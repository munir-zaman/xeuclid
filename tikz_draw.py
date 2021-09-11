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

    def begin(self,env):
        self.write('\\begin{'+env+'}\n')

    def end(self,env):
        self.write('\\end{'+env+'}')

    def pdf(self):
        os.system(f'{pdflatex_command} {self.file_name}')

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
        Config=f"[{def_grid_config}]" if def_grid_config!=None or def_grid_config!='' else ""
        grid_code=f"\\draw{Config} {str((xmin,xmax))} grid {str((xmax,xmin))};"
        self.write(grid_code)

    def draw_point(self, point, config=def_point_config, radius=2):
        X,Y=row_vector(point)
        Config=f"[{config}]" if (config!=None or config!='') else ""
        draw_point_code=f"\\filldraw{Config} ({X},{Y}) circle ({radius}pt);"
        self.write(draw_point_code)

    def draw_vector(self,vector,start=(0,0), config=def_vector_config):
        X,Y=row_vector(vector)

        Config=f"[{config},->]" if (config!=None or config!='') else "[->]"
        code=f"""
    %vector [{X}, {Y}]
    \\draw{Config} {str(tuple(start))} -- {str((X,Y))};
    """
        self.write(code)

    def draw_path(self,*points, config=def_path_config):
        points_xy=[(p[0,0], p[1,0]) for p in points]
        path_string=""
        for i in range(0,len(points_xy)-1):
            path_string=path_string+f"{str(points_xy[i])} -- "
        path_string=path_string+f"{str(points_xy[-1])};"
        Config=f"[{config}]" if (config!=None or config!='') else ""

        draw_path_code=f"\\draw{Config}  "+path_string
        self.write(draw_path_code)



