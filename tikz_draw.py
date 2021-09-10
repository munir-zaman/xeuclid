import os

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

    def draw_vector(vector,start=(0,0), config=def_vector_config):
        X,Y=row_vector(vector)

        Config=f"[{config},->]" if (config!=None or config!='') else "[->]"
        code=f"""
        %vector [{X}, {Y}]
        \\draw{Config} {str(tuple(start))} -- {str((X,Y))};

        """
        self.write(code)



