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
\\usepackage{tikzpicture}
%preamble
"""

def_editor="vim"


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

    def edit(self,editor=def_editor):
        os.system(f'{editor} {self.file_name}')

    def begin(self,env):
        self.write('\\begin{'+env+'}\n')

    def end(self,env):
        self.write('\\end{'+env+'}')



