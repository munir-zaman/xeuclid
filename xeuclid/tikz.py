import subprocess
import os
from xeuclid.euclid2 import *
from xeuclid.utils.file import *
import numpy as np

# trying to create a more polished version of tikz_draw.py

STANDALONE_TEMPLATE = \
"""\\documentclass[tikz, border=5, svgnames]{standalone}
\\usepackage[utf8]{inputenc}
\\usepackage[english]{babel}
\\usepackage{xcolor}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}"""

DEFAULT_PACKAGES = []
DEFAULT_TIKZLIBS = ['arrows.meta']


def parse_coordinate(coord):
    if isinstance(coord, (tuple, list, np.ndarray)):
        x, y = row_vector(coord)
        x, y = round(x, 8), round(y, 8)
        parsed_coord = f"({x}, {y})"

    elif isinstance(coord, str):
        parsed_coord = coord.strip()

    else:
        raise TypeError('`coord` must be of type list, tuple, np.ndarray or str.')

    return parsed_coord

def parse_kwargs(kwargs):
    args_list = [(args.replace("_", " ").strip(), kwargs[args]) for args in kwargs.keys() if args.strip()!='config']

    if 'config' in kwargs.keys():
        passed_config = kwargs['config']
    else:
        passed_config = None

    if len(kwargs.keys()) != 0:
        config_list = [f"{args_list[i][0]}={args_list[i][1]}" for i in range(0, len(args_list))]
        if passed_config != None:
            config_list.append(passed_config)

        config_str = ",".join(config_list)
    else:
        config_str = ""

    return config_str

class Node:
    def __init__(self,
                 name,
                 coordinate,
                 text=" ",
                 draw=None,
                 fill=None,
                 color=None,
                 **kwargs):

        self.name = name
        self.coordinate = coordinate
        self.text = text
        self.draw = draw
        self.fill = fill
        self.color = color
        self.kwargs = kwargs
        # set attributes from kwargs
        for key in self.kwargs.keys():
            setattr(self, key, kwargs[key])

        # aliases for specifying anchors
        if hasattr(self, 'anchor'):
            anchor_dict = { 'nw':"north west",
                            'ne':"north east",
                            "se":"south east",
                            "sw": "south west",
                            "n" : "north",
                            "s" : "south" }
            if self.anchor in anchor_dict.keys():
                self.anchor = anchor_dict[self.anchor]

    def get_code(self):
        # parse config string from class attributes 
        attrs = self.__dict__.copy()
        for attr in ['name', 'coordinate', 'text', 'fill', 'draw', 'color', 'kwargs']:
            attrs.pop(attr)
        config_str = parse_kwargs(attrs)

        if self.draw!=None:
            config_str = f"draw={self.draw}," + config_str
        if self.fill!=None:
            config_str = f"fill={self.fill}," + config_str
        if self.color!=None:
            config_str = f"{self.color}," + config_str

        if config_str.strip()!="":
            config=f"[{config_str}]"
        else:
            config=""

        code = f"\\node ({self.name}) at {self.coordinate} {config} {{{self.text}}};"
        self.code = code
        return code


class Tikz:
    def __init__(self, fp, template=STANDALONE_TEMPLATE, packages=DEFAULT_PACKAGES, tikzlibs=DEFAULT_TIKZLIBS):
        self.fp = fp # file path
        self.template = template # template
        self.packages = packages # packages
        self.tikzlibs = tikzlibs # tikzlibs

    def usepackage(self, *packages):
        """Adds the name of the packages to self.packages """
        self.packages = self.packages + list(packages)

    def usetikzlibrary(self, *libs):
        """Adds the name of the tikzlbs to self.tikzlibs """
        self.tikzlibs = self.tikzlibs + list(libs)

    def create_file(self):
        self.file = File(self.fp)
        self.file.create()

    def write(self, text, end="\n"):
        self.file.write(text, end=end)

    def init(self):
        # write template
        # load packages
        # load tikzlibs

        self.create_file() # create the file
        self.write(self.template) # write the template

        usepackages = [f"\\usepackage{{{package}}}" for package in self.packages]
        self.write(usepackages) # add packages

        usetikzlibs = [f"\\usetikzlibrary{{{lib}}}" for lib in self.tikzlibs]
        self.write(usetikzlibs) # add tikzlibs

        self.write("") # empty line

    def begin(self, env, **kwargs):
        config_str = parse_kwargs(kwargs)
        if config_str.strip()!="":
            config = f"[{config_str}]"
        else:
            config = ""

        self.write(f"\\begin{{{env}}}{config}")

    def end(self, env):
        self.write(f"\\end{{{env}}}")

    def add(self, obj):
        if isinstance(obj, Node):
            self.write(obj.get_code())

