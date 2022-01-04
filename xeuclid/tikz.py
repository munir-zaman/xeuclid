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

LINE_WIDTH_DICT = {
    "ultra thin": 0.1,
    "very thin": 0.2,
    "thin": 0.4 ,
    "semithick": 0.6,
    "thick": 0.8 ,
    "very thick": 1.2,
    "ultra thick": 1.6
}

POINT = {
    'draw' : 'Black',
    'fill' : 'DeepSkyBlue',
    'line_width' : 'thin'
}

PATH = {
    'line_width' : 'thick',
    'line_cap' : 'round'
}

ARC = {
    'line_width' : 'thick',
    'line_cap' : 'round'
}

CIRCLE = {
    'line_width' : 'thick'
}

def parse_coordinate(coord):
    if isinstance(coord, (tuple, list, np.ndarray)):
        coord = col_vector(coord)
        round_vec = np.vectorize(round)
        str_vec = np.vectorize(str)
        coord = str_vec(round_vec(coord))
        coord = list(row_vector(coord))
        parsed_coord_ = ",".join(coord)
        parsed_coord = f"({parsed_coord_})"

    elif isinstance(coord, str):
        parsed_coord = coord.strip()

    else:
        raise TypeError('`coord` must be of type list, tuple, np.ndarray or str.')

    return parsed_coord

def parse_kwargs(kwargs, add_to_config : list=[]):

    # handle line_width values

    if 'line_width' in kwargs.keys():
        line_width = kwargs['line_width']
        line_width_key = 'line_width'
    elif 'line width' in kwargs.keys():
        line_width = kwargs['line width']
        line_width_key = 'line width'
    else:
        line_width = None

    if line_width != None:
        if line_width.strip() in LINE_WIDTH_DICT.keys():
            line_width = LINE_WIDTH_DICT[line_width]
            kwargs[line_width_key] = line_width


    args_list = [(args.replace("_", " ").strip(), kwargs[args]) for args in kwargs.keys() if args.strip()!='config']

    passed_config = ""
    if 'config' in kwargs.keys():
        passed_config = kwargs['config']

    if len(kwargs.keys()) != 0:
        config_list = [f"{args_list[i][0]}={args_list[i][1]}" for i in range(0, len(args_list))]
        config_list.append(passed_config)

        if isinstance(add_to_config, list):
            config_list += add_to_config
        elif isinstance(add_to_config, str):
            config_list.append(add_to_config)

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

    def begin(self, *envs, **kwargs):
        if len(envs) > 0:
            for env in envs:
                config_str = parse_kwargs(kwargs)
                if config_str.strip()!="":
                    config = f"[{config_str}]"
                else:
                    config = ""

                self.write(f"\\begin{{{env}}}{config}")
        else:
            self.begin('document', 'tikzpicture')

    def end(self, *envs):
        if len(envs) > 0:
            for env in envs:
                self.write(f"\\end{{{env}}}")
        else:
            self.end('tikzpicture', 'document')

    def pdf(self, shell_escape=False, batchmode=True):
        """Generates ``pdf`` file using ``pdflatex``. """
        shell_escape_string = " -shell-escape"
        batchmode_string = " -interaction=batchmode"
        cmd = "pdflatex"

        if shell_escape:
            cmd += shell_escape_string
        if batchmode:
            cmd += batchmode_string

        cmd += " " + self.fp
        os.system(cmd)

    def draw_point(self, point, radius=2, **kwargs):
        kwargs = POINT | kwargs # use default values if not provided
        parsed_config = parse_kwargs(kwargs)
        config_str = f"[{parsed_config}]"
        point_coord =parse_coordinate(point)
        code = f"\\filldraw{config_str} {point_coord} circle ({radius}pt);"
        self.write(code)

    def  draw_points(self, *points, **kwargs):
        for point in points:
            self.draw_point(point, **kwargs)

    def draw_path(self, *points, cycle=False, arrows="", **kwargs):
        coords = [parse_coordinate(point) for point in points]
        path_str = coords[0]
        for coord in coords[1:]:
            path_str += " -- " + coord
        if cycle:
            path_str += " -- cycle"

        kwargs = PATH | kwargs
        parsed_config = parse_kwargs(kwargs, add_to_config=[arrows])
        config_str = f"[{parsed_config}]"

        code = f"\\draw{config_str} " + path_str + ";"
        self.write(code)

    def draw_circle(self, circle, **kwargs):
        center = circle.center
        radius = circle.radius
        center = parse_coordinate(center)

        kwargs = CIRCLE | kwargs
        parsed_config = parse_kwargs(kwargs)
        config_str = f"[{parsed_config}]"
        code = f"\\draw{config_str} {center} circle ({radius});"
        self.write(code)

    def draw_arc(self, pos, start_angle=None, end_angle=None, delta_angle=None, radius=None, arrows="", **kwargs):

        if (start_angle is not None) and (end_angle is not None):
            start_angle, end_angle = start_angle, end_angle
        # if end_angle is not given set end_angle = start_angle + delta_angle
        elif (start_angle is not None) and (end_angle is None) and (delta_angle is not None):
            end_angle = start_angle + delta_angle
        # if start_angle is not given set start_angle = end_angle - delta_angle
        elif (start_angle is None) and (end_angle is not None) and (delta_angle is not None):
            start_angle = end_angle - delta_angle
        else:
            ValueError("Could not compute start_angle and end_angle")

        if radius is None:
            radius = round(dist(col_vector([0, 0]), col_vector(pos)), 8)

        angle_str = f"[start angle={start_angle}, end angle={end_angle}]"
        pos_str = parse_coordinate(pos)

        kwargs = ARC | kwargs
        parsed_config = parse_kwargs(kwargs, add_to_config=[arrows])
        config_str = f"[radius={radius},{parsed_config}]"

        code = f"\\draw{config_str} {pos_str} arc {angle_str};"
        self.write(code)

    def add(self, obj, **kwargs):
        if isinstance(obj, Node):
            self.write(obj.get_code())

        elif isinstance(obj, Circle):
            self.draw_circle(obj, **kwargs)

