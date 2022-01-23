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
DEFAULT_TIKZLIBS = ['arrows.meta', 'calc']

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
    'fill' : 'cyan!20!black',
    'line_width' : 'thin'
}

POINT_VALID_KWARGS = ['draw', 'fill', 'opacity', 'line_width']
POINT_NON_KEYVALS = []

PATH = {
    'line_width' : 'thick',
    'line_cap' : 'round'
}

PATH_VALID_KWARGS = ['draw', 'line_width', 'opacity', 'line_cap', 'arrows', 'style']
PATH_NON_KEYVALS = ['arrows', 'style']

ARC = {
    'line_width' : 'thick',
    'line_cap' : 'round'
}

ARC_VALID_KWARGS = ['draw', 'opacity', 'line_width', 'line_cap', 'arrows', 'style']
ARC_NON_KEYVALS = ['arrows', 'style']

ANGLE = {
    "radius"         : 0.4,
    "right_angle"    : True,
    "fill"           : "cyan",
    "fill_opacity"   : 0.5,
    "arcs"           : 1,
    "arc_sep"        : 0.1,
    "ticks"          : 0,
    "tick_len"       : 0.1,
    "tick_sep"       : None,
    "tick_color"     : "black",
    "tick_width"     : "thick",
    "label"          : None,
    "label_sep"      : 1.5
}

CIRCLE = {
    'line_width' : 'thick'
}

CIRCLE_VALID_KWARGS = ['draw', 'fill', 'opacity', 'line_width', 'style']
CIRCLE_NON_KEYVALS = ['style']

round_vec = np.vectorize(lambda x: round(x, 8))
str_vec = np.vectorize(str)

def parse_coordinate(coord):
    if isinstance(coord, (tuple, list, np.ndarray)):
        coord = col_vector(coord)
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

def get_config_str(config_dict: dict, non_keyvals: list=[], valid_kwargs=None):

    config_dict = config_dict.copy()

    # by default 'config' will be non key val 
    non_keyvals.append('config')

    # replace line width values

    if 'line_width' in config_dict.keys():
        line_width = config_dict['line_width']
        line_width_key = 'line_width'
    elif 'line width' in config_dict.keys():
        line_width = config_dict['line width']
        line_width_key = 'line width'
    else:
        line_width = None

    if line_width != None:
        if line_width.strip() in LINE_WIDTH_DICT.keys():
            line_width = LINE_WIDTH_DICT[line_width]
            config_dict[line_width_key] = line_width


    keyval_list = []
    non_keyval_list = []

    for key in config_dict.keys():

        # check valid kwargs
        if isinstance(valid_kwargs, list) and key not in valid_kwargs:
            raise ValueError(f'{key} is not a valid key word argument.')

        #  generate 'key=value' key value string list
        if key not in non_keyvals:

            # replace underscores with spaces in key names
            if isinstance(key, str):
                key_name = key.replace('_', ' ')
            else:
                key_name = str(key)

            keyval_list.append(f"{key_name}={config_dict[key]}")

        #  generate 'value' non key value string list
        else:
            non_keyval_list.append(str(config_dict[key]))

    # merge the key value and non key value string lists
    config_str_list = keyval_list + non_keyval_list
    #  generate config string
    config_str = ",".join(config_str_list)

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
        config_str = get_config_str(kwargs, non_keyvals=POINT_NON_KEYVALS, valid_kwargs=POINT_VALID_KWARGS)
        point_coord = parse_coordinate(point)

        code = f"\\filldraw[{config_str}] {point_coord} circle ({radius}pt);"
        self.write(code)

    def  draw_points(self, *points, **kwargs):
        for point in points:
            self.draw_point(point, **kwargs)

    def draw_path(self, *points, cycle=False, **kwargs):
        coords = [parse_coordinate(point) for point in points]
        if cycle:
            coords.append('cycle')

        path_str = " -- ".join(coords)

        kwargs = PATH | kwargs
        config_str = get_config_str(kwargs, non_keyvals=PATH_NON_KEYVALS, valid_kwargs=PATH_VALID_KWARGS)

        code = f"\\draw[{config_str}] " + path_str + ";"
        self.write(code)

    def draw_circle(self, circle, **kwargs):
        center = circle.center
        radius = circle.radius
        center = parse_coordinate(center)

        kwargs = CIRCLE | kwargs
        config_str = get_config_str(kwargs, non_keyvals=CIRCLE_NON_KEYVALS, valid_kwargs=CIRCLE_VALID_KWARGS)

        code = f"\\draw[{config_str}] {center} circle ({radius});"
        self.write(code)

    def draw_arc(
        self,
        pos,
        start_angle=None,
        end_angle=None,
        delta_angle=None,
        radius=None,
        **kwargs
    ):

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

        angle_str = f"start angle={start_angle}, end angle={end_angle}"
        pos_str = parse_coordinate(pos)

        kwargs = ARC | kwargs
        config_str = get_config_str(kwargs, non_keyvals=ARC_NON_KEYVALS, valid_kwargs=ARC_VALID_KWARGS)

        code = f"\\draw[radius={radius},{config_str}] {pos_str} arc [{angle_str}];"
        self.write(code)

    def _draw_angle(self, A, B, C,
        radius=0.4,
        right_angle=True,
        fill="cyan",
        fill_opacity=0.5,
        arcs=1,
        arc_sep=0.1,
        ticks=0,
        tick_len=0.1,
        tick_sep=None,
        tick_color="black",
        tick_width="thick",
        label=None,
        label_sep=1.25,
        **kwargs):

        Ax, Ay = row_vector(A)
        Bx, By = row_vector(B)
        Cx, Cy = row_vector(C)

        start_angle = round(math.degrees(math.atan2(Ay - By, Ax - Bx)), 8)%360
        delta_angle = round(angle(A, B, C), 8)
        end_angle = start_angle + delta_angle

        _radius = radius

        if not (right_angle and isclose(delta_angle, 90)):

            # fill arc
            if isinstance(fill, str):
                fill_code = f"\\fill[{fill}, opacity={fill_opacity}] "
                fill_code += f"{round(Bx, 8), round(By, 8)} "
                fill_code += f"-- ([shift=({start_angle}:{radius})]{Bx},{By}) "
                fill_code += f"arc[start angle={start_angle}, end angle={end_angle}, radius={radius}] -- cycle;"

                self.write(fill_code)

            # arcs
            for n in range(0, arcs):
                self.draw_arc(polar(radius, start_angle) + B,
                    start_angle=start_angle,
                    delta_angle=delta_angle,
                    radius=radius, **kwargs)
                radius -= arc_sep

            # default tick_sep
            if not (isinstance(tick_sep, float) or isinstance(tick_sep, int)) \
            and ((isinstance(ticks, float) or isinstance(ticks, int)) and ticks!=0):
                tick_sep = delta_angle/ticks * 0.5

            # tick marks
            for t in range(0, ticks):
                tick_angle = start_angle + delta_angle/2 - ((ticks - 1)*tick_sep)/2 + t*tick_sep
                tick_start = polar(_radius - tick_len/2, tick_angle) + B
                tick_end = polar(_radius + tick_len/2, tick_angle) + B
                self.draw_path(tick_start, tick_end, line_width=tick_width, draw=tick_color)
        else:
            # right angles
            p1 = B + polar(radius/math.sqrt(2), start_angle)
            p2 = B + polar(radius, start_angle + 45)
            p3 = B + polar(radius/math.sqrt(2), start_angle + 90)

            # fill right angle
            if isinstance(fill, str):
                fill_code = f"\\fill[{fill}, opacity={fill_opacity}] {Bx, By} "
                fill_code += f"-- ([shift=({end_angle}:{round(radius/math.sqrt(2), 8)})]{Bx}, {By}) "
                fill_code += f"-- ([shift=({(start_angle+45)%360}:{radius})]{Bx}, {By}) "
                fill_code += f"-- ([shift=({start_angle}:{round(radius/math.sqrt(2), 8)})]{Bx}, {By}) "
                fill_code += f"-- cycle;"
                self.write(fill_code)

            self.draw_path(p1, p2, p3, **kwargs)

        #labels
        if isinstance(label, str):
            # label distance = label_sep * radius
            label_pos = polar(label_sep * _radius, start_angle + delta_angle/2) + B
            label_pos_x, label_pos_y = row_vector(label_pos)
            label_code = f"\\draw {label_pos_x, label_pos_y} node[shape=circle,] {{{label}}};"
            self.write(label_code)

    def draw_angle(self, A, B, C, **kwargs):
        kwargs = ANGLE | kwargs
        self._draw_angle(A, B, C, **kwargs)

    def add(self, obj, **kwargs):
        if isinstance(obj, Node):
            self.write(obj.get_code())

        elif isinstance(obj, Circle):
            self.draw_circle(obj, **kwargs)

