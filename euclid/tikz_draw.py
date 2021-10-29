import os
from euclid.euclid2 import *
from euclid.utils.file_edit import *
from euclid.utils.math import *
from pdf2image import convert_from_path, convert_from_bytes
import PIL
import euclid.tikzrc as tikz_config 
import sympy

RND8=np.vectorize(lambda x: round(x, 8))

def clean_latex(del_tex = False):
    latex_extn = "*.aux *.log *.gz" if not del_tex else "*.tex *.aux *.log *.gz"
    os.system(f"del {latex_extn}")

def convert_pdf(path, file_type="png", dpi=600, out_path=None, name="page"):
    if out_path is None:
        out_path = os.getcwd()
    convert_from_path(str(path), dpi=dpi, output_folder=out_path, fmt=file_type, output_file=name)
    print(f"pdf2image file output path: \n{out_path}")

def resize_image(image_path, out_path, res=(1080, 1080)):
    img = PIL.Image.open(image_path)

    if type(res) == float or type(res) == int:
        RES = (mth.floor(img.size[0]*res), mth.floor(img.size[1]*res))
    else:
        RES = res

    if type(res) != float and type(res) != int and type(res) != tuple:
        raise ValueError("`res` should be an number(int/float) or a tuple of length 2")

    img_ = img.resize(RES)
    img_.save(out_path)
    del img_
    del img

def _is_nonempty_string(string):
    """ returns True if `string` is a non-empty str """
    out = False
    if string is None:
        out = False
    else:
        if type(string)==str:
            if string.strip()=="":
                out = False
            else:
                out = True
        else:
            out = False
    return out


def_preamble="""%tikz_draw
\\documentclass[11pt,a4paper, svgnames]{article}
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
standalone="""%tikz_draw
\\documentclass[tikz, border=5, svgnames]{standalone}
\\usepackage[utf8]{inputenc}
\\usepackage[english]{babel}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
%tikzlibrary
\\usetikzlibrary{arrows.meta}
%standalone preamble

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
def_arc_tick_config=""
def_arc_config=""
def_node_draw_config=""
def_node_config="anchor=north"
def_circle_config="cyan!20!black"

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class Tikz():
    def __init__(self,file_name, preamble=tikz_config.default_preamble):
        if not file_name is None:
            try:
                create_file(file_name)
                #creates the file
            except FileExistsError:
                print('WARNING: FILE ALREADY EXISTS. FILE WILL BE OVERWRITTEN')
                os.system(f"del {file_name}")
            if preamble!=None:
                write_to_file(file_name,preamble)
                #writes the preamble
        else:
            self.tex_code = ""
        self.file_name=file_name

        self.line_width_dict = {
            "ultra thin": 0.1, 
            "very thin": 0.2, 
            "thin": 0.4 ,
            "semithick": 0.6, 
            "thick": 0.8 ,
            "very thick": 1.2,
            "ultra thick": 1.6
        }


    def _get_len_value(self, value, default_unit='cm') -> str:

        if type(value)==int or type(value)==float:
            out = f"{value}{default_unit}"
        elif type(value)==str:
            value = value.strip()
            if is_number(value):
                out = f"{value}{default_unit}"
            if (value.endswith("cm") or value.endswith("pt") or value.endswith("in")) and (is_number(value[0:-2])):
                out = value
            elif value in self.line_width_dict.keys():
                out = f"{self.line_width_dict[value]}pt"
            else:
                out = None
        else:
            out = None

        return out

    def write(self,text):
        if not self.file_name is None:
            write_to_file(self.file_name,text)
        else:
            self.tex_code += self.tex_code + text + "\n"

    def read(self, show=True) -> str:
        if not self.file_name is None:
            with open(self.file_name) as file:
                out = file.read()
        else:
            out = self.tex_code

        if show:
            print(out)
        return out

    def edit(self,editor=def_editor):
        os.system(f'{editor} {self.file_name}')

    def begin(self,env,config=None):
        Config=f"[{config}]" if _is_nonempty_string(config) else ""
        self.write('\\begin{'+env+'}'+f"{Config}"+'\n')

    def end(self,env):
        self.write('\\end{'+env+'}')

    def pdf(self, shell_escape=False, batch=True):
        shescape = " -shell-escape "
        empty_string = " "
        batch_string = " -interaction=batchmode"
        os.system(f'pdflatex {batch_string if batch else empty_string}{shescape if shell_escape else empty_string}{self.file_name}')

    def png(self,dpi=500, pdf=True, clean=True, out_path=None, res=None):
        if pdf or ((self.file_name.replace(".tex", ".pdf") not in os.listdir())):
            self.pdf()

        def_out_path = os.getcwd() + "\\" + self.file_name.replace(".tex", "_images") + "\\"
        out_path = out_path if not out_path is None  else def_out_path
        file_path = os.getcwd() + "\\" + self.file_name.replace(".tex", ".pdf")

        try:
            os.mkdir(out_path)
        except FileExistsError:
            pass

        convert_pdf(file_path, dpi=dpi, out_path=out_path)
        if clean:
            clean_latex()

        if not res is None:
            for images in os.listdir(out_path):
                if images.startswith("page") and images.endswith(".png"):
                    path = os.path.join(out_path, images)
                    resize_image(path, path, res=res)
            print(f"resized {len(os.listdir(out_path))} images")

    def svg(self, clean=True):
        latex_ = f"latex {self.file_name}"
        os.system(latex_)
        dvi_name = self.file_name.replace(".tex", ".dvi")
        dvisvgm_ = f"dvisvgm {dvi_name}"
        if dvi_name in os.listdir():
            os.system(dvisvgm_)
        if clean:
            clean_latex()

    def usepackage(self, package, config=None):
        if not _is_nonempty_string(config):
            self.write("\\usepackage{"+ package +"}")
        else:
            self.write("\\usepackage["+ config +"]{"+ package +"}")

    def clip(self, x_range=[-5,5], y_range=[-5,5]):
        xmin,xmax=x_range
        ymin,ymax=y_range
        clip_code=f"\\clip {str((xmin, ymin))} rectangle {str((xmax, ymax))};"
        self.write(clip_code)

    def draw_axis(self, x_range=[-5,5], 
                        y_range=[-5,5], 
                        arrow_tip=tikz_config.axis_arrow_tip,
                        tick_labels=False,
                        axis_labels=("$x$", "$y$")):

        xmin,xmax=x_range
        ymin,ymax=y_range

        Tip=f"[{arrow_tip}]" if _is_nonempty_string(arrow_tip) else ""

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

        if (type(axis_labels)==list or type(axis_labels)==tuple):
            self.node(x_vect*x_range[1], node_config="anchor= north east", text=axis_labels[0])
            self.node(y_vect*y_range[1], node_config="anchor= north east", text=axis_labels[1])

        self.write(code)

    def draw_grid(self, x_range=[-5,5], 
                        y_range=[-5,5],
                        step=1,
                        line_width='thin',
                        line_cap='round',
                        color='RoyalBlue',
                        opacity=1,
                        style='dashed',
                        xshift=0,
                        yshift=0,
                        config=''):

        xmin,xmax=x_range
        ymin,ymax=y_range

        _line_width = self._get_len_value(line_width)
        step = self._get_len_value(step)
        xshift = self._get_len_value(xshift)
        yshift = self._get_len_value(yshift)

        Config = f"[step={step},line width={_line_width},line cap={line_cap},xshift={xshift},yshift={yshift},{color},opacity={opacity},{style},{config}]"

        grid_code=f"\\draw{Config} {str((xmin,ymin))} grid {str((xmax,ymax))};"
        self.write(grid_code)

    def draw_point(self, point,
                        radius=2,
                        color='Black',
                        opacity=1,
                        fill_color='DeepSkyBlue',
                        line_width='thin',
                        config=""):

        X,Y=row_vector(point)
        X,Y=round(X, 8),round(Y, 8)

        radius = self._get_len_value(radius, default_unit="pt")
        line_width = self._get_len_value(line_width)

        Config=f"[line width={line_width}, fill={fill_color}, draw={color},opacity={opacity},{config}]"

        draw_point_code=f"\\filldraw{Config} ({X},{Y}) circle ({radius});"
        self.write(draw_point_code)

    def draw_points(self, *points, **kwordargs):
        for point in points:
            self.draw_point(point, **kwordargs)

    def draw_vector(self, vector, start=origin,
                                  color="Black",
                                  opacity="1",
                                  line_width="thick",
                                  style="solid",
                                  line_cap="round",
                                  arrow_tip=tikz_config.arrow_tip,
                                  config=''):

        """ draws the vector `vector` with tail at `start`
            If the `vector` contains `NaN`/`None` nothing will be drawn.

            vector: np.array([[x], [y]])
         """
        X,Y=row_vector(vector)
        X,Y=round(X, 8),round(Y, 8)

        line_width = self._get_len_value(line_width, default_unit="pt")
        Config = f"[{color},opacity={opacity},line width={line_width},line cap={line_cap},{arrow_tip},{config},{style}]"

        if (not np.isnan(vector).any()) and (X!=0 or Y!=0):
            code=f"\\draw{Config} {(start[0,0], start[1,0])} -- {str((X + start[0,0], Y + start[1,0]))};"
            self.write(code)

        elif (X==0 and Y==0):
            self.draw_point(start, color=color)
        else:
            print("WARNING: Invalid value encountered.")

    def draw_path(self,*points,
                        config="",
                        line_width="thick",
                        color="Black",
                        opacity="1",
                        style="solid",
                        rounded_corners="1pt",
                        line_cap="round",
                        cycle=False):

        line_width = self._get_len_value(line_width, default_unit='pt')
        rounded_corners = self._get_len_value(rounded_corners, default_unit='pt')

        points_xy=[(round(p[0,0], 8), round(p[1,0], 8)) for p in points]
        path_string=""

        for i in range(0,len(points_xy)-1):
            path_string=path_string+f"{str(points_xy[i])} -- "

        path_string=path_string+f"{str(points_xy[-1])};" if not cycle else path_string+f"{str(points_xy[-1])} -- cycle;"

        Config=f"[{color},opacity={opacity},{style},line cap={line_cap},rounded corners={rounded_corners},line width={line_width},{config}]"

        draw_path_code=f"\\draw{Config}  "+path_string
        self.write(draw_path_code)

    def fill_path(self, *points,
                         fill_config="",
                         cycle=False,
                         fill_color=tikz_config.path_fill_color,
                         opacity=0.3):

        points_xy=[(round(p[0,0], 8), round(p[1,0], 8)) for p in points]
        path_string=""

        for i in range(0,len(points_xy)-1):
            path_string=path_string+f"{str(points_xy[i])} -- "

        path_string=path_string+f"{str(points_xy[-1])};" if not cycle else path_string+f"{str(points_xy[-1])} -- cycle;"

        Config=f"[{fill_config},{fill_color},opacity={opacity},]"

        fill_path_code=f"\\fill{Config}  "+path_string
        self.write(fill_path_code)

    def mark_segment(self, start, end, ticks=2, tick_len=0.2, tick_dist=0.2, tick_pos=None, tick_config=tikz_config.path_config):
        A = start
        B = end
        l = Line(A, B)
        m = tick_pos if not (tick_pos is None) else dist(A, mid(A, B))

        for i in range(0, ticks):
            T = m - ((ticks-1)*tick_dist)/2 + i*tick_dist
            Point = l.normt(T)
            Theta = atan2(T, tick_len/2)
            tick_A = rotate(Point, A, Theta)
            tick_B = rotate(Point, A, -Theta)
            self.draw_path(tick_A, tick_B, config=tick_config)

    def draw_segment(self, segment, ticks=0, tick_len=0.2, tick_dist=0.2, tick_pos=None, tick_config=tikz_config.path_config, end_points=False, end_point_config=tikz_config.point_config):
        start, end = segment.A, segment.B
        self.draw_path(start, end)
        self.mark_segment(start, end, ticks=ticks, tick_len=tick_len, tick_dist=tick_dist, tick_pos=tick_pos, tick_config=tick_config)
        if end_points:
            self.draw_points(start, end, config=end_point_config)

    def draw_segments(self, *segments):
        for segment in segments:
            self.draw_segment(segment)

    def draw_line(self, line,
                        config='',
                        x_range=[-5,5], y_range=[-5, 5],
                        color="Black",
                        opacity="1",
                        arrow_tip="{Stealth[round]}-{Stealth[round]}",
                        line_width="thick",
                        line_cap="round",
                        style="solid"):

        line_width = self._get_len_value(line_width, default_unit="pt")

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
        elif len(int1_)==2 and len(int2_)==0:
            p1,p2=int1_
        elif len(int1_)==0 and len(int2_)==2:
            p1,p2=int2_
        else:
            p1=int1_[0]
            p2=int2_[0]

        p1=tuple(RND8(row_vector(p1)))
        p2=tuple(RND8(row_vector(p2)))

        draw_Config=f"[{config},line width={line_width},{color},opacity={opacity},line cap={line_cap},{arrow_tip},{style},]"
        line_draw_code=f"\\draw{draw_Config} {p1} -- {p2};"

        self.write(line_draw_code)

    def draw_lines(self, *lines, **kwargs):
        for line in lines:
            self.draw_line(line, **kwargs)

    def draw_ray(self, ray, config=tikz_config.ray_config, x_range=[-5,5], y_range=[-5, 5]):
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

        draw_Config=f"[{config}]" if _is_nonempty_string(config) else ""
        ray_draw_code=f"\\draw{draw_Config} {p1} -- {p2};"

        self.write(ray_draw_code)
#TODO: CONFIG CHANGE
    def draw_angle(self, A, B, C, config=def_arc_config, radius=0.5, fill_config=def_arc_fill_config, right_angle=True, arcs=1, arc_dist=0.075, ticks=0, tick_dtheta=None, tick_len=0.2, tick_config=def_arc_tick_config, no_fill=False):

        Angle=angle(A, B, C)
        Bx, By=RND8(row_vector(B))

        start_angle=atan2((A-B)[0,0], (A-B)[1,0])
        end_angle= start_angle + Angle

        Angle, start_angle, end_angle = round(Angle, 8), round(start_angle, 8), round(end_angle, 8)

        draw_Config=f"[{config}]" if _is_nonempty_string(config) else ""
        fill_Config=f"[{fill_config}]" if _is_nonempty_string(fill_config) else ""
        tick_Config=f"[{tick_config}]" if _is_nonempty_string(tick_config) else ""

        if (not right_angle) or (not isclose(Angle, 90)):
            draw_angle_code=f"\\draw{draw_Config}  ([shift=({start_angle}:{radius})]{Bx},{By}) arc[start angle={start_angle}, end angle={end_angle}, radius={radius}];"
            if not no_fill:
                fill_angle_code=f"\\fill{fill_Config} {Bx,By} -- ([shift=({start_angle}:{radius})]{Bx},{By}) arc[start angle={start_angle}, end angle={end_angle}, radius={radius}] -- cycle;"

            n_arc = arcs
            d_arc = arc_dist
            arc_code = ""
            for i in range(1, n_arc):
                arc_code += f"\\draw{draw_Config}  ([shift=({start_angle}:{radius - i * d_arc})]{Bx},{By}) arc[start angle={start_angle}, end angle={end_angle}, radius={radius - i * d_arc}];\n"

            n_ticks = ticks
            d_ticks = tick_dtheta
            k_ticks = 0.5
            if d_ticks is None and n_ticks!=0:
                d_ticks = k_ticks * (Angle)/(n_ticks)
            tick_code = ""
            for j in range(0, n_ticks):
                tick_code += f"\\draw{tick_Config} ([shift=({start_angle + Angle/2 - ((n_ticks-1) * d_ticks)/2 + (j * d_ticks)}:{radius - tick_len/2})]{Bx},{By}) -- ([shift=({start_angle + Angle/2 - ((n_ticks-1) * d_ticks)/2 + (j * d_ticks) }:{radius + tick_len/2})]{Bx},{By});\n"

        else:
            draw_angle_code=f"\\draw{draw_Config} {Bx, By} -- ([shift=({end_angle}:{radius/sqrt(2)})]{Bx}, {By}) -- ([shift=({(start_angle+Angle/2)%360}:{radius})]{Bx}, {By}) -- ([shift=({start_angle}:{radius/sqrt(2)})]{Bx}, {By}) -- cycle;"
            if not no_fill:
                fill_angle_code=f"\\fill{fill_Config} {Bx, By} -- ([shift=({end_angle}:{radius/sqrt(2)})]{Bx}, {By}) -- ([shift=({(start_angle+Angle/2)%360}:{radius})]{Bx}, {By}) -- ([shift=({start_angle}:{radius/sqrt(2)})]{Bx}, {By}) -- cycle;"
            arc_code=""
            tick_code=""

        if not no_fill:
            self.write(fill_angle_code)
        self.write(draw_angle_code)
        self.write(arc_code)
        self.write(tick_code)

    def draw_circle(self, circle, config=tikz_config.circle_config):
        Cx, Cy= RND8(row_vector(circle.center))
        radius= round(circle.radius, 8)

        draw_Config=f"[{config}]" if _is_nonempty_string(config) else ""

        draw_circle_code=f"\\draw{draw_Config} ({Cx}, {Cy}) circle ({radius});"
        self.write(draw_circle_code)

    def node(self, position, node_config=def_node_config , config=def_node_draw_config, text=""):
        X,Y=row_vector(position)
        X,Y=round(X, 8), round(Y, 8)

        Config=f"[{config}]" if _is_nonempty_string(config) else ""
        node_Config=f"[{node_config}]" if _is_nonempty_string(node_config) else ""

        node_code=f"\\draw{Config} {X,Y} node {node_Config} "+"{"+f"{text}"+"};"

        self.write(node_code)

    def smooth_plot_from_file(self, file_path, config=tikz_config.path_config, plot_config=""):

        Config=f"[{config}]" if _is_nonempty_string(config) else ""
        plot_Config=f"{plot_config}," if _is_nonempty_string(plot_config) else ""

        code = f"\\draw{Config} plot[{plot_Config} smooth] file " + "{"+ file_path.replace("\\","/") +"};"
        self.write(code)

    def smooth_plot_from_points(self, points, config=def_path_config, plot_config=""):

        Config=f"[{config}]" if _is_nonempty_string(config) else ""
        plot_Config=f"{plot_config}," if _is_nonempty_string(plot_config) else ""

        points_string=""
        for point in points:
            points_string += str(tuple(point)) + " "

        code = f"\\draw{Config} plot[{plot_Config} smooth] " + "{"+ points_string +"};"
        self.write(code)

    def draw_lagrange_polynomial_from_table(self, points, table_name, x_range=[-10, 10], samples=100, config=tikz_config.path_config, plot_config=""):
        func = get_lagrange_polynomial_as_func(points)

        for x in np.linspace(x_range[0], x_range[1], samples):
            with open(table_name, "a") as file:
                file.write(f"{x} {func(x)} \n")

        self.smooth_plot_from_file(table_name, config=config, plot_config=plot_config)

    def draw_vector_field_from_func(self, func, x_range=[-10, 10], y_range=[-10, 10], grad=("red", "blue"), grad_range=(0, 100), 
                                    config=tikz_config.path_config, len_range=(0, 1), arrow_tip=tikz_config.arrow_tip, normalize=True):
        points=[]
        for x in range(x_range[0], x_range[1] + 1):
            for y in range(y_range[0], y_range[1] + 1):
                points.append(col_vector([x, y]))

        vectors_ = [func(point) for point in points if not np.isnan(func(point)).any()]
        length = lambda vector: dist(col_vector([0, 0]), vector)
        maxR = length(max(vectors_, key = length))

        if normalize:
            def normalize_func(vector):
                minL, maxL = len_range
                vnorm = norm(vector)
                V = mth.sqrt( (maxL - minL)*length(vector)/maxR + minL ) * vnorm
                return V
        else:
            normalize_func = lambda x: x

        def get_grad_value(vector):
            minG, maxG = grad_range
            n = (maxG - minG)*length(vector)/maxR + minG
            return n

        for p in points:
            V = func(p)
            grad_code = f"{grad[1]}!{get_grad_value(V)}!{grad[0]}"
            Config = f"{grad_code}, {config}" if _is_nonempty_string(config) else f"{grad_code}"
            self.draw_vector(normalize_func(V), start=p, config=Config, arrow_tip=arrow_tip)

    def pgfplots_begin_axis(self, config=None, show_axis_lines=False, x_range=(-10, 10), y_range=(-10, 10)):
        if not show_axis_lines:
            dont_show_axis_code = "axis x line=none,axis y line=none,xticklabels=\\empty,yticklabels=\\empty"
        else:
            dont_show_axis_code = ""

        Config = ""
            
        if show_axis_lines:
            if not ((config is None) or config.strip()==""):
                Config = f"[{config}, xmin={x_range[0]}, xmax={x_range[1]},ymin={y_range[0]}, ymax={y_range[1]}, xshift={x_range[0]}cm, yshift={y_range[0]}cm, x=1cm,  y=1cm]"
            else:
                Config = f"[xmin={x_range[0]}, xmax={x_range[1]},ymin={y_range[0]}, ymax={y_range[1]}, xshift={x_range[0]}cm, yshift={y_range[0]}cm, x=1cm,  y=1cm]"
        else:
            if not ((config is None) or config.strip()==""):
                Config = f"[{dont_show_axis_code},{config}, xmin={x_range[0]}, xmax={x_range[1]},ymin={y_range[0]}, ymax={y_range[1]}, xshift={x_range[0]}cm, yshift={y_range[0]}cm, x=1cm,  y=1cm]"
            else: 
                Config = f"[{dont_show_axis_code},xmin={x_range[0]}, xmax={x_range[1]},ymin={y_range[0]}, ymax={y_range[1]}, xshift={x_range[0]}cm, yshift={y_range[0]}cm, x=1cm,  y=1cm]"

        self.write("\\begin{axis}"+Config)

    def pgfplots_end_axis(self):
        self.write("\\end{axis}")

    def pgfplots_addplot_from_expr(self, expr, samples=500, domain=(-5, 5), config=tikz_config.path_config):
        Expr = expr.replace("**", "^")
        Config = f"[{config}, samples={samples}, domain={domain[0]}:{domain[1]}]" if _is_nonempty_string(config) else f"[samples={samples}, domain={domain[0]}:{domain[1]}]"

        code = f"\\addplot {Config} " + "{"+f"{Expr}"+"};"
        self.write(code)

    def pgfplots_lagrange_polynomial_from_points(self, points):
        def get_lagrange_poly_expr():
            x = sympy.symbols("x")
            expr = str(sympy.interpolate(points, x))
            return expr
        self.pgfplots_addplot_from_expr(get_lagrange_poly_expr())

    def get_texcode(self):
        file_name = self.file_name
        file_content = File(file_name).readlines()
        begin_index, end_index = None, None
        for i in range(0, len(file_content)):
            if "\\begin{document}" in file_content[i]:
                begin_index = i
            if "\\end{document}" in file_content[i]:
                end_index = i

        texcode_ = file_content[begin_index+1: end_index]
        texcode = ""
        for s in texcode_:
            texcode += s

        return texcode



