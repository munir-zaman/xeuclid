import os
from euclid.euclid2 import *
from euclid.utils.file_edit import *
from euclid.utils.math import *
from pdf2image import convert_from_path, convert_from_bytes
import PIL
import euclid.tikz_config as tikz_config 

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


def_preamble="""%tikz_draw
\\documentclass[11pt,a4paper]{article}
\\usepackage[utf8]{inputenc}
\\usepackage[english]{babel}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage[left=2cm,right=2cm,top=2cm,bottom=2cm]{geometry}
\\usepackage[svgnames]{xcolor}
\\usepackage{tikz}
%tikzlibrary
\\usetikzlibrary{arrows.meta}
%preamble

"""
standalone="""%tikz_draw
\\documentclass[tikz, border=5]{standalone}
\\usepackage[utf8]{inputenc}
\\usepackage[english]{babel}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage[svgnames]{xcolor}
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

    def write(self,text):
        if not self.file_name is None:
            write_to_file(self.file_name,text)
        else:
            self.tex_code += self.tex_code + text + "\n"

    def read(self, show=True):
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
        Config=f"[{config}]" if (not (config is None) and config.strip()!="") else ""
        self.write('\\begin{'+env+'}'+f"{Config}"+'\n')

    def end(self,env):
        self.write('\\end{'+env+'}')

    def pdf(self):
        os.system(f'{pdflatex_command} -interaction=batchmode {self.file_name}')

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

    def clip(self, x_range=[-5,5], y_range=[-5,5]):
        xmin,xmax=x_range
        ymin,ymax=y_range
        clip_code=f"\\clip {str((xmin, ymin))} rectangle {str((xmax, ymax))};"
        self.write(clip_code)

    def draw_axis(self, x_range=[-5,5], y_range=[-5,5], arrow_tip=tikz_config.axis_arrow_tip ,tick_labels=False):
        xmin,xmax=x_range
        ymin,ymax=y_range

        Tip=f"[{arrow_tip}]" if (not (arrow_tip is None) and arrow_tip.strip()!="") else ""

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

    def draw_grid(self, x_range=[-5,5], y_range=[-5,5], config=tikz_config.grid_config):
        xmin,xmax=x_range
        ymin,ymax=y_range

        Config=f"[{config}]" if (not (config is None) and config.strip()!="") else ""
        grid_code=f"\\draw{Config} {str((xmin,ymin))} grid {str((xmax,ymax))};"
        self.write(grid_code)

    def draw_point(self, point, config=tikz_config.point_config, radius=2):
        X,Y=row_vector(point)
        X,Y=round(X, 8),round(Y, 8)

        Config=f"[{config}]" if (not (config is None) and config.strip()!="") else ""
        draw_point_code=f"\\filldraw{Config} ({X},{Y}) circle ({radius}pt);"
        self.write(draw_point_code)

    def draw_vector(self,vector,start=origin, config=tikz_config.vector_config, arrow_tip=tikz_config.arrow_tip):
        X,Y=row_vector(vector)
        X,Y=round(X, 8),round(Y, 8)

        Config=f"[{config},{arrow_tip}]" if (not (config in None) and config.strip()!="") else f"[{Tip}]"
        code=f"%vector [{X}, {Y}]\n\\draw{Config} {(start[0,0], start[1,0])} -- {str((X + start[0,0], Y + start[1,0]))};"
        self.write(code)

    def draw_path(self,*points, config=tikz_config.path_config, cycle=False):
        points_xy=[(round(p[0,0], 8), round(p[1,0], 8)) for p in points]
        path_string=""

        for i in range(0,len(points_xy)-1):
            path_string=path_string+f"{str(points_xy[i])} -- "

        path_string=path_string+f"{str(points_xy[-1])};" if not cycle else path_string+f"{str(points_xy[-1])} -- cycle;"

        Config=f"[{config}]" if (not (config is None) and config.strip()!="") else ""

        draw_path_code=f"\\draw{Config}  "+path_string
        self.write(draw_path_code)

    def fill_path(self, *points, fill_config=tikz_config.path_fill_config, cycle=False, fill_color=tikz_config.path_fill_color):
        points_xy=[(round(p[0,0], 8), round(p[1,0], 8)) for p in points]
        path_string=""

        for i in range(0,len(points_xy)-1):
            path_string=path_string+f"{str(points_xy[i])} -- "

        path_string=path_string+f"{str(points_xy[-1])};" if not cycle else path_string+f"{str(points_xy[-1])} -- cycle;"

        Config=f"[{fill_config},{fill_color}]" if (not (fill_config is None) and fill_config.strip()!="") else f"{fill_color}"

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

    def draw_points(self, *points, config=tikz_config.point_config, radius=2):
        for point in points:
            self.draw_point(point, config=config, radius=radius)

    def draw_line(self, line, config=tikz_config.line_config, x_range=[-5,5], y_range=[-5, 5]):
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

        draw_Config=f"[{config}]" if (not (config is None) and config.strip()!="") else ""
        line_draw_code=f"\\draw{draw_Config} {p1} -- {p2};"

        self.write(line_draw_code)

    def draw_lines(self, *lines, config=tikz_config.line_config, x_range=[-5,5], y_range=[-5,5]):
        for line in lines:
            self.draw_line(line, config=config, x_range=x_range, y_range=y_range)

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

        draw_Config=f"[{config}]" if (not (config is None) and config.strip()!="") else ""
        ray_draw_code=f"\\draw{draw_Config} {p1} -- {p2};"

        self.write(ray_draw_code)
#TODO: CONFIG CHANGE
    def draw_angle(self, A, B, C, config=def_arc_config, radius=0.5, fill_config=def_arc_fill_config, right_angle=True, arcs=1, arc_dist=0.075, ticks=0, tick_dtheta=None, tick_len=0.2, tick_config=def_arc_tick_config, no_fill=False):

        Angle=angle(A, B, C)
        Bx, By=RND8(row_vector(B))

        start_angle=atan2((A-B)[0,0], (A-B)[1,0])
        end_angle= start_angle + Angle

        Angle, start_angle, end_angle = round(Angle, 8), round(start_angle, 8), round(end_angle, 8)

        draw_Config=f"[{config}]" if (not isnone(config) and config!="") else ""
        fill_Config=f"[{fill_config}]" if (not isnone(fill_config) and fill_config!="") else ""
        tick_Config=f"[{tick_config}]" if (not isnone(tick_config) and tick_config!="") else ""

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

        draw_Config=f"[{config}]" if (not (config is None) and config.strip()!="") else ""

        draw_circle_code=f"\\draw{draw_Config} ({Cx}, {Cy}) circle ({radius});"
        self.write(draw_circle_code)

    def node(self, position, node_config=def_node_config , config=def_node_draw_config, text=""):
        X,Y=row_vector(position)
        X,Y=round(X, 8), round(Y, 8)

        Config=f"[{config}]" if (not isnone(config) and config!="") else ""
        node_Config=f"[{node_config}]" if (not isnone(node_config) and node_config!="") else ""

        node_code=f"\\draw{Config} {X,Y} node {node_Config} "+"{"+f"{text}"+"};"

        self.write(node_code)

    def smooth_plot_from_file(self, file_path, config=tikz_config.path_config, plot_config=""):

        Config=f"[{config}]" if (not isnone(config) and config!="") else ""
        plot_Config=f"{plot_config}," if (not (plot_config is None) and plot_config.strip()!="") else ""

        code = f"\\draw{Config} plot[{plot_Config} smooth] file " + "{"+ file_path.replace("\\","/") +"};"
        self.write(code)

    def smooth_plot_from_points(self, points, config=def_path_config, plot_config=""):

        Config=f"[{config}]" if (not isnone(config) and config!="") else ""
        plot_Config=f"{plot_config}," if (not (plot_config is None) and plot_config.strip()!="") else ""

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
