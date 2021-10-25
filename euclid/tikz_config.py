
default_preamble="""%tikz_draw
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


point_config = "fill=cyan!20!black, draw=black"
line_config = "{Stealth[round]}-{Stealth[round]}, thick"
ray_config = "-{Stealth[round]}, line cap=round"

path_config = "thick, line cap=round"
path_fill_config="opacity=0.3"
path_fill_color = "DeepSkyBlue"

vector_config = "-{Stealth[round]}, thick, line cap=round"
circle_config = "RoyalBlue, thick"
grid_config="LightSteelBlue, dashed"

arrow_tip = "-{Stealth[round]}"

axis_arrow_tip = "{Stealth[round]}-{Stealth[round]}"
