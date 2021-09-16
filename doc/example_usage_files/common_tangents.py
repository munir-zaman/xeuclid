from euclid2 import *
from tikz_draw import *

A=origin
B=col_vector([3,3])

c1=Circle(A, 2)
c2=Circle(B, 1)

T=common_tangents(c1, c2)

tikz=Tikz('common_tangents.tex')

tikz.begin('document')
tikz.begin('tikzpicture')

tikz.draw_circle(c1, config="cyan!50!black, thick")
tikz.draw_circle(c2, config="cyan!50!black, thick")

tikz.draw_line(T[0], x_range=[-8,8], y_range=[-8,8])
tikz.draw_line(T[1], x_range=[-8,8], y_range=[-8,8])
tikz.draw_line(Line(c1.center, c2.center), config="gray", x_range=[-8,8], y_range=[-8,8])
tikz.draw_path(c1.center, T[0] & T[1])

c1T0=intersection_line_circle(T[0], c1)[0]
c2T0=intersection_line_circle(T[0], c2)[0]

tikz.draw_path(c1.center, c1T0)
tikz.draw_path(c2.center, c2T0)

tikz.draw_points(c1.center, c2.center, T[0] & T[1], c1T0, c2T0)

tikz.node(c1.center, node_config="anchor=north, inner sep=8pt", text="$C_2$")
tikz.node(c2.center, node_config="anchor=north, inner sep=8pt", text="$C_1$")

tikz.node(c2T0, node_config="anchor=south, inner sep=8pt", text="$T_1$")
tikz.node(c1T0, node_config="anchor=south, inner sep=8pt", text="$T_2$")

tikz.node(mid(c2T0, c2.center)-col_vector((0,0.25)), node_config="anchor=east, inner sep=2pt", text="$r_1$")
tikz.node(mid(c1T0, c1.center), node_config="anchor=east, inner sep=4pt", text="$r_2$")

tikz.node(T[0] & T[1], node_config="anchor=north, inner sep=5pt", text="$P$")

tikz.end('tikzpicture')
tikz.end('document')

tikz.pdf()


